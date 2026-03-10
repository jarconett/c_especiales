import os
import glob
import base64
import hashlib
import pickle
import json
from datetime import datetime, timezone
from typing import List, Dict, Tuple

import pandas as pd
import requests


DF_CHUNK_BYTES = 42 * 1024 * 1024  # mismo tamaño de chunk que en streamlit_app


def _parse_repo_url(repo_url: str) -> Tuple[str, str]:
    """Parsea la URL del repositorio y retorna (owner, repo)."""
    import re as _re
    if repo_url.count("/") == 1 and "/" in repo_url:
        parts = repo_url.split("/")
        return parts[0], parts[1]
    else:
        m = _re.match(r"https?://github.com/([^/]+)/([^/]+)", repo_url)
        if not m:
            return "", ""
        return m.group(1), m.group(2).replace(".git", "")


def _get_github_headers() -> Tuple[Dict[str, str], str]:
    """Obtiene headers y token para la API de GitHub desde GITHUB_TOKEN."""
    token = os.getenv("GITHUB_TOKEN", "").strip()
    headers = {"Accept": "application/vnd.github.v3+json"}
    if token:
        headers["Authorization"] = f"token {token}"
    return headers, token


def _get_file_sha(files: List[dict]) -> dict:
    """Crea un índice de archivos similar al usado en la app."""
    file_index = {}
    for file_info in files:
        filename = file_info.get("name", "")
        folder = file_info.get("folder", "").lower()
        content = file_info.get("content", "")
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        file_index[filename] = {
            "sha": content_hash,
            "folder": folder,
            "size": len(content),
        }
    return file_index


def parse_transcription_text(name: str, text: str, folder: str = "") -> pd.DataFrame:
    """Parsea un archivo de transcripción al formato de la app."""
    import re as _re

    pattern = _re.compile(r"\[([^\]]+)\]\s*(.*?)((?=\[)|$)", _re.S)
    rows = []
    for idx, m in enumerate(pattern.finditer(text)):
        speaker = m.group(1).strip()
        content = _re.sub(r"\s+", " ", m.group(2).strip().replace("\r\n", "\n")).strip()
        rows.append(
            {
                "file": name,
                "speaker": speaker,
                "text": content,
                "block_index": idx,
                "folder": folder.lower(),
            }
        )
    if not rows:
        cleaned = _re.sub(r"\s+", " ", text).strip()
        rows.append(
            {
                "file": name,
                "speaker": "UNKNOWN",
                "text": cleaned,
                "block_index": 0,
                "folder": folder.lower(),
            }
        )
    return pd.DataFrame(rows)


def build_transcriptions_dataframe(files: List[dict]) -> pd.DataFrame:
    """Construye el DataFrame como en la app."""
    dfs = [
        parse_transcription_text(f["name"], f["content"], f.get("folder", ""))
        for f in files
    ]
    if dfs:
        df = pd.concat(dfs, ignore_index=True)
        return df
    else:
        return pd.DataFrame(
            columns=["file", "speaker", "text", "block_index", "folder"]
        )


def _save_dataframe_to_github(
    repo_url: str, df: pd.DataFrame, file_index: dict, path: str = "data"
) -> Tuple[bool, str]:
    """
    Guarda el DataFrame serializado y el índice en GitHub en partes,
    imitando el comportamiento de streamlit_app.
    """
    owner, repo = _parse_repo_url(repo_url)
    if not owner or not repo:
        return False, "URL de repositorio no válida"

    headers, token = _get_github_headers()
    if not token:
        return False, "Se requiere GITHUB_TOKEN para guardar el DataFrame"

    try:
        df_bytes = pickle.dumps(df)
        base_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"

        # Dividir en chunks
        chunks = []
        offset = 0
        while offset < len(df_bytes):
            chunk = df_bytes[offset : offset + DF_CHUNK_BYTES]
            chunks.append(chunk)
            offset += len(chunk)

        n_parts = len(chunks)

        # Guardar cada parte
        for i, chunk in enumerate(chunks):
            chunk_b64 = base64.b64encode(chunk).decode("utf-8")
            part_url = f"{base_url}/transcripciones_df_part_{i}.pkl"
            part_data = {
                "message": f"Actualizar DataFrame parte {i+1}/{n_parts} ({len(df)} filas)",
                "content": chunk_b64,
            }
            resp = requests.get(part_url, headers=headers)
            if resp.status_code == 200:
                part_data["sha"] = resp.json().get("sha")
            put_resp = requests.put(part_url, headers=headers, json=part_data, timeout=60)
            if put_resp.status_code not in [200, 201]:
                return (
                    False,
                    f"Error al guardar parte {i+1}: {put_resp.status_code} - {put_resp.text[:200]}",
                )

        # Eliminar partes sobrantes si antes había más
        part_idx = n_parts
        while True:
            part_url = f"{base_url}/transcripciones_df_part_{part_idx}.pkl"
            resp = requests.get(part_url, headers=headers)
            if resp.status_code != 200:
                break
            sha = resp.json().get("sha")
            del_resp = requests.delete(
                part_url,
                headers=headers,
                json={"message": "Eliminar parte obsoleta", "sha": sha},
                timeout=30,
            )
            part_idx += 1

        # Índice
        index_data = {
            "file_index": file_index,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_files": len(file_index),
            "df_rows": len(df),
            "df_parts": n_parts,
        }
        index_json = json.dumps(index_data, indent=2)
        index_base64 = base64.b64encode(index_json.encode("utf-8")).decode("utf-8")
        index_file_url = f"{base_url}/transcripciones_index.json"
        index_data_put = {
            "message": f"Actualizar índice ({len(file_index)} archivos, DataFrame en {n_parts} partes)",
            "content": index_base64,
        }
        index_resp = requests.get(index_file_url, headers=headers)
        if index_resp.status_code == 200:
            index_data_put["sha"] = index_resp.json().get("sha")
        index_put = requests.put(
            index_file_url, headers=headers, json=index_data_put, timeout=30
        )
        if index_put.status_code not in [200, 201]:
            return (
                False,
                f"Error al guardar índice: {index_put.status_code} - {index_put.text[:200]}",
            )

        return True, ""

    except Exception as e:
        return False, f"Error al guardar: {str(e)}"


def load_local_txts(base_dir: str, folder_name: str) -> List[dict]:
    """Lee .txt locales desde una carpeta (transcripciones o spoti)."""
    folder_path = os.path.join(base_dir, folder_name)
    files: List[dict] = []
    if not os.path.isdir(folder_path):
        return files
    for path in glob.glob(os.path.join(folder_path, "*.txt")):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        name = os.path.basename(path)
        files.append({"name": name, "content": content, "folder": folder_name.lower()})
    return files


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Regenera el DataFrame de transcripciones desde archivos locales "
            "y lo sube en partes a GitHub (mismo formato que la app Streamlit)."
        )
    )
    parser.add_argument(
        "--repo",
        required=True,
        help="URL del repo GitHub (ej: https://github.com/jarconett/c_especiales/ o jarconett/c_especiales)",
    )
    parser.add_argument(
        "--data-path",
        default="data",
        help="Ruta en el repo donde guardar el DataFrame (por defecto: data)",
    )
    parser.add_argument(
        "--base-dir",
        default=".",
        help="Directorio base local donde están las carpetas 'transcripciones' y 'spoti' (por defecto: .)",
    )

    args = parser.parse_args()

    headers, token = _get_github_headers()
    if not token:
        print("ERROR: Debes definir GITHUB_TOKEN en el entorno para subir a GitHub.")
        return

    base_dir = os.path.abspath(args.base_dir)
    print(f"Usando base_dir = {base_dir}")

    files_trans = load_local_txts(base_dir, "transcripciones")
    files_spoti = []
    for variant in ("spoti", "Spoti", "SPOTI"):
        files_spoti = load_local_txts(base_dir, variant)
        if files_spoti:
            break

    all_files = files_trans + files_spoti
    if not all_files:
        print(
            "No se encontraron archivos .txt en las carpetas locales 'transcripciones' ni 'spoti'."
        )
        return

    print(
        f"Encontrados {len(all_files)} archivos locales "
        f"({len(files_trans)} en transcripciones, {len(files_spoti)} en spoti)."
    )

    df = build_transcriptions_dataframe(all_files)
    print(f"DataFrame construido con {len(df)} filas.")

    file_index = _get_file_sha(all_files)
    ok, err = _save_dataframe_to_github(args.repo, df, file_index, path=args.data_path)
    if ok:
        print(
            f"✅ DataFrame subido correctamente a GitHub en {args.data_path} "
            f"({len(df)} filas, {len(file_index)} archivos, en partes)."
        )
    else:
        print(f"❌ Error al guardar en GitHub: {err}")


if __name__ == "__main__":
    main()

