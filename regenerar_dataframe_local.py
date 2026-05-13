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

TRANSCRIPTS_TXT_REPO_DEFAULT = "jarconett/dbcesp"


def _txt_repo_arg_to_owner_repo(s: str) -> str:
    """Normaliza 'owner/repo' o URL https://github.com/... a 'owner/repo'."""
    s = (s or "").strip()
    if not s:
        return ""
    if s.count("/") == 1 and "/" in s:
        return s.replace(".git", "").strip("/")
    import re as _re

    m = _re.match(r"https?://github.com/([^/]+)/([^/]+)", s)
    if m:
        return f"{m.group(1)}/{m.group(2).replace('.git', '')}"
    return s


def _transcripts_txt_repo_from_env(cli_arg: str) -> str:
    env = (
        os.getenv("TRANSCRIPTS_TXT_REPO", "").strip()
        or os.getenv("TRANSCRIPTS_CONTENT_REPO_URL", "").strip()
    )
    if env:
        return _txt_repo_arg_to_owner_repo(env)
    return _txt_repo_arg_to_owner_repo(cli_arg or TRANSCRIPTS_TXT_REPO_DEFAULT)


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


def _get_db_token_headers_cli() -> Tuple[Dict[str, str], str]:
    """Token para leer el repo privado jarconett/dbcesp (misma variable que Streamlit: DB_TOKEN)."""
    token = os.getenv("DB_TOKEN", "").strip()
    if token.startswith('"') and token.endswith('"'):
        token = token[1:-1].strip()
    if token.startswith("'") and token.endswith("'"):
        token = token[1:-1].strip()
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
        sha_val = (file_info.get("sha") or "").strip()
        if sha_val:
            file_index[filename] = {
                "sha": sha_val,
                "folder": folder,
                "size": int(file_info.get("size", 0) or 0),
            }
        else:
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
    Guarda el DataFrame serializado y el índice en el repo privado (carpeta path, p. ej. data/).
    Requiere DB_TOKEN con permiso de escritura en ese repo.
    """
    owner, repo = _parse_repo_url(repo_url)
    if not owner or not repo:
        return False, "URL de repositorio no válida"

    headers, token = _get_db_token_headers_cli()
    if not token:
        return False, "Se requiere DB_TOKEN para guardar el DataFrame en el repo privado"

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


def load_github_txts(owner_repo: str, folder_name: str) -> List[dict]:
    """
    Lista y descarga *.txt desde owner/repo (API GitHub).
    Requiere DB_TOKEN. Devuelve [] si la carpeta no existe o no hay .txt.
    """
    owner_repo_n = _txt_repo_arg_to_owner_repo(owner_repo)
    if not owner_repo_n:
        return []
    headers, token = _get_db_token_headers_cli()
    if not token:
        raise RuntimeError("DB_TOKEN no definido en el entorno")
    api_url = f"https://api.github.com/repos/{owner_repo_n}/contents/{folder_name}"
    resp = requests.get(api_url, headers=headers, timeout=45)
    if resp.status_code != 200:
        return []
    items = resp.json()
    if not isinstance(items, list):
        return []
    out: List[dict] = []
    for it in items:
        if it.get("type") != "file" or not str(it.get("name", "")).lower().endswith(".txt"):
            continue
        name = it["name"]
        fu = f"https://api.github.com/repos/{owner_repo_n}/contents/{folder_name}/{name}"
        fr = requests.get(fu, headers=headers, timeout=180)
        if fr.status_code != 200:
            print(f"  (aviso) omitido {folder_name}/{name}: HTTP {fr.status_code}")
            continue
        j = fr.json()
        content_bytes = base64.b64decode(j.get("content", ""))
        content = content_bytes.decode("utf-8", errors="ignore")
        out.append(
            {
                "name": name,
                "content": content,
                "folder": folder_name.lower(),
                "sha": j.get("sha", it.get("sha", "")),
                "size": int(j.get("size", len(content_bytes)) or 0),
            }
        )
    return out


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Regenera el DataFrame desde .txt (repo privado o local) y lo sube a la carpeta data/ "
            f"del repo privado (por defecto {TRANSCRIPTS_TXT_REPO_DEFAULT}) con DB_TOKEN."
        )
    )
    parser.add_argument(
        "--repo",
        default=TRANSCRIPTS_TXT_REPO_DEFAULT,
        help=(
            "Repo owner/repo donde está la carpeta data/ con el DataFrame serializado "
            f"(por defecto {TRANSCRIPTS_TXT_REPO_DEFAULT}). Requiere DB_TOKEN con escritura."
        ),
    )
    parser.add_argument(
        "--data-path",
        default="data",
        help="Ruta en el repo donde guardar el DataFrame (por defecto: data)",
    )
    parser.add_argument(
        "--txt-source",
        choices=("github", "local"),
        default="github",
        help="Origen de los .txt: github (repo privado + DB_TOKEN) o local (--base-dir). Por defecto: github",
    )
    parser.add_argument(
        "--txt-repo",
        default=TRANSCRIPTS_TXT_REPO_DEFAULT,
        help=(
            "Repo owner/repo con carpetas transcripciones/ y spoti/ "
            f"(por defecto {TRANSCRIPTS_TXT_REPO_DEFAULT}). "
            "Se puede sobreescribir con env TRANSCRIPTS_TXT_REPO."
        ),
    )
    parser.add_argument(
        "--base-dir",
        default=".",
        help="Con --txt-source local: directorio con carpetas transcripciones/ y spoti/",
    )

    args = parser.parse_args()

    _, db_tok = _get_db_token_headers_cli()
    if not db_tok:
        print(
            "ERROR: define DB_TOKEN en el entorno (lectura de .txt y escritura del DataFrame en el repo privado)."
        )
        return

    if args.txt_source == "github":
        txt_repo = _transcripts_txt_repo_from_env(args.txt_repo)
        print(f"Leyendo .txt desde GitHub: {txt_repo}")
        try:
            files_trans = load_github_txts(txt_repo, "transcripciones")
            files_spoti: List[dict] = []
            for variant in ("spoti", "Spoti", "SPOTI"):
                files_spoti = load_github_txts(txt_repo, variant)
                if files_spoti:
                    break
        except RuntimeError as e:
            print(f"ERROR: {e}")
            return
    else:
        base_dir = os.path.abspath(args.base_dir)
        print(f"Usando base_dir local = {base_dir}")
        files_trans = load_local_txts(base_dir, "transcripciones")
        files_spoti = []
        for variant in ("spoti", "Spoti", "SPOTI"):
            files_spoti = load_local_txts(base_dir, variant)
            if files_spoti:
                break

    all_files = files_trans + files_spoti
    if not all_files:
        if args.txt_source == "github":
            print(
                "No se encontraron archivos .txt en transcripciones/ ni spoti/ del repo indicado."
            )
        else:
            print(
                "No se encontraron archivos .txt en las carpetas locales 'transcripciones' ni 'spoti'."
            )
        return

    print(
        f"Encontrados {len(all_files)} archivos "
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

