"""
Streamlit app: split .m4a audio into 30-minute chunks + build/search transcriptions

Usage notes:
- Requires: streamlit, pydub, pandas, requests
- pydub requires ffmpeg available in the environment (Streamlit Cloud usually has it; if not, install via apt or include a buildpack).
- The app supports:
    * Uploading a .m4a (or other audio) file which will be split into 30-minute segments (1800s). Segments are offered as downloads.
    * Pointing to a public GitHub repo containing a folder `transcripciones` (or uploading multiple .txt files) — the app reads all .txt files, merges into a DataFrame, extracts speaker labels in square brackets and the following text.
    * Searching the merged transcriptions (substring or regex, case-insensitive), showing matches, their speaker, file and surrounding context.

Drop the file into Streamlit Cloud repo, or deploy this file as `streamlit_app.py` in your Streamlit Cloud project.
"""

import streamlit as st
from pydub import AudioSegment
import io
import math
import pandas as pd
import re
import requests
from typing import List

st.set_page_config(page_title="Audio splitter + Transcriptions search", layout="wide")

st.title("🔊 Audio splitter (.m4a) + Transcriptions search")

# --- Helper functions ---

def split_audio(audio_bytes: bytes, filename: str, segment_seconds: int = 1800) -> List[dict]:
    """Split an audio file given as bytes into segments of segment_seconds.
    Returns list of dicts: {"name": ..., "bytes": ...}
    Attempts to preserve format; falls back to WAV if needed.
    """
    audio_file = io.BytesIO(audio_bytes)
    # try to detect format from filename
    fmt = None
    if filename and "." in filename:
        fmt = filename.rsplit('.', 1)[1].lower()

    try:
        # pydub can detect format from file content if format None
        audio = AudioSegment.from_file(audio_file, format=fmt)
    except Exception as e:
        st.error(f"Error loading audio file with pydub: {e}")
        return []

    duration_ms = len(audio)
    segment_ms = segment_seconds * 1000
    n_segments = math.ceil(duration_ms / segment_ms)

    segments = []
    for i in range(n_segments):
        start = i * segment_ms
        end = min((i + 1) * segment_ms, duration_ms)
        seg = audio[start:end]
        seg_io = io.BytesIO()
        # try to export with same extension if known and supported, else wav
        export_format = None
        if fmt in ("mp3", "wav", "ogg", "flv", "raw", "wma", "aac", "m4a"):
            export_format = fmt
        else:
            export_format = "wav"
        try:
            seg.export(seg_io, format=export_format)
        except Exception:
            # fallback
            seg_io = io.BytesIO()
            seg.export(seg_io, format="wav")
            export_format = "wav"
        seg_io.seek(0)
        seg_name = f"{filename.rsplit('.',1)[0]}_part{i+1}.{export_format}"
        segments.append({"name": seg_name, "bytes": seg_io.read()})
    return segments


def read_txt_files_from_github(repo_url: str, path: str = "transcripciones") -> List[dict]:
    """Given a GitHub repo URL like https://github.com/owner/repo (optionally with .git)
    fetch the list of files under `path` using the GitHub API and download .txt files.
    Returns list of {"name":..., "content":...}
    Works for public repos. If access fails, returns empty list and sets st.error.
    """
    # parse owner/repo
    m = re.match(r"https?://github.com/([^/]+)/([^/]+)", repo_url)
    if not m:
        st.error("No se pudo reconocer la URL del repositorio. Use la forma https://github.com/owner/repo")
        return []
    owner, repo = m.group(1), m.group(2).replace('.git','')
    api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
    resp = requests.get(api_url)
    if resp.status_code != 200:
        st.error(f"Error fetching contents from GitHub API: {resp.status_code} - {resp.text}")
        return []
    files = resp.json()
    data = []
    for f in files:
        if f.get('type') == 'file' and f.get('name','').lower().endswith('.txt'):
            raw_url = f.get('download_url')
            txt_resp = requests.get(raw_url)
            if txt_resp.status_code == 200:
                data.append({"name": f['name'], "content": txt_resp.text})
    if not data:
        st.warning("No se encontraron archivos .txt en la carpeta transcripciones del repo (o están vacíos).")
    return data


def parse_transcription_text(name: str, text: str) -> pd.DataFrame:
    """Parse a single transcription text into rows with speaker and text.
    Looks for patterns like [Speaker] line... and also handles multiple blocks per file.
    Returns DataFrame with columns: file, speaker, text, block_index
    """
    # We'll find all occurrences of [Speaker] followed by some text until next [ or EOF
    pattern = re.compile(r"\[([^\]]+)\]\s*(.*?)((?=\[)|$)", re.S)
    rows = []
    for idx, m in enumerate(pattern.finditer(text)):
        speaker = m.group(1).strip()
        content = m.group(2).strip().replace('\r\n','\n')
        # collapse excess whitespace but preserve sentence breaks
        content = re.sub(r"\n+", " ", content)
        content = re.sub(r"\s+", " ", content).strip()
        rows.append({"file": name, "speaker": speaker, "text": content, "block_index": idx})
    # If no matches, fallback: whole file as unknown speaker
    if not rows:
        cleaned = re.sub(r"\s+", " ", text).strip()
        rows.append({"file": name, "speaker": "UNKNOWN", "text": cleaned, "block_index": 0})
    return pd.DataFrame(rows)


def build_transcriptions_dataframe(files: List[dict]) -> pd.DataFrame:
    dfs = []
    for f in files:
        df = parse_transcription_text(f['name'], f['content'])
        dfs.append(df)
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    else:
        return pd.DataFrame(columns=["file","speaker","text","block_index"])


def search_transcriptions(df: pd.DataFrame, query: str, use_regex: bool=False) -> pd.DataFrame:
    if df.empty or not query:
        return pd.DataFrame(columns=["file","speaker","text","block_index","match_preview"]) 
    results = []
    flags = re.IGNORECASE
    for _, row in df.iterrows():
        try:
            if use_regex:
                if re.search(query, row['text'], flags):
                    results.append(row.to_dict())
            else:
                if query.lower() in row['text'].lower():
                    results.append(row.to_dict())
        except re.error:
            st.error("Expresión regular inválida")
            return pd.DataFrame()
    if not results:
        return pd.DataFrame(columns=["file","speaker","text","block_index","match_preview"]) 
    res_df = pd.DataFrame(results)
    # add a preview with highlighted match (simple) — here we return a snippet
    def make_preview(text):
        idx = text.lower().find(query.lower()) if not use_regex else None
        if idx is None and use_regex:
            try:
                m = re.search(query, text, flags)
                if m:
                    idx = max(m.start()-30,0)
                    return ("..." + text[idx:idx+160] + "...")
            except re.error:
                return text[:160]
        if idx is not None:
            start = max(idx-30,0)
            return ("..." + text[start:start+160] + "...")
        return text[:160]
    res_df['match_preview'] = res_df['text'].apply(make_preview)
    return res_df[['file','speaker','text','block_index','match_preview']]


# --- UI: Audio splitting ---
st.header("1) Cortar audio (.m4a) en fragmentos de 30 minutos")
col1, col2 = st.columns([2,1])
with col1:
    uploaded = st.file_uploader("Sube un archivo de audio (.m4a, .mp3, etc.)", type=["m4a","mp3","wav","ogg","flac"], accept_multiple_files=False)
    segment_minutes = st.number_input("Duración de cada fragmento (minutos)", min_value=1, max_value=180, value=30)
    if uploaded:
        if st.button("Procesar audio y generar fragmentos"):
            audio_bytes = uploaded.read()
            with st.spinner("Cortando audio..."):
                segments = split_audio(audio_bytes, uploaded.name, segment_seconds=int(segment_minutes*60))
            if segments:
                st.success(f"Generados {len(segments)} fragmentos")
                for seg in segments:
                    st.download_button(label=f"Descargar {seg['name']}", data=seg['bytes'], file_name=seg['name'])
            else:
                st.error("No se generaron fragmentos. Revisa el archivo o los códecs (ffmpeg).")

with col2:
    st.markdown("**Consejos**:\n- Si la descarga no funciona, intenta exportar a WAV o MP3.\n- pydub necesita ffmpeg; en Streamlit Cloud normalmente ya está disponible.\n- Los archivos WAV pueden ser grandes.")

st.markdown("---")

# --- UI: Transcriptions loader and search ---
st.header("2) Leer transcripciones (carpeta `transcripciones` en GitHub o subir .txt)")
repo_col, upload_col = st.columns(2)
with repo_col:
    gh_url = st.text_input("(Opcional) URL del repositorio público en GitHub (ej: https://github.com/owner/repo)", value="")
    if gh_url:
        if st.button("Cargar archivos .txt desde GitHub" , key="gh_load"):
            with st.spinner("Leyendo archivos desde GitHub..."):
                files = read_txt_files_from_github(gh_url, path="transcripciones")
                if files:
                    st.session_state['trans_files'] = files
                    st.success(f"Cargados {len(files)} archivos desde transcripciones/")

with upload_col:
    uploaded_txts = st.file_uploader("O sube uno o varios archivos .txt", type=['txt'], accept_multiple_files=True)
    if uploaded_txts:
        files = []
        for f in uploaded_txts:
            try:
                txt = f.read().decode('utf-8')
            except Exception:
                # try latin-1
                txt = f.read().decode('latin-1')
            files.append({"name": f.name, "content": txt})
        st.session_state['trans_files'] = files
        st.success(f"Se han cargado {len(files)} archivos .txt")

# Build dataframe if files exist in session_state
if 'trans_files' in st.session_state:
    st.subheader("Transcripciones cargadas")
    files = st.session_state['trans_files']
    if files:
        if st.button("Construir DataFrame de transcripciones", key='build_df'):
            with st.spinner("Parseando transcripciones..."):
                df = build_transcriptions_dataframe(files)
                st.session_state['trans_df'] = df
                st.success(f"DataFrame generado con {len(df)} bloques (filas)")
                st.dataframe(df[['file','speaker','text']].head(200))
else:
    st.info("No hay archivos de transcripciones cargados. Usa la URL de GitHub o sube archivos .txt.")

st.markdown("---")

# --- Search UI ---
st.header("3) Buscar dentro de las transcripciones")
if 'trans_df' in st.session_state:
    df = st.session_state['trans_df']
    q_col, opt_col = st.columns([3,1])
    with q_col:
        query = st.text_input("Palabra o frase a buscar (soporta regex si marcas la opción)")
    with opt_col:
        use_regex = st.checkbox("Usar regex", value=False)
        speaker_filter = st.selectbox("Filtrar por orador (opcional)", options=["(todos)"] + sorted(df['speaker'].unique().tolist()))
    if st.button("Buscar"):
        with st.spinner("Buscando..."):
            res = search_transcriptions(df, query, use_regex)
            if speaker_filter != "(todos)":
                res = res[res['speaker'] == speaker_filter]
            if res.empty:
                st.warning("No se encontraron coincidencias.")
            else:
                st.success(f"Encontradas {len(res)} coincidencias")
                # show results with preview and allow filtering by file
                st.dataframe(res[['file','speaker','match_preview']])
                # expand individual result
                for i, row in res.iterrows():
                    with st.expander(f"{row['speaker']} — {row['file']} (bloque {row['block_index']})"):
                        st.write(row['text'])
                        st.write("---")
else:
    st.info("Construye primero el DataFrame de transcripciones para poder buscar.")

st.markdown("---")
st.caption("Hecho con ❤️ — sube un audio .m4a para dividirlo y apunta a tu repo público con carpeta `transcripciones` para buscar en los textos. Ajusta la lógica de parsing según el formato exacto de tus transcripciones.")
