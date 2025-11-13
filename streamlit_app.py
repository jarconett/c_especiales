import streamlit as st
from moviepy.editor import AudioFileClip
import io, math, pandas as pd, re, requests, tempfile, os, base64, unicodedata
from typing import List
from rapidfuzz import fuzz

# -------------------------------
# CONFIGURACIÓN DE LA APP
# -------------------------------
st.set_page_config(
    page_title="Audio splitter + Transcriptions search optimizado",
    layout="wide",
    initial_sidebar_state="collapsed",
    page_icon="💵"
)
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)
st.title("💰🔊 A ganar billete 💵 💶 💴")

# --- Helper functions ---
FFMPEG_BIN = r"C:\Users\Javier\Downloads\ffmpeg.exe"
os.environ["IMAGEIO_FFMPEG_EXE"] = FFMPEG_BIN

def split_audio(audio_bytes: bytes, filename: str, segment_seconds: int = 1800):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".m4a") as tmpfile:
        tmpfile.write(audio_bytes)
        tmp_path = tmpfile.name

    clip = AudioFileClip(tmp_path)
    duration = clip.duration
    n_segments = math.ceil(duration / segment_seconds)
    segments = []

    for i in range(n_segments):
        start = i * segment_seconds
        end = min((i + 1) * segment_seconds, duration)
        seg_clip = clip.subclip(start, end)

        seg_name = f"{filename.rsplit('.',1)[0]}_part{i+1}.m4a"
        with tempfile.NamedTemporaryFile(delete=False, suffix=".m4a") as seg_tmp:
            seg_clip.write_audiofile(seg_tmp.name, codec="aac", verbose=True, logger=None)
            seg_tmp.seek(0)
            seg_bytes = open(seg_tmp.name, "rb").read()

        segments.append({"name": seg_name, "bytes": seg_bytes})
        os.unlink(seg_tmp.name)

    clip.close()
    os.unlink(tmp_path)
    return segments


# --- GitHub utils ---
def _get_github_headers():
    token = None
    try:
        token = st.secrets.get("GITHUB_TOKEN")
    except Exception:
        token = None
    if not token:
        token = os.getenv("GITHUB_TOKEN")
    headers = {}
    if token:
        headers["Authorization"] = f"token {token}"
    headers["Accept"] = "application/vnd.github.v3+json"
    return headers


def read_txt_files_from_github(repo_url: str, path: str = "transcripciones") -> List[dict]:
    import re as _re
    if repo_url.count("/") == 1 and "/" in repo_url:
        owner_repo = repo_url
    else:
        m = _re.match(r"https?://github.com/([^/]+)/([^/]+)", repo_url)
        if not m:
            st.error("URL de repo no válida.")
            return []
        owner_repo = f"{m.group(1)}/{m.group(2).replace('.git','')}"
    
    headers = _get_github_headers()
    api_url = f"https://api.github.com/repos/{owner_repo}/contents/{path}"
    resp = requests.get(api_url, headers=headers)
    
    if resp.status_code != 200:
        return []  # No error, solo retornar vacío para que pueda intentar otra carpeta
    
    items = resp.json()
    data = []
    for f in items:
        if f.get("type") == "file" and f.get("name","").lower().endswith(".txt"):
            file_api = f"https://api.github.com/repos/{owner_repo}/contents/{path}/{f['name']}"
            file_resp = requests.get(file_api, headers=headers)
            if file_resp.status_code == 200:
                file_info = file_resp.json()
                try:
                    content_bytes = base64.b64decode(file_info.get("content", ""))
                    content = content_bytes.decode("utf-8", errors="ignore")
                except Exception:
                    content = ""
                data.append({"name": f['name'], "content": content})
    return data


def load_transcriptions_from_github(repo_url: str) -> tuple[List[dict], str]:
    """
    Intenta cargar archivos desde 'transcripciones' primero, 
    si no encuentra nada, intenta desde 'spoti'.
    Retorna (files, folder_used)
    """
    # Intentar primero en "transcripciones"
    files = read_txt_files_from_github(repo_url, path="transcripciones")
    if files:
        return files, "transcripciones"
    
    # Si no encuentra nada, intentar en "spoti"
    files = read_txt_files_from_github(repo_url, path="spoti")
    if files:
        return files, "spoti"
    
    # Si no encuentra nada en ninguna carpeta
    return [], ""


def parse_transcription_text(name: str, text: str) -> pd.DataFrame:
    pattern = re.compile(r"\[([^\]]+)\]\s*(.*?)((?=\[)|$)", re.S)
    rows = []
    for idx, m in enumerate(pattern.finditer(text)):
        speaker = m.group(1).strip()
        content = re.sub(r"\s+", " ", m.group(2).strip().replace('\r\n','\n')).strip()
        rows.append({"file": name, "speaker": speaker, "text": content, "block_index": idx})
    if not rows:
        cleaned = re.sub(r"\s+"," ", text).strip()
        rows.append({"file": name, "speaker": "UNKNOWN", "text": cleaned, "block_index": 0})
    return pd.DataFrame(rows)


def build_transcriptions_dataframe(files: List[dict]) -> pd.DataFrame:
    dfs = [parse_transcription_text(f['name'], f['content']) for f in files]
    if dfs:
        df = pd.concat(dfs, ignore_index=True)
        df["text_norm"] = df["text"].apply(normalize_text)  # 💥 normalizamos una vez
        return df
    else:
        return pd.DataFrame(columns=["file","speaker","text","block_index"])


# --- Texto y búsqueda optimizados ---
def normalize_text(text: str) -> str:
    """Normaliza texto: minúsculas, sin tildes, sin saltos ni espacios extra."""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


# --- Búsqueda optimizada con fuzzy ---
def search_transcriptions(df: pd.DataFrame, query: str, use_regex: bool=False, fuzzy_mode: str="palabra", threshold: int=80) -> pd.DataFrame:
    if df.empty or not query:
        return pd.DataFrame(columns=["file","speaker","text","block_index","match_preview"])

    flags = re.IGNORECASE
    query_norm = query.lower().strip()
    results = []

    # --- búsqueda vectorizada exacta o regex ---
    if not use_regex:
        mask = df["text"].str.contains(query_norm, case=False, na=False)
        results = df[mask].copy()
    else:
        try:
            mask = df["text"].str.contains(query, flags=flags, regex=True, na=False)
            results = df[mask].copy()
        except re.error:
            st.error("Regex inválida")
            return pd.DataFrame()

    # --- búsqueda fuzzy si no hay coincidencias exactas ---
    if results.empty and fuzzy_mode:
        st.info(f"🔍 Búsqueda fuzzy activada (modo: {fuzzy_mode}, umbral: {threshold}%)")
        matched_rows = []

        for _, row in df.iterrows():
            text = row["text"].lower()
            if fuzzy_mode == "palabra":
                # palabra a palabra
                for word in query_norm.split():
                    score = fuzz.partial_ratio(word, text)
                    if score >= threshold:
                        matched_rows.append(row)
                        break
            elif fuzzy_mode == "contextual":
                # similitud global de frase completa
                score = fuzz.partial_ratio(query_norm, text)
                if score >= threshold:
                    matched_rows.append(row)

        results = pd.DataFrame(matched_rows)

    if results.empty:
        return pd.DataFrame(columns=["file","speaker","text","block_index","match_preview"])

    # --- generar previsualización ---
    def make_preview(text):
        idx = text.lower().find(query_norm)
        if idx != -1:
            start = max(idx - 30, 0)
            return "..." + text[start:start + 160] + "..."
        else:
            return text[:160] + "..."

    results["match_preview"] = results["text"].apply(make_preview)
    return results[["file","speaker","text","block_index","match_preview"]]
    
# --- Colorear oradores ---
def color_speaker_row(row):
    s = row["speaker"].strip().lower()
    if s == "eva": return ["background-color: mediumslateblue"]*len(row)
    if s == "nacho": return ["background-color: salmon"]*len(row)
    if s == "lala": return ["background-color: #FF8C00"]*len(row)
    return [""]*len(row)


# --- Mostrar contexto ±4 líneas con bloque central resaltado ---
def show_context(df, file, block_idx, context=4):
    sub_df = df[df['file'] == file].reset_index(drop=True)
    idx = sub_df.index[sub_df['block_index'] == block_idx][0]
    start = max(idx - context, 0)
    end = min(idx + context + 1, len(sub_df))

    for i in range(start, end):
        row = sub_df.loc[i]
        speaker = row['speaker']
        text = row['text']

        if speaker.lower() == "eva":
            bg_color = "mediumslateblue"
        elif speaker.lower() == "nacho":
            bg_color = "salmon"
        elif speaker.lower() == "lala":
            bg_color = "#FF8C00"
        else:
            bg_color = "#f0f0f0"

        border_style = "2px solid yellow" if i == idx else "none"
        text_color = "white" if bg_color.lower() not in ["#f0f0f0", "salmon", "#FF8C00"] else "black"

        st.markdown(
            f"<div style='background-color: {bg_color}; padding:4px; border-radius:4px; border: {border_style}; color: {text_color};'><b>{speaker}:</b> {text}</div>",
            unsafe_allow_html=True
        )


# --- UI: Audio splitting ---
st.header("1) Cortar audio (.m4a) en fragmentos de 30 minutos")
col1, col2 = st.columns([2,1])
with col1:
    uploaded = st.file_uploader("Sube un archivo de audio", type=["m4a","mp3","wav","ogg","flac"])
    segment_minutes = st.number_input("Duración de cada fragmento (minutos)", min_value=1, max_value=180, value=30)
    if uploaded and st.button("Procesar audio y generar fragmentos"):
        audio_bytes = uploaded.read()
        with st.spinner("Cortando audio..."):
            segments = split_audio(audio_bytes, uploaded.name, segment_seconds=int(segment_minutes*60))
            st.session_state['audio_segments'] = segments
            st.success(f"Generados {len(segments)} fragmentos")
    if 'audio_segments' in st.session_state:
        st.markdown("### Descargar fragmentos")
        for seg in st.session_state['audio_segments']:
            st.download_button(f"Descargar {seg['name']}", data=seg['bytes'], file_name=seg['name'])

with col2:
    st.markdown("""
    **Importante**:
    - [Cortar audio online](https://mp3cut.net/es)
    - [Transcripción automática](https://turboscribe.ai/)
    - [ffmpeg](https://www.gyan.dev/ffmpeg/builds)
    """, unsafe_allow_html=True)

st.markdown("---")


# --- UI: Transcriptions loader ---
st.header("2) Leer transcripciones")
repo_col, _ = st.columns(2)
with repo_col:
    gh_url = st.text_input("Repo público GitHub (carpeta transcripciones)", value="https://github.com/jarconett/c_especiales/")
    
    # Carga automática al inicio si no hay datos
    if gh_url and 'trans_files' not in st.session_state:
        with st.spinner("Cargando archivos .txt desde GitHub..."):
            files, folder_used = load_transcriptions_from_github(gh_url)
            if files:
                st.session_state['trans_files'] = files
                st.session_state['trans_df'] = build_transcriptions_dataframe(files)
                st.success(f"Cargados {len(files)} archivos desde carpeta '{folder_used}' y DataFrame con {len(st.session_state['trans_df'])} bloques")
            else:
                st.warning("No se encontraron archivos .txt en las carpetas 'transcripciones' ni 'spoti'")
    
    # Botón para recargar manualmente
    if st.button("🔄 Recargar archivos .txt desde GitHub", key="reload_transcriptions"):
        if gh_url:
            with st.spinner("Recargando archivos .txt desde GitHub..."):
                files, folder_used = load_transcriptions_from_github(gh_url)
                if files:
                    st.session_state['trans_files'] = files
                    st.session_state['trans_df'] = build_transcriptions_dataframe(files)
                    st.success(f"Recargados {len(files)} archivos desde carpeta '{folder_used}' y DataFrame con {len(st.session_state['trans_df'])} bloques")
                else:
                    st.warning("No se encontraron archivos .txt en las carpetas 'transcripciones' ni 'spoti'")
        else:
            st.error("Por favor, ingresa una URL de repositorio GitHub válida")


# --- UI: Search ---
st.header("3) Buscar en transcripciones")
if 'trans_df' in st.session_state:
    df = st.session_state['trans_df']
    q_col, opt_col = st.columns([3,1])
    with q_col:
        query = st.text_input("Palabra o frase a buscar")
    with opt_col:
        use_regex = st.checkbox("Usar regex", value=False)
        speaker_filter = st.selectbox("Filtrar por orador", options=["(todos)"] + sorted(df['speaker'].unique().tolist()))
        fuzzy_mode = st.radio("Modo fuzzy", options=["ninguno", "palabra", "contextual"], index=0)
        threshold = st.slider("Umbral similitud (%)", 60, 95, 80)

    # 🔍 Opciones fuzzy
    fuzzy_mode = st.checkbox("Permitir errores menores (búsqueda difusa)", value=True)
    threshold = st.slider("Sensibilidad fuzzy (mayor = más estricta)", 60, 90, 75)

    if st.button("Buscar"):
        res = search_transcriptions(df, query, use_regex, fuzzy_mode if fuzzy_mode != "ninguno" else None, threshold)
        if speaker_filter != "(todos)":
            res = res[res['speaker'] == speaker_filter]
        if res.empty:
            st.warning("No se encontraron coincidencias.")
        else:
            st.success(f"Encontradas {len(res)} coincidencias")
            st.dataframe(
                res[['file','speaker','match_preview'] + (['fuzzy_score'] if 'fuzzy_score' in res else [])]
                .style.apply(color_speaker_row, axis=1),
                use_container_width=True
            )
            for i, row in res.iterrows():
                with st.expander(f"{i+1}. {row['speaker']} — {row['file']} (bloque {row['block_index']})", expanded=False):
                    show_context(df, row['file'], row['block_index'], context=4)

