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
def search_transcriptions(
    df: pd.DataFrame,
    query: str,
    use_regex: bool = False,
    fuzzy_mode: str = "contextual",
    threshold: int = 86
) -> pd.DataFrame:
    """Búsqueda flexible con texto normalizado en todos los modos."""
    if df.empty or not query:
        return pd.DataFrame(columns=["file", "speaker", "text", "block_index", "match_preview"])

    # --- Normalizar texto y consulta una sola vez ---
    if "text_norm" not in df.columns:
        df = df.copy()
        df["text_norm"] = df["text"].apply(normalize_text)
    query_norm = normalize_text(query)

    results = pd.DataFrame()

    # --- Búsqueda exacta / regex sobre texto normalizado ---
    if not use_regex:
        # 1️⃣ Primero busca la frase completa
        mask = df["text_norm"].str.contains(re.escape(query_norm), na=False)

         # 2️⃣ Si no hay resultados, busca todas las palabras en cualquier orden
        if not mask.any():
            terms = [t for t in query_norm.split() if t]
            if terms:
                mask = df["text_norm"].apply(lambda t: all(term in t for term in terms))
        results = df.loc[mask].copy()
    else:
        try:
            mask = df["text_norm"].str.contains(query_norm, flags=re.IGNORECASE, regex=True, na=False)
            results = df.loc[mask].copy()
        except re.error:
            st.error("Regex inválida")
            return pd.DataFrame()


    # --- Búsqueda fuzzy si no hay coincidencias exactas ---
    if results.empty and fuzzy_mode != "ninguno":
        st.info(f"🔍 Búsqueda fuzzy activada (modo: {fuzzy_mode}, umbral: {threshold}%)")
        matched_rows = []
        texts = df["text_norm"].tolist()
        rows = df.to_dict("records")

        if fuzzy_mode == "palabra":
            query_terms = [t for t in query_norm.split() if t]
            for text, row in zip(texts, rows):
                for term in query_terms:
                    score = fuzz.partial_ratio(term, text)
                    if score >= threshold:
                        matched_rows.append(row)
                        break
        elif fuzzy_mode == "contextual":
            for text, row in zip(texts, rows):
                score = fuzz.partial_ratio(query_norm, text)
                if score >= threshold:
                    matched_rows.append(row)

        results = pd.DataFrame(matched_rows)

    if results.empty:
        return pd.DataFrame(columns=["file", "speaker", "text", "block_index", "match_preview"])

    # --- Crear vista previa con resaltado ---
    def make_preview(text):
        tnorm = normalize_text(text)
        idx = tnorm.find(query_norm)
        if idx != -1:
            start = max(0, idx - 30)
            snippet = text[start:start + 160]
            preview_text = ("..." if start > 0 else "") + snippet + ("..." if len(text) > start + 160 else "")
        else:
            preview_text = text[:160] + "..."
        
        # Aplicar resaltado a las palabras coincidentes
        return highlight_matching_words(preview_text, query)

    results["match_preview"] = results["text"].apply(make_preview)
    return results[["file", "speaker", "text", "block_index", "match_preview"]]
    
# --- Colorear oradores ---
def color_speaker_row(row):
    s = row["speaker"].strip().lower()
    if s == "eva": return ["background-color: mediumslateblue"]*len(row)
    if s == "nacho": return ["background-color: salmon"]*len(row)
    if s == "lala": return ["background-color: #FF8C00"]*len(row)
    return [""]*len(row)


# --- Obtener color de fondo según speaker ---
def get_speaker_bg_color(speaker: str) -> str:
    """Retorna el color de fondo según el speaker."""
    s = speaker.strip().lower()
    if s == "eva": return "mediumslateblue"
    if s == "nacho": return "salmon"
    if s == "lala": return "#FF8C00"
    return "#f0f0f0"


# --- Mostrar tabla de resultados con HTML ---
def display_results_table(results_df: pd.DataFrame):
    """Muestra los resultados en una tabla HTML que respeta colores y renderiza HTML del resaltado."""
    if results_df.empty:
        return
    
    # Construir todas las filas primero
    rows_html = []
    for i, row in results_df.iterrows():
        bg_color = get_speaker_bg_color(row['speaker'])
        text_color = "white" if bg_color.lower() not in ["#f0f0f0", "salmon", "#ff8c00"] else "black"
        
        # Escapar el nombre del archivo y speaker para HTML (solo caracteres especiales, no el HTML del preview)
        file_name = str(row['file']).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        speaker_name = str(row['speaker']).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        preview = row['match_preview']  # Ya contiene HTML del resaltado, no escapamos esto
        
        rows_html.append(f'<tr style="background-color: {bg_color}; color: {text_color};"><td>{file_name}</td><td><b>{speaker_name}</b></td><td>{preview}</td></tr>')
    
    # Construir el HTML completo de una vez
    html_content = f"""
<style>
.results-table {{
    width: 100%;
    border-collapse: collapse;
    margin: 10px 0;
}}
.results-table th {{
    background-color: #4C98AF;
    color: white;
    padding: 8px;
    text-align: left;
    border: 1px solid #ddd;
}}
.results-table td {{
    padding: 8px;
    border: 1px solid #ddd;
}}
</style>
<table class="results-table">
<thead>
<tr>
<th>Archivo</th>
<th>Orador</th>
<th>Vista Previa</th>
</tr>
</thead>
<tbody>
{''.join(rows_html)}
</tbody>
</table>
"""
    
    st.markdown(html_content, unsafe_allow_html=True)


# --- Resaltar palabras coincidentes en el texto ---
def highlight_matching_words(text: str, query: str) -> str:
    """Resalta las palabras que coinciden con la búsqueda en rojo y negrita."""
    if not query or not text:
        return text
    
    # Normalizar query y obtener términos
    query_norm = normalize_text(query)
    query_terms = set([t for t in query_norm.split() if t])
    
    if not query_terms:
        return text
    
    # Dividir el texto en tokens (palabras y no-palabras) preservando todo
    # Usar regex para dividir manteniendo los delimitadores
    tokens = re.split(r'(\w+)', text)
    
    result_parts = []
    for token in tokens:
        if not token:
            continue
        # Si es una palabra (solo letras/números)
        if re.match(r'^\w+$', token):
            # Normalizar y verificar si coincide
            token_norm = normalize_text(token)
            if token_norm in query_terms:
                result_parts.append(f'<span style="color: red; font-weight: bold;">{token}</span>')
            else:
                result_parts.append(token)
        else:
            # Es puntuación o espacios, mantenerlo tal cual
            result_parts.append(token)
    
    return ''.join(result_parts)


# --- Mostrar contexto ±4 líneas con bloque central resaltado ---
def show_context(df, file, block_idx, query="", context=4):
    sub_df = df[df['file'] == file].reset_index(drop=True)
    idx = sub_df.index[sub_df['block_index'] == block_idx][0]
    start = max(idx - context, 0)
    end = min(idx + context + 1, len(sub_df))

    for i in range(start, end):
        row = sub_df.loc[i]
        speaker = row['speaker']
        text = row['text']
        
        # Resaltar palabras coincidentes si hay query
        if query:
            text = highlight_matching_words(text, query)

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
    q_col, opt_col = st.columns([3, 1])

    with q_col:
        query = st.text_input("Palabra o frase a buscar")

    with opt_col:
        use_regex = st.checkbox("Usar regex", value=False)
        speaker_filter = st.selectbox(
            "Filtrar por orador",
            options=["(todos)"] + sorted(df['speaker'].unique().tolist())
        )
        fuzzy_mode = st.radio("Modo fuzzy", options=["ninguno", "palabra", "contextual"], index=2)
        threshold = st.slider("Umbral similitud (%)", 60, 95, 86)

    if st.button("Buscar"):
        res = search_transcriptions(df, query, use_regex, fuzzy_mode, threshold)
        if speaker_filter != "(todos)":
            res = res[res['speaker'] == speaker_filter]

        if res.empty:
            st.warning("No se encontraron coincidencias.")
        else:
            st.success(f"Encontradas {len(res)} coincidencias")
            display_results_table(res[['file', 'speaker', 'match_preview']])
            for i, row in res.iterrows():
                with st.expander(f"{i+1}. {row['speaker']} — {row['file']} (bloque {row['block_index']})", expanded=False):
                    show_context(df, row['file'], row['block_index'], query=query, context=4)
