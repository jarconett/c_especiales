import streamlit as st
from moviepy.editor import AudioFileClip
import io, math, pandas as pd, re, requests, tempfile, os, base64, unicodedata, html
from typing import List

# Se eliminan las importaciones de búsqueda semántica
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np
# import torch
# import torch.nn.functional as F
# from transformers import AutoModel, AutoTokenizer

st.set_page_config(page_title="Audio splitter + Transcriptions search", layout="wide")
st.title("⚠️⚠️⚠️⚠️TEST BRANCH⚠️⚠️⚠️⚠️💰🔊 A ganar billete 💵 💶 💴 (Búsqueda semántica ELIMINADA)")

# Inicializar session_state
def initialize_session_state():
    """Inicializa las variables de session_state necesarias"""
    if 'trans_df' not in st.session_state:
        st.session_state['trans_df'] = pd.DataFrame()
    if 'spoti_df' not in st.session_state:
        st.session_state['spoti_df'] = pd.DataFrame()
    # Se eliminan las variables de estado de embeddings
    # if 'has_embeddings' not in st.session_state:
    #     st.session_state['has_embeddings'] = False
    # if 'spoti_has_embeddings' not in st.session_state:
    #     st.session_state['spoti_has_embeddings'] = False
    # if 'embed_model' not in st.session_state:
    #     st.session_state['embed_model'] = None
    # if 'spoti_embed_model' not in st.session_state:
    #     st.session_state['spoti_embed_model'] = None

# Inicializar estado
initialize_session_state()

# -------------------------------
# Configuración FFMPEG
# -------------------------------
# Buscar FFMPEG en ubicaciones comunes
FFMPEG_PATHS = [
    r"C:\Users\Javier\Downloads\ffmpeg.exe",  # Ruta específica del usuario
    "ffmpeg",  # Si está en PATH
    "/usr/bin/ffmpeg",  # Linux
    "/usr/local/bin/ffmpeg",  # macOS/Linux
]

FFMPEG_BIN = None
for path in FFMPEG_PATHS:
    # En Streamlit Cloud, confiar en el PATH o no usar funciones de audio que requieran FFMPEG.
    # Eliminamos la verificación de existencia para 'ffmpeg' en el PATH.
    if path == "ffmpeg" or os.path.exists(path):
        FFMPEG_BIN = path
        break

if FFMPEG_BIN:
    # Esto puede no funcionar en Streamlit Cloud sin una instalación explícita
    os.environ["IMAGEIO_FFMPEG_EXE"] = FFMPEG_BIN
else:
    st.warning("⚠️ FFMPEG no encontrado. Algunas funciones de audio pueden no funcionar en Streamlit Cloud.")

# -------------------------------
# FUNCIONES DE AUDIO
# -------------------------------
def split_audio(audio_bytes: bytes, filename: str, segment_seconds: int = 1800):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".m4a") as tmpfile:
            tmpfile.write(audio_bytes)
            tmp_path = tmpfile.name

        clip = AudioFileClip(tmp_path)
        duration = clip.duration
        n_segments = math.ceil(duration / segment_seconds)
        segments = []

        for i in range(n_segments):
            start = i * segment_seconds
            end = min((i+1)*segment_seconds, duration)
            seg_clip = clip.subclip(start, end)

            seg_name = f"{filename.rsplit('.',1)[0]}_part{i+1}.m4a"
            with tempfile.NamedTemporaryFile(delete=False, suffix=".m4a") as seg_tmp:
                # Es crucial que Streamlit Cloud tenga las dependencias de audio/ffmpeg
                # Si esto falla, Streamlit crasheará o la función no generará el archivo
                seg_clip.write_audiofile(seg_tmp.name, codec="aac", verbose=False, logger=None)
                seg_tmp.seek(0)
                seg_bytes = open(seg_tmp.name, "rb").read()
            segments.append({"name": seg_name, "bytes": seg_bytes})
            os.unlink(seg_tmp.name)

        clip.close()
        os.unlink(tmp_path)
        return segments
        
    except Exception as e:
        # Limpiar archivos temporales en caso de error
        try:
            if 'clip' in locals():
                clip.close()
            if 'tmp_path' in locals() and os.path.exists(tmp_path):
                os.unlink(tmp_path)
        except:
            pass
        raise Exception(f"Error procesando audio: {str(e)}")

# -------------------------------
# FUNCIONES GITHUB
# -------------------------------
def _get_github_headers():
    token = None
    try:
        token = os.getenv("GITHUB_TOKEN")
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
    
    if resp.status_code == 403:
        st.error("❌ Error 403: Acceso denegado a GitHub")
        st.warning("""
        **Posibles soluciones:**
        1. **Token de GitHub**: Configura un token personal de GitHub
        2. **Límite de API**: Has excedido el límite de la API de GitHub (60 requests/hora sin token)
        3. **Repo privado**: El repositorio puede ser privado
        
        **Para configurar un token:**
        - Ve a GitHub → Settings → Developer settings → Personal access tokens
        - Crea un token con permisos de 'repo' (para repos privados) o 'public_repo'
        - Configúralo como variable de entorno: `GITHUB_TOKEN=tu_token`
        - O añádelo a los secrets de Streamlit: `GITHUB_TOKEN`
        """)
        return []
    elif resp.status_code == 404:
        st.error(f"❌ Error 404: No se encontró el repositorio o la carpeta '{path}'")
        st.info(f"Verifica que el repositorio '{owner_repo}' existe y tiene una carpeta '{path}'")
        return []
    elif resp.status_code != 200:
        st.error(f"❌ Error {resp.status_code}: {resp.text}")
        return []
    
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
            else:
                st.warning(f"No se pudo cargar el archivo {f['name']}: {file_resp.status_code}")
    return data

def parse_transcription_text(name: str, text: str) -> pd.DataFrame:
    pattern = re.compile(r"\[([^\]]+)\]\s*(.*?)((?=\[)|$)", re.S)
    rows = []
    for idx, m in enumerate(pattern.finditer(text)):
        speaker = m.group(1).strip()
        content = re.sub(r"\s+"," ", m.group(2).strip().replace('\r\n','\n')).strip()
        rows.append({"file": name, "speaker": speaker, "text": content, "block_index": idx})
    if not rows:
        cleaned = re.sub(r"\s+"," ", text).strip()
        rows.append({"file": name, "speaker": "UNKNOWN", "text": cleaned, "block_index": 0})
    return pd.DataFrame(rows)

def build_transcriptions_dataframe(files: List[dict]) -> pd.DataFrame:
    dfs = [parse_transcription_text(f['name'], f['content']) for f in files]
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    else:
        return pd.DataFrame(columns=["file","speaker","text","block_index"])

def parse_spoti_text(name: str, text: str) -> pd.DataFrame:
    """
    Parsea archivos de spoti que no tienen identificadores de orador.
    Divide el texto en párrafos o bloques lógicos.
    """
    # Dividir por párrafos (doble salto de línea) o por longitud
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    # Si no hay párrafos claros, dividir por líneas largas
    if not paragraphs:
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        paragraphs = []
        current_paragraph = []
        
        for line in lines:
            if len(line) > 50:  # Línea suficientemente larga
                if current_paragraph:
                    paragraphs.append(' '.join(current_paragraph))
                    current_paragraph = []
                paragraphs.append(line)
            else:
                current_paragraph.append(line)
        
        if current_paragraph:
            paragraphs.append(' '.join(current_paragraph))
    
    rows = []
    for idx, paragraph in enumerate(paragraphs):
        if paragraph:
            # Limpiar el texto
            cleaned = re.sub(r"\s+", " ", paragraph.strip()).strip()
            if cleaned:
                rows.append({
                    "file": name,  
                    "speaker": "DESCONOCIDO",  
                    "text": cleaned,  
                    "block_index": idx
                })
    
    if not rows:
        # Si no se pudo dividir, usar todo el texto como un bloque
        cleaned = re.sub(r"\s+", " ", text.strip()).strip()
        if cleaned:
            rows.append({
                "file": name,  
                "speaker": "DESCONOCIDO",  
                "text": cleaned,  
                "block_index": 0
            })
    
    return pd.DataFrame(rows)

def build_spoti_dataframe(files: List[dict]) -> pd.DataFrame:
    dfs = [parse_spoti_text(f['name'], f['content']) for f in files]
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    else:
        return pd.DataFrame(columns=["file","speaker","text","block_index"])

# -------------------------------
# FUNCIONES BÚSQUEDA LITERAL
# -------------------------------
def normalize_text(text: str) -> str:
    # Función para eliminar acentos y pasar a minúsculas
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return text.lower()

def norm_and_mapping(orig: str):
    # Ayuda a mapear caracteres normalizados de vuelta al texto original
    mapping = []
    norm_chars = []
    for i, ch in enumerate(orig):
        decomp = unicodedata.normalize('NFD', ch)
        base = ''.join(c for c in decomp if unicodedata.category(c) != 'Mn')
        if base == '':
            continue
        for b in base:
            mapping.append(i)
            norm_chars.append(b)
    return ''.join(norm_chars).lower(), mapping

def highlight_html(orig_text: str, terms: list) -> str:
    norm, mapping = norm_and_mapping(orig_text)
    spans = []
    for term in terms:
        if not term: continue
        start = 0
        while True:
            idx = norm.find(term, start)
            if idx == -1: break
            # Mapear índice normalizado al original
            s_orig = mapping[idx]
            e_orig = mapping[idx + len(term) - 1] + 1
            spans.append((s_orig, e_orig))
            start = idx + len(term)
    if not spans: return html.escape(orig_text)
    
    # Fusionar superposiciones
    spans.sort()
    merged = []
    cur_s, cur_e = spans[0]
    for s,e in spans[1:]:
        if s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s,e
    merged.append((cur_s, cur_e))
    
    # Crear HTML con resaltado
    parts = []
    last = 0
    for s,e in merged:
        parts.append(html.escape(orig_text[last:s]))
        parts.append(f"<b>{html.escape(orig_text[s:e])}</b>")
        last = e
    parts.append(html.escape(orig_text[last:]))
    return ''.join(parts)

def highlight_preview(orig_text: str, terms: list, preview_len: int = 160) -> str:
    norm, mapping = norm_and_mapping(orig_text)
    for term in terms:
        if not term: continue
        idx = norm.find(term)
        if idx != -1:
            match_start = mapping[idx]
            match_end = mapping[idx + len(term) - 1] + 1
            start = max(0, match_start - 30)
            end = min(len(orig_text), start + preview_len)
            text_snip = orig_text[start:end]
            rel_s = match_start - start
            rel_e = rel_s + (match_end - match_start)
            marked = text_snip[:rel_s] + "**" + text_snip[rel_s:rel_e] + "**" + text_snip[rel_e:]
            prefix = "..." if start > 0 else ""
            suffix = "..." if end < len(orig_text) else ""
            return prefix + marked + suffix
    return orig_text[:preview_len] + ("..." if len(orig_text) > preview_len else "")

def search_transcriptions(df: pd.DataFrame, query: str, use_regex: bool=False, all_words: bool=True) -> pd.DataFrame:
    """Realiza una búsqueda literal (por palabras clave/regex) en el DataFrame."""
    expected_cols = ['file','speaker','text','block_index','match_preview']
    if df.empty or not query:
        return pd.DataFrame(columns=expected_cols)

    results = []
    flags = re.IGNORECASE
    query_terms = [t for t in normalize_text(query).split() if t]

    for _, row in df.iterrows():
        text = row['text'] or ""
        norm_text = normalize_text(text)
        try:
            if use_regex:
                if re.search(query, text, flags):
                    results.append(row.to_dict())
            else:
                if not query_terms: continue
                # Se mantiene la lógica de búsqueda literal por palabras (todas o cualquiera)
                if all_words and all(term in norm_text for term in query_terms):
                    results.append(row.to_dict())
                elif not all_words and any(term in norm_text for term in query_terms):
                    results.append(row.to_dict())
        except re.error:
            st.error("Regex inválida")
            return pd.DataFrame(columns=expected_cols)

    res_df = pd.DataFrame(results)
    if res_df.empty:
        return pd.DataFrame(columns=expected_cols)

    res_df['match_preview'] = res_df['text'].apply(lambda t: highlight_preview(t, query_terms))
    # Aquí se añaden las columnas 'text' y 'block_index' al resultado final
    # para que se puedan usar en el contexto, y se reordenan
    return res_df[['file','speaker','match_preview', 'text', 'block_index']]

# -------------------------------
# CONTEXTO y ESTILOS
# -------------------------------
def show_context(df, file, block_idx, query_terms, context=4):
    sub_df = df[df['file'] == file].reset_index(drop=True)
    matches = sub_df.index[sub_df['block_index'] == block_idx].tolist()
    idx = matches[0] if matches else 0
    start = max(idx - context, 0)
    end = min(idx + context + 1, len(sub_df))
    for i in range(start, end):
        row = sub_df.loc[i]
        speaker = row['speaker']
        text_html = highlight_html(row['text'], query_terms)
        bg_color = {"eva":"mediumslateblue","nacho":"salmon","lala":"#FF8C00","desconocido":"#E0E0E0"}.get(speaker.lower(), "#f0f0f0")
        border_style = "2px solid yellow" if i==idx else "none"
        text_color = "white" if bg_color.lower() not in ["#f0f0f0","salmon","#ff8c00"] else "black"
        st.markdown(
            f"<div style='background-color:{bg_color};padding:4px;border-radius:4px;border:{border_style};color:{text_color};'><b>{html.escape(speaker)}:</b> {text_html}</div>",
            unsafe_allow_html=True
        )

def color_speaker_row(row):
    """Retorna una lista de strings de estilo CSS para colorear la fila según el orador."""
    s = row["speaker"].strip().lower()
    style = ""
    if s == "eva": style = "background-color: mediumslateblue; color: white"
    elif s == "nacho": style = "background-color: salmon; color: black"
    elif s == "lala": style = "background-color: #FF8C00; color: black"
    elif s == "desconocido": style = "background-color: #E0E0E0; color: black"
    # Devuelve el estilo replicado para cada columna
    return [style] * len(row)

# -------------------------------
# UI: AUDIO
# -------------------------------
st.header("1) Cortar audio (.m4a) en fragmentos")
col1, col2 = st.columns([2,1])
with col1:
    uploaded = st.file_uploader("Sube un archivo de audio", type=["m4a","mp3","wav","ogg","flac"])
    segment_minutes = st.number_input("Duración de cada fragmento (minutos)", min_value=1, max_value=180, value=30)
    if uploaded and st.button("Procesar audio y generar fragmentos"):
        audio_bytes = uploaded.read()
        with st.spinner("Cortando audio..."):
            try:
                segments = split_audio(audio_bytes, uploaded.name, segment_seconds=int(segment_minutes*60))
                st.session_state['audio_segments'] = segments
                st.success(f"Generados {len(segments)} fragmentos")
            except Exception as e:
                st.error(f"Error procesando audio: {str(e)}")
                st.info("Asegúrate de que FFMPEG esté instalado y accesible. **Esta función puede fallar en Streamlit Cloud por falta de dependencias**.")
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

# -------------------------------
# UI: TRANSCRIPCIONES
# -------------------------------
st.header("2) Leer transcripciones desde GitHub")

# Mostrar estado del token de GitHub
token_status = _get_github_headers().get("Authorization")
if token_status:
    st.success("✅ Token de GitHub configurado")
else:
    st.warning("⚠️ Sin token de GitHub - Límite de 60 requests/hora")

repo_col, _ = st.columns(2)
with repo_col:
    gh_url = st.text_input(
        "Repo público GitHub (carpeta transcripciones)",
        value="https://github.com/jarconett/c_especiales/"
    )

    # Cargar transcripciones desde GitHub
    if gh_url and ('trans_files' not in st.session_state or st.button("📥 Recargar archivos .txt desde GitHub")):
        with st.spinner("Cargando archivos .txt desde GitHub..."):
            # Cargar transcripciones principales
            trans_files = read_txt_files_from_github(gh_url, path="transcripciones")
            if trans_files:
                st.session_state['trans_files'] = trans_files
                st.session_state['trans_df'] = build_transcriptions_dataframe(trans_files)
                st.success(f"✅ Transcripciones: {len(trans_files)} archivos, {len(st.session_state['trans_df'])} bloques")
            
            # Cargar archivos de spoti como respaldo
            spoti_files = read_txt_files_from_github(gh_url, path="spoti")
            if spoti_files:
                st.session_state['spoti_files'] = spoti_files
                st.session_state['spoti_df'] = build_spoti_dataframe(spoti_files)
                st.success(f"✅ Archivos Spoti: {len(spoti_files)} archivos, {len(st.session_state['spoti_df'])} bloques")
            else:
                st.warning("⚠️ No se encontraron archivos en la carpeta 'spoti'")

# Alternativa: Cargar archivos locales
st.markdown("### 📁 Alternativa: Cargar archivos locales")
uploaded_files = st.file_uploader(
    "Sube archivos .txt de transcripciones", 
    type=['txt'], 
    accept_multiple_files=True,
    help="Si GitHub no funciona, puedes subir los archivos .txt directamente"
)

if uploaded_files and st.button("📥 Procesar archivos locales"):
    files = []
    for uploaded_file in uploaded_files:
        content = uploaded_file.read().decode("utf-8", errors="ignore")
        files.append({"name": uploaded_file.name, "content": content})
    
    if files:
        st.session_state['trans_files'] = files
        st.session_state['trans_df'] = build_transcriptions_dataframe(files)
        st.success(f"Cargados {len(files)} archivos locales y DataFrame con {len(st.session_state['trans_df'])} bloques")

# Solo mostrar controles si hay transcripciones
if 'trans_df' in st.session_state:
    df = st.session_state['trans_df']

    st.markdown("---")
    
    # -------------------------------
    # UI: BÚSQUEDA
    # -------------------------------
    st.header("3) Búsqueda Literal por Palabras Clave")
    search_query = st.text_input("Ingresa tu término de búsqueda", "")
    
    if search_query:
        # Pestañas de resultados
        tab1, tab2 = st.tabs(["Transcripciones principales", "Archivos Spoti"])

        # --- Transcripciones principales ---
        with tab1:
            st.markdown("#### 🔎 Búsqueda Literal: Transcripciones")
            
            # Controles de búsqueda literal
            col_search_type, col_search_match = st.columns([1, 1])
            with col_search_type:
                use_regex = st.checkbox("Usar Regex", value=False)
            with col_search_match:
                all_words = st.checkbox("Buscar TODAS las palabras (AND)", value=True, disabled=use_regex)
            
            if st.button("Buscar en Transcripciones"):
                with st.spinner("Buscando coincidencias literales..."):
                    search_results = search_transcriptions(
                        st.session_state['trans_df'], 
                        search_query, 
                        use_regex=use_regex, 
                        all_words=all_words
                    )
                
                if search_results.empty:
                    st.info("No se encontraron resultados en las transcripciones principales.")
                else:
                    st.success(f"Se encontraron {len(search_results)} coincidencias.")
                    
                    # 💡 APLICAR ESTILO DE FILA AQUÍ
                    st.dataframe(
                        search_results[['file', 'speaker', 'match_preview', 'block_index']]
                            .style.apply(color_speaker_row, axis=1),
                        column_config={
                            "file": "Archivo",
                            "speaker": "Orador",
                            "match_preview": st.column_config.TextColumn("Vista Previa (Literal)"),
                            "block_index": "Bloque"
                        },
                        hide_index=True,
                        use_container_width=True
                    )
                    
                    st.markdown("---")
                    st.markdown("### 📚 Contexto de los resultados (Transcripciones)")
                    for i, row in search_results.iterrows():
                        file = row['file']
                        block_idx = row['block_index']
                        query_terms = [t for t in normalize_text(search_query).split() if t]
                        
                        st.subheader(f"Archivo: {file} - Bloque: {block_idx} (Orador: {row['speaker']})")
                        show_context(st.session_state['trans_df'], file, block_idx, query_terms, context=2)
                        st.markdown("---")


        # --- Archivos Spoti ---
        with tab2:
            if 'spoti_df' in st.session_state and not st.session_state['spoti_df'].empty:
                st.markdown("#### 🔎 Búsqueda Literal: Archivos Spoti")
                
                # Controles de búsqueda literal
                col_search_type_spoti, col_search_match_spoti = st.columns([1, 1])
                with col_search_type_spoti:
                    use_regex_spoti = st.checkbox("Usar Regex Spoti", value=False)
                with col_search_match_spoti:
                    all_words_spoti = st.checkbox("Buscar TODAS las palabras Spoti (AND)", value=True, disabled=use_regex_spoti)
                
                if st.button("Buscar en Archivos Spoti"):
                    with st.spinner("Buscando coincidencias literales en Spoti..."):
                        search_results_spoti = search_transcriptions(
                            st.session_state['spoti_df'], 
                            search_query, 
                            use_regex=use_regex_spoti, 
                            all_words=all_words_spoti
                        )
                    
                    if search_results_spoti.empty:
                        st.info("No se encontraron resultados en los archivos Spoti.")
                    else:
                        st.success(f"Se encontraron {len(search_results_spoti)} coincidencias en Spoti.")
                        
                        # 💡 APLICAR ESTILO DE FILA AQUÍ (Spoti)
                        st.dataframe(
                            search_results_spoti[['file', 'speaker', 'match_preview', 'block_index']]
                                .style.apply(color_speaker_row, axis=1),
                            column_config={
                                "file": "Archivo",
                                "speaker": "Orador",
                                "match_preview": st.column_config.TextColumn("Vista Previa (Literal)"),
                                "block_index": "Bloque"
                            },
                            hide_index=True,
                            use_container_width=True
                        )
                        
                        st.markdown("---")
                        st.markdown("### 📚 Contexto de los resultados (Spoti)")
                        for i, row in search_results_spoti.iterrows():
                            file = row['file']
                            block_idx = row['block_index']
                            query_terms = [t for t in normalize_text(search_query).split() if t]
                            
                            st.subheader(f"Archivo: {file} - Bloque: {block_idx} (Orador: {row['speaker']})")
                            show_context(st.session_state['spoti_df'], file, block_idx, query_terms, context=2)
                            st.markdown("---")
            else:
                st.info("No hay archivos en la carpeta 'spoti' cargados.")




