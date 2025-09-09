import streamlit as st
from moviepy.editor import AudioFileClip
import io, math, pandas as pd, re, requests, tempfile, os, time
from typing import List

st.set_page_config(page_title="Audio splitter + Transcriptions search", layout="wide")
st.title("🔊 Audio splitter (.m4a) + Transcriptions search (moviepy)")

# --- Helper functions ---

def split_audio(audio_bytes: bytes, filename: str, segment_seconds: int = 1800) -> List[dict]:
    """Divide audio en fragmentos con barra de progreso + ETA"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".m4a") as tmpfile:
        tmpfile.write(audio_bytes)
        tmp_path = tmpfile.name

    clip = AudioFileClip(tmp_path)
    duration = clip.duration
    n_segments = math.ceil(duration / segment_seconds)
    segments = []
    progress_bar = st.progress(0, text="Preparando...")
    start_time = time.time()

    for i in range(n_segments):
        seg_start = time.time()
        start = i * segment_seconds
        end = min((i + 1) * segment_seconds, duration)
        seg_clip = clip.subclip(start, end)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".m4a") as seg_tmp:
            seg_clip.write_audiofile(seg_tmp.name, codec="aac", verbose=False, logger=None)
            seg_tmp.seek(0)
            seg_bytes = open(seg_tmp.name, "rb").read()

        seg_name = f"{filename.rsplit('.',1)[0]}_part{i+1}.m4a"
        segments.append({"name": seg_name, "bytes": seg_bytes})

        seg_clip.close()
        os.unlink(seg_tmp.name)

        elapsed = time.time() - start_time
        avg_per_segment = elapsed / (i + 1)
        remaining = (n_segments - (i + 1)) * avg_per_segment
        eta_min = int(remaining // 60)
        eta_sec = int(remaining % 60)

        progress_bar.progress((i + 1) / n_segments, text=f"Cortando fragmento {i+1}/{n_segments} — ETA: {eta_min:02d}:{eta_sec:02d}")

    clip.close()
    os.unlink(tmp_path)
    progress_bar.empty()
    return segments

def read_txt_files_from_github(repo_url: str, path: str = "transcripciones") -> List[dict]:
    """Fetch .txt files from GitHub folder"""
    m = re.match(r"https?://github.com/([^/]+)/([^/]+)", repo_url)
    if not m:
        st.error("URL de repo no válida")
        return []
    owner, repo = m.group(1), m.group(2).replace('.git','')
    api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
    resp = requests.get(api_url)
    if resp.status_code != 200:
        st.error(f"Error fetching GitHub contents: {resp.status_code}")
        return []
    files = resp.json()
    data = []
    for f in files:
        if f.get('type')=='file' and f.get('name','').lower().endswith('.txt'):
            raw_url = f.get('download_url')
            txt_resp = requests.get(raw_url)
            if txt_resp.status_code == 200:
                data.append({"name": f['name'], "content": txt_resp.text})
    if not data:
        st.warning("No se encontraron archivos .txt en la carpeta transcripciones/")
    return data

def parse_transcription_text(name: str, text: str) -> pd.DataFrame:
    pattern = re.compile(r"\[([^\]]+)\]\s*(.*?)((?=\[)|$)", re.S)
    rows = []
    for idx, m in enumerate(pattern.finditer(text)):
        speaker = m.group(1).strip()
        content = m.group(2).strip().replace('\r\n','\n')
        content = re.sub(r"\n+"," ",content)
        content = re.sub(r"\s+"," ",content).strip()
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

def search_transcriptions(df: pd.DataFrame, query: str, use_regex: bool=False) -> pd.DataFrame:
    if df.empty or not query:
        return pd.DataFrame(columns=["file","speaker","text","block_index","match_preview"])
    results = []
    flags = re.IGNORECASE
    for _, row in df.iterrows():
        try:
            if use_regex:
                if re.search(query,row['text'], flags):
                    results.append(row.to_dict())
            else:
                if query.lower() in row['text'].lower():
                    results.append(row.to_dict())
        except re.error:
            st.error("Regex inválida")
            return pd.DataFrame()
    if not results:
        return pd.DataFrame(columns=["file","speaker","text","block_index","match_preview"])
    res_df = pd.DataFrame(results)
    def make_preview(text):
        idx = text.lower().find(query.lower()) if not use_regex else None
        if idx is None and use_regex:
            try:
                m = re.search(query, text, flags)
                if m:
                    idx = max(m.start()-30,0)
                    return ("..." + text[idx:idx+160] + "...")
            except:
                return text[:160]
        if idx is not None:
            start = max(idx-30,0)
            return ("..." + text[start:start+160] + "...")
        return text[:160]
    res_df['match_preview'] = res_df['text'].apply(make_preview)
    return res_df[['file','speaker','text','block_index','match_preview']]

def color_speaker_row(row):
    speaker = row["speaker"].strip().lower()
    if speaker == "eva":
        return ["background-color: palegreen"]*len(row)
    elif speaker == "nacho":
        return ["background-color: salmon"]*len(row)
    elif speaker == "lala":
        return ["background-color: lightblue"]*len(row)
    else:
        return [""]*len(row)

# --- UI: Audio splitting ---
st.header("1) Cortar audio (.m4a) en fragmentos de 30 minutos")
col1, col2 = st.columns([2,1])
with col1:
    uploaded = st.file_uploader("Sube un archivo de audio", type=["m4a","mp3","wav","ogg","flac"])
    segment_minutes = st.number_input("Duración de cada fragmento (minutos)", min_value=1, max_value=180, value=30)
    if uploaded and st.button("Procesar audio"):
        audio_bytes = uploaded.read()
        with st.spinner("Cortando audio..."):
            segments = split_audio(audio_bytes, uploaded.name, segment_seconds=int(segment_minutes*60))
        if segments:
            st.success(f"Generados {len(segments)} fragmentos")
            for seg in segments:
                st.download_button(label=f"Descargar {seg['name']}", data=seg['bytes'], file_name=seg['name'])

with col2:
    st.markdown("""
    **Consejos**:
    - [Cortar audio online](https://mp3cut.net/es)
    - [Transcripción automática](https://turboscribe.ai/)
    """, unsafe_allow_html=True)

st.markdown("---")

# --- UI: Transcriptions loader ---
st.header("2) Leer transcripciones (GitHub o .txt)")
repo_col, upload_col = st.columns(2)

with repo_col:
    gh_url = st.text_input("Repo público GitHub (carpeta transcripciones)", 
                           value="https://github.com/jarconett/c_especiales/")

    # Carga automática al iniciar si no hay session_state
    if 'trans_files' not in st.session_state and gh_url:
        with st.spinner("Cargando archivos .txt desde GitHub..."):
            files = read_txt_files_from_github(gh_url, path="transcripciones")
            if files:
                st.session_state['trans_files'] = files
                st.success(f"Cargados {len(files)} archivos desde GitHub")

    if st.button("Recargar archivos .txt desde GitHub", key="gh_load"):
        with st.spinner("Leyendo archivos desde GitHub..."):
            files = read_txt_files_from_github(gh_url, path="transcripciones")
            if files:
                st.session_state['trans_files'] = files
                st.success(f"Cargados {len(files)} archivos desde GitHub")

with upload_col:
    uploaded_txts = st.file_uploader("O sube archivos .txt", type=['txt'], accept_multiple_files=True)
    if uploaded_txts:
        files = []
        for f in uploaded_txts:
            try:
                txt = f.read().decode('utf-8')
            except:
                txt = f.read().decode('latin-1')
            files.append({"name": f.name, "content": txt})
        st.session_state['trans_files'] = files
        st.success(f"Se han cargado {len(files)} archivos .txt")

# Build DataFrame
if 'trans_files' in st.session_state:
    files = st.session_state['trans_files']
    if files and st.button("Construir DataFrame de transcripciones", key='build_df'):
        with st.spinner("Parseando transcripciones..."):
            df = build_transcriptions_dataframe(files)
            st.session_state['trans_df'] = df
            st.success(f"DataFrame con {len(df)} bloques")
            st.dataframe(df[['file','speaker','text']].head(200))

st.markdown("---")

# --- Search UI ---
st.header("3) Buscar en transcripciones")
if 'trans_df' in st.session_state:
    df = st.session_state['trans_df']
    q_col, opt_col = st.columns([3,1])
    with q_col:
        query = st.text_input("Palabra o frase a buscar")
    with opt_col:
        use_regex = st.checkbox("Usar regex", value=False)
        speaker_filter = st.selectbox("Filtrar por orador", options=["(todos)"] + sorted(df['speaker'].unique().tolist()))
    
    if st.button("Buscar"):
        with st.spinner("Buscando..."):
            res = search_transcriptions(df, query, use_regex)
            if speaker_filter != "(todos)":
                res = res[res['speaker'] == speaker_filter]
            if res.empty:
                st.warning("No se encontraron coincidencias.")
            else:
                st.success(f"Encontradas {len(res)} coincidencias")

                # Leyenda colores
                st.markdown("""
                    <div style="display:flex; gap:1em; align-items:center; margin-bottom:1em;">
                        <div style="background-color:palegreen; padding:4px 10px; border-radius:4px;">Eva</div>
                        <div style="background-color:salmon; padding:4px 10px; border-radius:4px;">Nacho</div>
                        <div style="background-color:lightblue; padding:4px 10px; border-radius:4px;">Lala</div>
                    </div>
                    """, unsafe_allow_html=True)

                # DataFrame coloreado
                styled = res[['file','speaker','match_preview']].style.apply(color_speaker_row, axis=1)
                st.dataframe(styled, use_container_width=True)

                # Expanders con texto coloreado
                for i, row in res.iterrows():
                    color = ""
                    if row['speaker'].lower() == "eva":
                        color = "background-color: palegreen"
                    elif row['speaker'].lower() == "nacho":
                        color = "background-color: salmon"
                    elif row['speaker'].lower() == "lala":
                        color = "background-color: lightblue"
                    with st.expander(f"{row['speaker']} — {row['file']} (bloque {row['block_index']})"):
                        st.markdown(f"<div style='{color}; padding:0.5em; border-radius:6px;'>{row['text']}</div>",
                                    unsafe_allow_html=True)
