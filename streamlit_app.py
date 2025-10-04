import streamlit as st
from moviepy.editor import AudioFileClip
import io, math, pandas as pd, re, requests, tempfile, os, base64, unicodedata, html
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List
from tqdm import tqdm
import numpy as np

st.set_page_config(page_title="Audio splitter + Transcriptions search", layout="wide")
st.title("⚠️⚠️⚠️⚠️TEST BRANCH⚠️⚠️⚠️⚠️💰🔊 A ganar billete 💵 💶 💴")

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
    if os.path.exists(path) or path == "ffmpeg":
        FFMPEG_BIN = path
        break

if FFMPEG_BIN:
    os.environ["IMAGEIO_FFMPEG_EXE"] = FFMPEG_BIN
else:
    st.warning("⚠️ FFMPEG no encontrado. Algunas funciones de audio pueden no funcionar.")

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
        st.error(f"Error fetching GitHub contents: {resp.status_code}")
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

# -------------------------------
# FUNCIONES BÚSQUEDA
# -------------------------------
def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return text.lower()

def norm_and_mapping(orig: str):
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
            s_orig = mapping[idx]
            e_orig = mapping[idx + len(term) - 1] + 1
            spans.append((s_orig, e_orig))
            start = idx + len(term)
    if not spans: return html.escape(orig_text)
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
    return res_df[expected_cols]

# -------------------------------
# FUNCIONES EMBEDDINGS
# -------------------------------

@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()

@st.cache_data(show_spinner=False)
def compute_embeddings(df, model_name="AkDieg0/audit_distilbeto", batch_size=64):
    """
    Genera embeddings para el texto en df['text'] usando SentenceTransformer.
    Procesa en lotes para mayor velocidad y muestra progreso en Streamlit.
    """
    import streamlit as st

    # Cargar modelo
    st.write(f"🧠 Cargando modelo: `{model_name}` ...")
    model = SentenceTransformer(model_name)

    # Asegurar que hay columna 'text'
    if 'text' not in df.columns:
        st.error("❌ No se encontró columna 'text' en el DataFrame.")
        return df

    # Limpiar embeddings previos (por si el modelo cambió)
    if 'embedding' in df.columns:
        df = df.drop(columns=['embedding'])

    texts = df['text'].astype(str).tolist()
    total = len(texts)
    embeddings = []

    progress_bar = st.progress(0)
    status_text = st.empty()

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = texts[start:end]

        # Codificar el lote
        batch_embeds = model.encode(batch, show_progress_bar=False, normalize_embeddings=True)
        embeddings.extend(batch_embeds)

        # Actualizar progreso
        progress = int((end / total) * 100)
        progress_bar.progress(progress)
        status_text.text(f"Procesando textos {end}/{total}...")

    # Convertir a array y asignar al DataFrame
    df["embedding"] = embeddings

    progress_bar.empty()
    status_text.empty()
    st.success(f"✅ Embeddings generados correctamente ({len(df)} bloques)")

    return df

def semantic_search(df, query, top_k=10):
    if df.empty or not query:
        return pd.DataFrame(columns=["file","speaker","text","block_index","score"])
    
    query_emb = embedder.encode([query], normalize_embeddings=True)
    all_embs = np.vstack(df["embedding"].to_numpy())
    sims = cosine_similarity(query_emb, all_embs)[0]
    
    df_res = df.copy()
    df_res["score"] = sims
    df_res = df_res.sort_values("score", ascending=False).head(top_k)
    df_res["match_preview"] = df_res["text"].apply(lambda t: highlight_preview(t, normalize_text(query).split()))
    return df_res[["file","speaker","text","block_index","score","match_preview"]]



# -------------------------------
# CONTEXTO
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
        bg_color = {"eva":"mediumslateblue","nacho":"salmon","lala":"#FF8C00"}.get(speaker.lower(), "#f0f0f0")
        border_style = "2px solid yellow" if i==idx else "none"
        text_color = "white" if bg_color.lower() not in ["#f0f0f0","salmon","#ff8c00"] else "black"
        st.markdown(
            f"<div style='background-color:{bg_color};padding:4px;border-radius:4px;border:{border_style};color:{text_color};'><b>{html.escape(speaker)}:</b> {text_html}</div>",
            unsafe_allow_html=True
        )

def color_speaker_row(row):
    s = row["speaker"].strip().lower()
    if s == "eva": return ["background-color: mediumslateblue"]*len(row)
    if s == "nacho": return ["background-color: salmon"]*len(row)
    if s == "lala": return ["background-color: #FF8C00"]*len(row)
    return [""]*len(row)

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
                st.info("Asegúrate de que FFMPEG esté instalado y accesible.")
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

repo_col, _ = st.columns(2)
with repo_col:
    gh_url = st.text_input(
        "Repo público GitHub (carpeta transcripciones)",
        value="https://github.com/jarconett/c_especiales/"
    )

    # Cargar transcripciones desde GitHub
    if gh_url and ('trans_files' not in st.session_state or st.button("📥 Recargar archivos .txt desde GitHub")):
        with st.spinner("Cargando archivos .txt desde GitHub..."):
            files = read_txt_files_from_github(gh_url, path="transcripciones")
            if files:
                st.session_state['trans_files'] = files
                st.session_state['trans_df'] = build_transcriptions_dataframe(files)
                st.session_state['has_embeddings'] = False
                st.session_state['embed_model'] = None  # reiniciar modelo activo
                st.success(f"Cargados {len(files)} archivos y DataFrame con {len(st.session_state['trans_df'])} bloques")

# Solo mostrar controles si hay transcripciones
if 'trans_df' in st.session_state:
    df = st.session_state['trans_df']

    # --- Selección de modelo de embeddings ---
    st.markdown("### 🔤 Modelo de embeddings")
    model_choice = st.selectbox("Modelo embeddings", [
        "AkDieg0/audit_distilbeto (DistilBERT personalizado en español)",
        "fredymad/albeto_Pfinal_4CLASES_2e-5_16_2 (ALBERT personalizado en español)",
        "distiluse-base-multilingual-cased (Multilingüe, incluye español)",
        "paraphrase-multilingual-MiniLM-L12-v2 (Multilingüe optimizado)", 
        "all-MiniLM-L6-v2 (Rápido, inglés)",
        "distiluse-base-multilingual-cased-v2 (Multilingüe v2)",
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2 (MPNet multilingüe)"
    ],
        help="Modelos de SentenceTransformers para generar embeddings semánticos. Los modelos personalizados están optimizados para español."
    )

    # Mapear selección a nombre real
    model_map = {
        "AkDieg0/audit_distilbeto (DistilBERT personalizado en español)": "AkDieg0/audit_distilbeto",
        "fredymad/albeto_Pfinal_4CLASES_2e-5_16_2 (ALBERT personalizado en español)": "fredymad/albeto_Pfinal_4CLASES_2e-5_16_2",
        "distiluse-base-multilingual-cased (Multilingüe, incluye español)": "distiluse-base-multilingual-cased",
        "paraphrase-multilingual-MiniLM-L12-v2 (Multilingüe optimizado)": "paraphrase-multilingual-MiniLM-L12-v2",
        "all-MiniLM-L6-v2 (Rápido, inglés)": "all-MiniLM-L6-v2",
        "distiluse-base-multilingual-cased-v2 (Multilingüe v2)": "distiluse-base-multilingual-cased-v2",
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2 (MPNet multilingüe)": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    }
    selected_model = model_map[model_choice]

    # --- Control de embeddings ---
    colA, colB = st.columns([2, 1])
    with colA:
        # Estado de embeddings
        if not st.session_state.get('has_embeddings', False) or st.session_state.get('embed_model') != selected_model:
            st.info(f"Generando embeddings con modelo **{selected_model}** (puede tardar unos segundos)...")
            with st.spinner("Creando vectores semánticos..."):
                st.session_state['trans_df'] = compute_embeddings(st.session_state['trans_df'], model_name=selected_model)
                st.session_state['has_embeddings'] = True
                st.session_state['embed_model'] = selected_model
                st.success(f"✅ Embeddings generados con **{selected_model}**")
        else:
            st.success(f"✅ Embeddings ya generados con **{selected_model}**")

    with colB:
        if st.button("🔁 Regenerar embeddings manualmente"):
            with st.spinner(f"Recalculando embeddings con {selected_model}..."):
                st.session_state['trans_df'] = compute_embeddings(st.session_state['trans_df'], model_name=selected_model)
                st.session_state['has_embeddings'] = True
                st.session_state['embed_model'] = selected_model
                st.success("Embeddings recalculados correctamente ✅")



# -------------------------------
# UI: BÚSQUEDA
# -------------------------------
st.header("3) Buscar en transcripciones")
if 'trans_df' in st.session_state and not st.session_state['trans_df'].empty:
    df = st.session_state['trans_df']

    # Entrada de búsqueda
    q_col, opt_col = st.columns([3,1])
    with q_col:
        query = st.text_input("Palabra o frase a buscar")
    with opt_col:
        speaker_filter = st.selectbox("Filtrar por orador", options=["(todos)"] + sorted(df['speaker'].unique().tolist()))

    # Tipo de búsqueda
    search_mode = st.radio("Modo de búsqueda", ["Texto literal", "Semántica", "Híbrida (texto + semántica)"], index=0)

    # Configuración específica
    if search_mode == "Texto literal":
        use_regex = st.checkbox("Usar regex", value=False)
        match_mode = st.radio("Modo de coincidencia", ["Todas las palabras", "Alguna palabra"], index=0)
        all_words = (match_mode == "Todas las palabras")
    elif search_mode == "Semántica":
        threshold = st.slider("Umbral mínimo de similitud", 0.0, 1.0, 0.6, 0.05)
        top_k = st.number_input("Máximo de resultados", min_value=1, max_value=100, value=20)
    else:  # Híbrida
        semantic_weight = st.slider("Peso de la similitud semántica", 0.0, 1.0, 0.7, 0.05)
        threshold = st.slider("Umbral mínimo del score combinado", 0.0, 1.0, 0.5, 0.05)
        top_k = st.number_input("Máximo de resultados", min_value=1, max_value=100, value=30)

    # Acción de búsqueda
    if st.button("Buscar"):
        query_terms = [t for t in normalize_text(query).split() if t]

        if search_mode == "Texto literal":
            res = search_transcriptions(df, query, use_regex, all_words=all_words)

        elif search_mode == "Semántica":
            res = semantic_search(df, query, top_k=top_k)
            res = res[res["score"] >= threshold]

        else:  # ---- HÍBRIDA ----
            # Parte semántica
            sem_res = semantic_search(df, query, top_k=len(df))
            sem_scores = dict(zip(zip(sem_res.file, sem_res.block_index), sem_res.score))

            # Parte literal
            lit_res = search_transcriptions(df, query, use_regex=False, all_words=False)
            lit_boost = set(zip(lit_res.file, lit_res.block_index))

            # Combinar scores
            df_comb = df.copy()
            df_comb["sem_score"] = df_comb.apply(lambda r: sem_scores.get((r.file, r.block_index), 0), axis=1)
            df_comb["lit_score"] = df_comb.apply(lambda r: 1.0 if (r.file, r.block_index) in lit_boost else 0.0, axis=1)
            df_comb["combined_score"] = semantic_weight * df_comb["sem_score"] + (1 - semantic_weight) * df_comb["lit_score"]

            df_comb = df_comb[df_comb["combined_score"] >= threshold].sort_values("combined_score", ascending=False).head(top_k)
            df_comb["score"] = df_comb["combined_score"]
            df_comb["match_preview"] = df_comb["text"].apply(lambda t: highlight_preview(t, query_terms))
            res = df_comb[["file","speaker","text","block_index","score","match_preview"]]

        # Filtrar por orador
        if speaker_filter != "(todos)":
            res = res[res['speaker'].str.lower() == speaker_filter.lower()]

        # Mostrar resultados
        if res.empty:
            st.warning("No se encontraron coincidencias.")
        else:
            st.success(f"Encontradas {len(res)} coincidencias")

            # Mostrar score si aplica
            cols_to_show = ['file','speaker','match_preview']
            if "score" in res.columns:
                cols_to_show.insert(2, 'score')

            st.dataframe(
                res[cols_to_show]
                .style.format({'score': '{:.3f}'})
                .apply(color_speaker_row, axis=1),
                use_container_width=True
            )

            # Mostrar contexto
            for i, row in res.iterrows():
                score_str = f" (score {row['score']:.3f})" if "score" in row else ""
                with st.expander(f"{i+1}. {row['speaker']} — {row['file']} (bloque {row['block_index']}){score_str}", expanded=False):
                    show_context(df, row['file'], row['block_index'], query_terms, context=4)
else:
    st.info("Carga las transcripciones en el paso 2 para comenzar a buscar.")

