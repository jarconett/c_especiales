import streamlit as st
from moviepy.editor import AudioFileClip
import io, math, pandas as pd, re, requests, tempfile, os, base64, unicodedata, html
from typing import List

st.set_page_config(page_title="Audio splitter + Transcriptions search", layout="wide")
st.title("⚠️⚠️⚠️⚠️⚠️⚠️TEST BRANCH⚠️⚠️⚠️⚠️⚠️💰🔊 A ganar billete 💵 💶 💴")

# -----------------------------------
# Helper functions
# -----------------------------------
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
        text_color = "white" if bg_color.lower() not in ["#f0f0f0", "salmon", "#ff8c00"] else "black"
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
# MAIN UI
# -------------------------------
st.header("Buscar en transcripciones")
# Simulación de DataFrame (reemplaza por tus transcripciones reales)
data = {
    "file": ["archivo1","archivo1","archivo2"],
    "speaker": ["Ana","Luis","Ana"],
    "text": ["La dinámica fue interesante", "Reunión muy dinámica", "Otra cosa distinta"],
    "block_index": [0,1,0]
}
df = pd.DataFrame(data)

query = st.text_input("Palabra o frase a buscar")
use_regex = st.checkbox("Usar regex", value=False)
match_mode = st.radio("Modo de coincidencia", ["Todas las palabras", "Alguna palabra"], index=0)
all_words = (match_mode=="Todas las palabras")
speaker_filter = st.selectbox("Filtrar por orador", options=["(todos)"] + sorted(df['speaker'].unique().tolist()))

res = pd.DataFrame()  # inicialización para evitar NameError

if st.button("Buscar"):
    res = search_transcriptions(df, query, use_regex, all_words=all_words)
    query_terms = [t for t in normalize_text(query).split() if t]
    if speaker_filter != "(todos)":
        res = res[res['speaker'].str.lower() == speaker_filter.lower()]

    if res.empty:
        st.warning("No se encontraron coincidencias.")
    else:
        st.success(f"Encontradas {len(res)} coincidencias")
        st.dataframe(res[['file','speaker','match_preview']].style.apply(color_speaker_row, axis=1), use_container_width=True)
        for i, row in res.iterrows():
            with st.expander(f"{i+1}. {row['speaker']} — {row['file']} (bloque {row['block_index']})", expanded=False):
                show_context(df, row['file'], row['block_index'], query_terms, context=4)
