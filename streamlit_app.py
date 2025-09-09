import streamlit as st
from moviepy.editor import AudioFileClip
import pandas as pd
import requests
from io import BytesIO
import os

st.set_page_config(page_title="Audio Split & Transcription Search", layout="wide")

st.title("Cortar audio y buscar en transcripciones")

# -------------------------------
# 1️⃣ Subida de audio y corte
# -------------------------------
uploaded_file = st.file_uploader("Sube tu archivo .m4a", type=["m4a"])

if uploaded_file:
    st.audio(uploaded_file, format="audio/m4a")
    audio = AudioFileClip(uploaded_file.name) if hasattr(uploaded_file, "name") else AudioFileClip(BytesIO(uploaded_file.read()))
    duration_min = audio.duration / 60
    st.write(f"Duración del audio: {duration_min:.2f} minutos")

    # Duración de fragmento
    chunk_duration_min = 30
    chunk_duration_sec = chunk_duration_min * 60
    chunks = int(audio.duration // chunk_duration_sec) + 1

    st.write(f"El audio se dividirá en {chunks} fragmentos de {chunk_duration_min} minutos aproximadamente.")

    # Carpeta temporal
    os.makedirs("chunks", exist_ok=True)
    download_links = []

    for i in range(chunks):
        start = i * chunk_duration_sec
        end = min((i + 1) * chunk_duration_sec, audio.duration)
        chunk = audio.subclip(start, end)
        filename = f"chunks/chunk_{i+1}.m4a"
        chunk.write_audiofile(filename, codec="aac", verbose=False, logger=None)
        download_links.append(filename)

    st.success("Fragmentos generados:")
    for file in download_links:
        with open(file, "rb") as f:
            st.download_button(f"Descargar {os.path.basename(file)}", f.read(), file_name=os.path.basename(file), mime="audio/m4a")

# -------------------------------
# 2️⃣ Leer transcripciones desde GitHub
# -------------------------------
GITHUB_RAW_URL = "https://raw.githubusercontent.com/<usuario>/<repositorio>/main/transcripciones/"  # Cambiar según tu repo

def get_transcriptions():
    files_list = ["file1.txt", "file2.txt"]  # Lista de archivos, o se puede hacer scrape dinámico
    data = []
    for fname in files_list:
        url = GITHUB_RAW_URL + fname
        r = requests.get(url)
        if r.status_code == 200:
            lines = r.text.splitlines()
            orador = None
            for line in lines:
                line = line.strip()
                if line.startswith("[") and line.endswith("]"):
                    orador = line[1:-1]
                elif line:
                    data.append({"orador": orador, "texto": line})
    df = pd.DataFrame(data)
    return df

st.subheader("Transcripciones")
df = get_transcriptions()
st.dataframe(df)

# -------------------------------
# 3️⃣ Búsqueda de palabras o frases
# -------------------------------
st.subheader("Buscar en transcripciones")
query = st.text_input("Escribe palabra o frase a buscar:")

if query:
    mask = df["texto"].str.contains(query, case=False, na=False)
    results = df[mask]
    st.write(f"Se encontraron {len(results)} coincidencias:")
    st.dataframe(results)
