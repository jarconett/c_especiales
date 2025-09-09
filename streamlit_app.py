import streamlit as st
import pandas as pd
import requests
import os
from io import BytesIO
import subprocess

st.title("Cortar audio .m4a y buscar en transcripciones")

# Subida de archivo
uploaded_file = st.file_uploader("Sube tu archivo .m4a", type=["m4a"])
if uploaded_file:
    audio_path = f"temp_audio.m4a"
    with open(audio_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.audio(audio_path, format="audio/m4a")

    # Duración total usando ffmpeg
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "format=duration", "-of",
         "default=noprint_wrappers=1:nokey=1", audio_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )
    duration_sec = float(result.stdout)
    st.write(f"Duración del audio: {duration_sec/60:.2f} minutos")

    # Dividir en fragmentos de 30 min
    chunk_dur = 30*60
    os.makedirs("chunks", exist_ok=True)
    num_chunks = int(duration_sec//chunk_dur) + 1

    for i in range(num_chunks):
        start = i*chunk_dur
        end = min((i+1)*chunk_dur, duration_sec)
        output_file = f"chunks/chunk_{i+1}.m4a"
        subprocess.run([
            "ffmpeg", "-y", "-i", audio_path,
            "-ss", str(start), "-to", str(end),
            "-c", "copy", output_file
        ])
        with open(output_file, "rb") as f:
            st.download_button(f"Descargar {os.path.basename(output_file)}", f.read(), file_name=os.path.basename(output_file), mime="audio/m4a")

# Leer transcripciones
GITHUB_RAW_URL = "https://raw.githubusercontent.com/<usuario>/<repositorio>/main/transcripciones/"
files_list = ["file1.txt","file2.txt"]

def get_transcriptions():
    data = []
    for fname in files_list:
        r = requests.get(GITHUB_RAW_URL + fname)
        if r.status_code == 200:
            lines = r.text.splitlines()
            orador = None
            for line in lines:
                line = line.strip()
                if line.startswith("[") and line.endswith("]"):
                    orador = line[1:-1]
                elif line:
                    data.append({"orador": orador, "texto": line})
    return pd.DataFrame(data)

st.subheader("Transcripciones")
df = get_transcriptions()
st.dataframe(df)

# Búsqueda
st.subheader("Buscar en transcripciones")
query = st.text_input("Palabra o frase:")
if query:
    results = df[df["texto"].str.contains(query, case=False, na=False)]
    st.write(f"Se encontraron {len(results)} coincidencias:")
    st.dataframe(results)
