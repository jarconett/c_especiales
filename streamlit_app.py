import os
import io
import tempfile
import math
from moviepy.editor import AudioFileClip

# Ruta a tu ffmpeg local
FFMPEG_BIN = r"C:\Users\Javier\Downloads\ffmpeg.exe"
os.environ["IMAGEIO_FFMPEG_EXE"] = FFMPEG_BIN  # MoviePy usa imageio-ffmpeg

def split_audio(audio_bytes: bytes, filename: str, segment_seconds: int = 1800):
    """
    Divide un archivo de audio en segmentos de segment_seconds usando MoviePy.
    Retorna lista de dicts: {"name": ..., "bytes": ...}
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".m4a") as tmpfile:
        tmpfile.write(audio_bytes)
        tmp_path = tmpfile.name

    clip = AudioFileClip(tmp_path)
    duration = clip.duration  # segundos
    n_segments = math.ceil(duration / segment_seconds)
    segments = []

    for i in range(n_segments):
        start = i * segment_seconds
        end = min((i + 1) * segment_seconds, duration)
        seg_clip = clip.subclip(start, end)

        seg_name = f"{filename.rsplit('.',1)[0]}_part{i+1}.m4a"
        with tempfile.NamedTemporaryFile(delete=False, suffix=".m4a") as seg_tmp:
            seg_clip.write_audiofile(
                seg_tmp.name,
                codec="aac",
                verbose=True,     # muestra progreso
                logger=None       # evita problemas con stdout en Python 3.13
            )
            seg_tmp.seek(0)
            seg_bytes = open(seg_tmp.name, "rb").read()

        segments.append({"name": seg_name, "bytes": seg_bytes})
        os.unlink(seg_tmp.name)

    clip.close()
    os.unlink(tmp_path)
    return segments
