import streamlit as st
import io, math, pandas as pd, re, requests, tempfile, os, base64, unicodedata, html, json
import subprocess
from typing import List
from rapidfuzz import fuzz
import hashlib
from datetime import datetime, timedelta, timezone

# Candado global para regeneración de DataFrame (compartido entre sesiones en el mismo servidor)
DF_REGEN_LOCK = False

# Función helper para obtener la zona horaria de España
def get_spain_timezone():
    """Retorna la zona horaria de España (Europe/Madrid)."""
    try:
        # Intentar usar zoneinfo (Python 3.9+)
        from zoneinfo import ZoneInfo
        return ZoneInfo("Europe/Madrid")
    except ImportError:
        # Fallback a pytz si zoneinfo no está disponible
        try:
            import pytz
            return pytz.timezone("Europe/Madrid")
        except ImportError:
            # Si no hay ninguna librería de timezone, usar UTC+1 como aproximación
            return timezone(timedelta(hours=1))

def timestamp_to_spain_time(timestamp):
    """Convierte un timestamp Unix a datetime en hora española."""
    try:
        dt_utc = datetime.fromtimestamp(int(timestamp), tz=timezone.utc)
        spain_tz = get_spain_timezone()
        dt_spain = dt_utc.astimezone(spain_tz)
        return dt_spain
    except Exception:
        # Fallback si hay algún error
        return datetime.fromtimestamp(int(timestamp))

def now_spain():
    """Retorna la hora actual en España."""
    try:
        spain_tz = get_spain_timezone()
        return datetime.now(spain_tz)
    except Exception:
        # Fallback a hora local
        return datetime.now()

# -------------------------------
# SISTEMA DE AUTENTICACIÓN
# -------------------------------
def get_password_hash():
    """Obtiene la contraseña desde secrets o variable de entorno."""
    password = ""
    try:
        # Intentar obtener desde st.secrets (recomendado para Streamlit Cloud)
        # Primero intentar directamente
        if "APP_PASSWORD" in st.secrets:
            password = st.secrets["APP_PASSWORD"]
        # Si no está, intentar bajo [default]
        elif "default" in st.secrets and "APP_PASSWORD" in st.secrets["default"]:
            password = st.secrets["default"]["APP_PASSWORD"]
        # También intentar con get() por si acaso
        elif hasattr(st.secrets, 'get'):
            password = st.secrets.get("APP_PASSWORD", "")
    except Exception:
        pass
    
    # Si no se encontró en secrets, intentar variable de entorno
    if not password:
        password = os.getenv("APP_PASSWORD", "")
    
    return password

def hash_password(password: str) -> str:
    """Genera un hash SHA256 de la contraseña."""
    return hashlib.sha256(password.encode()).hexdigest()

def safe_rerun():
    """Ejecuta un rerun de forma segura."""
    # Usar st.rerun() directamente - debería funcionar ahora que st.set_page_config() 
    # solo se llama una vez al inicio
    try:
        st.rerun()
    except AttributeError:
        # Si st.rerun() no está disponible, intentar experimental_rerun
        try:
            st.experimental_rerun()
        except AttributeError:
            # Si tampoco está disponible, usar un enfoque alternativo
            # Forzar rerun mediante cambio de estado
            if 'force_rerun' not in st.session_state:
                st.session_state['force_rerun'] = 0
            st.session_state['force_rerun'] += 1

def check_password(password: str) -> bool:
    """Verifica si la contraseña es correcta."""
    correct_password = get_password_hash()
    if not correct_password:
        # Si no hay contraseña configurada, usar una por defecto (cambiar en producción)
        # Para producción, configura APP_PASSWORD en Streamlit Cloud secrets
        # Puedes generar el hash con: hashlib.sha256("tu_contraseña".encode()).hexdigest()
        default_hash = "240be518fabd2724ddb6f04eeb1da5967448d7e831c08c8fa822809f74c720a9"  # hash de "admin123"
        return hash_password(password) == default_hash
    
    # Si la contraseña en secrets es un hash (64 caracteres hex), comparar hashes
    if len(correct_password) == 64 and all(c in '0123456789abcdef' for c in correct_password.lower()):
        return hash_password(password) == correct_password.lower()
    
    # Si no es un hash, comparar directamente (para compatibilidad)
    return password == correct_password

# -------------------------------
# CONFIGURACIÓN INICIAL DE LA APP
# -------------------------------
# Configurar la página una sola vez al inicio
st.set_page_config(
    page_title="Audio splitter + Transcriptions search optimizado",
    layout="wide",
    initial_sidebar_state="collapsed",
    page_icon="💵"
)

def show_login_page():
    """Muestra la página de login."""
    # Estilos para la página de login
    st.markdown("""
        <style>
        .login-container {
            max-width: 400px;
            margin: 0 auto;
            padding: 2rem;
            background-color: #f0f2f6;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .login-title {
            text-align: center;
            color: #1f77b4;
            margin-bottom: 2rem;
        }
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
    """, unsafe_allow_html=True)
    
    # Contenedor de login
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<div class="login-container">', unsafe_allow_html=True)
        st.markdown('<h1 class="login-title">🔐 Acceso Protegido</h1>', unsafe_allow_html=True)
        
        password = st.text_input("Contraseña", type="password", key="login_password")
        
        col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
        with col_btn2:
            login_button = st.button("Iniciar Sesión", type="primary", use_container_width=True)
        
        if login_button:
            if check_password(password):
                # Establecer el estado de autenticación
                st.session_state['authenticated'] = True
                st.session_state['password_entered'] = password
                # Mostrar mensaje de éxito
                st.success("✅ Contraseña correcta!")
                # Usar JavaScript para recargar la página de forma segura
                st.markdown(
                    """
                    <script>
                    setTimeout(function() {
                        window.location.reload();
                    }, 500);
                    </script>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.error("❌ Contraseña incorrecta. Intenta nuevamente.")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Información adicional
        st.markdown("---")
        st.caption("💡 Para configurar la contraseña en Streamlit Cloud, agrega 'APP_PASSWORD' en los Secrets de la aplicación.")

# Verificar autenticación
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False

if not st.session_state['authenticated']:
    show_login_page()
    st.stop()

# -------------------------------
# ESTILOS DE LA APP (solo se muestra si está autenticado)
# -------------------------------
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Botón de logout en la esquina superior derecha
col_logout, col_title = st.columns([1, 10])
with col_logout:
    if st.button("🚪 Salir", key="logout_button"):
        st.session_state['authenticated'] = False
        # Limpiar otros estados relacionados si es necesario
        if 'password_entered' in st.session_state:
            del st.session_state['password_entered']
        # Rerun para mostrar la página de login
        st.rerun()

with col_title:
    st.title("💰🔊 A ganar billete 💵 💶 💴")

# --- Helper functions: corte de audio con ffmpeg (subprocess) - sin pydub/moviepy, soporta archivos grandes ---
FFMPEG_BIN = os.environ.get("FFMPEG_BIN", r"C:\Users\Javier\Downloads\ffmpeg.exe" if os.name == "nt" else "ffmpeg")
FFPROBE_BIN = os.environ.get("FFPROBE_BIN", FFMPEG_BIN.replace("ffmpeg", "ffprobe") if isinstance(FFMPEG_BIN, str) and "ffmpeg" in FFMPEG_BIN else "ffprobe")
if not (FFPROBE_BIN and os.path.isfile(FFPROBE_BIN)):
    FFPROBE_BIN = "ffprobe"

def _get_audio_duration_seconds(ffprobe_exe: str, path: str) -> float:
    """Obtiene la duración en segundos con ffprobe."""
    cmd = [
        ffprobe_exe, "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe falló: {result.stderr or result.stdout}")
    return float(result.stdout.strip())

def split_audio(audio_bytes: bytes, filename: str, segment_seconds: int = 1800):
    """Divide el audio en fragmentos usando ffmpeg (subprocess). Compatible con archivos grandes (p. ej. 96 MB)."""
    ffmpeg_exe = FFMPEG_BIN if (FFMPEG_BIN and os.path.isfile(FFMPEG_BIN)) else "ffmpeg"
    ffprobe_exe = FFPROBE_BIN if (FFPROBE_BIN and os.path.isfile(FFPROBE_BIN)) else "ffprobe"

    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "m4a"
    if ext not in ("m4a", "mp3", "wav", "ogg", "flac", "webm"):
        ext = "m4a"

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmpfile:
            tmpfile.write(audio_bytes)
            tmp_path = tmpfile.name

        duration_sec = _get_audio_duration_seconds(ffprobe_exe, tmp_path)
        n_segments = math.ceil(duration_sec / segment_seconds)
        if n_segments > 40:
            raise RuntimeError(f"Demasiados fragmentos ({n_segments}). Reduce la duración o el tamaño del archivo.")

        segments = []
        base_name = filename.rsplit(".", 1)[0] if "." in filename else filename

        for i in range(n_segments):
            start_sec = i * segment_seconds
            duration_seg = min(segment_seconds, duration_sec - start_sec)
            if duration_seg <= 0:
                break
            seg_name = f"{base_name}_part{i+1}.m4a"
            seg_tmp_path = tempfile.mktemp(suffix=".m4a")
            try:
                cmd = [
                    ffmpeg_exe, "-hide_banner", "-loglevel", "error", "-y",
                    "-i", tmp_path, "-ss", str(start_sec), "-t", str(duration_seg),
                    "-c:a", "aac", "-b:a", "128k", seg_tmp_path
                ]
                run_kw = {"capture_output": True, "timeout": 120}
                if os.name == "nt" and getattr(subprocess, "CREATE_NO_WINDOW", None) is not None:
                    run_kw["creationflags"] = subprocess.CREATE_NO_WINDOW
                result = subprocess.run(cmd, **run_kw)
                if result.returncode != 0:
                    err = (result.stderr or result.stdout or b"").decode("utf-8", errors="replace").strip()
                    raise RuntimeError(f"ffmpeg falló: {err[:800]}")
                with open(seg_tmp_path, "rb") as f:
                    seg_bytes = f.read()
                segments.append({"name": seg_name, "bytes": seg_bytes})
            finally:
                if os.path.isfile(seg_tmp_path):
                    try:
                        os.unlink(seg_tmp_path)
                    except OSError:
                        pass

        return segments
    finally:
        if tmp_path and os.path.isfile(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


# --- GitHub utils ---
def _get_github_headers():
    """Obtiene los headers para las peticiones a la API de GitHub."""
    token = ""
    try:
        # Intentar obtener desde st.secrets
        # Primero intentar directamente
        if "GITHUB_TOKEN" in st.secrets:
            token = st.secrets["GITHUB_TOKEN"]
        # Si no está, intentar bajo [default]
        elif "default" in st.secrets and "GITHUB_TOKEN" in st.secrets["default"]:
            token = st.secrets["default"]["GITHUB_TOKEN"]
        # También intentar con get() por si acaso
        elif hasattr(st.secrets, 'get'):
            token = st.secrets.get("GITHUB_TOKEN", "")
    except Exception as e:
        # Si hay un error, intentar de otra forma
        try:
            token = st.secrets.get("GITHUB_TOKEN", "")
        except:
            pass
    
    # Si no se encontró en secrets, intentar variable de entorno
    if not token:
        token = os.getenv("GITHUB_TOKEN", "")
    
    # Limpiar el token de espacios en blanco y comillas
    if token:
        token = str(token).strip()
        # Remover comillas si las tiene (por si se copió con comillas)
        if token.startswith('"') and token.endswith('"'):
            token = token[1:-1]
        if token.startswith("'") and token.endswith("'"):
            token = token[1:-1]
        token = token.strip()
    
    headers = {
        "Accept": "application/vnd.github.v3+json"
    }
    
    if token:
        # GitHub acepta tanto "token" como "Bearer" para la autorización
        # Usar "token" (formato tradicional de GitHub API v3)
        # También funciona "Bearer" pero "token" es más compatible
        headers["Authorization"] = f"token {token}"
    
    return headers, token


def _parse_repo_url(repo_url: str) -> tuple[str, str]:
    """Parsea la URL del repositorio y retorna (owner, repo)."""
    import re as _re
    if repo_url.count("/") == 1 and "/" in repo_url:
        parts = repo_url.split("/")
        return parts[0], parts[1]
    else:
        m = _re.match(r"https?://github.com/([^/]+)/([^/]+)", repo_url)
        if not m:
            return "", ""
        return m.group(1), m.group(2).replace('.git', '')


def _get_file_sha(files: List[dict]) -> dict:
    """Crea un diccionario con SHA de archivos para detectar cambios."""
    file_index = {}
    for file_info in files:
        filename = file_info.get('name', '')
        folder = file_info.get('folder', '').lower()  # Normalizar a minúsculas
        # Usar SHA del archivo si está disponible, o generar hash del contenido
        if 'sha' in file_info:
            file_index[filename] = {
                'sha': file_info['sha'],
                'folder': folder,
                'size': file_info.get('size', 0)
            }
        else:
            # Si no hay SHA, generar hash del contenido
            content = file_info.get('content', '')
            content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
            file_index[filename] = {
                'sha': content_hash,
                'folder': folder,
                'size': len(content)
            }
    return file_index


# Tamaño máximo por archivo para la API de GitHub (~45 MB en binario; base64 ~33% más)
_DF_CHUNK_BYTES = 42 * 1024 * 1024

def _save_dataframe_to_github(repo_url: str, df: pd.DataFrame, file_index: dict, path: str = "data") -> tuple[bool, str]:
    """
    Guarda el DataFrame serializado y el índice en GitHub.
    Si el DataFrame es muy grande, lo divide en partes (transcripciones_df_part_0.pkl, ...).
    Retorna (éxito, mensaje_error)
    """
    owner, repo = _parse_repo_url(repo_url)
    if not owner or not repo:
        return False, "URL de repositorio no válida"
    
    headers, token = _get_github_headers()
    if not token:
        return False, "Se requiere GITHUB_TOKEN para guardar el DataFrame"
    
    try:
        import pickle
        import json
        
        df_bytes = pickle.dumps(df)
        base_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
        
        # Dividir en chunks si supera el límite
        chunks = []
        offset = 0
        while offset < len(df_bytes):
            chunk = df_bytes[offset:offset + _DF_CHUNK_BYTES]
            chunks.append(chunk)
            offset += len(chunk)
        
        n_parts = len(chunks)
        
        # Guardar cada parte
        for i, chunk in enumerate(chunks):
            chunk_b64 = base64.b64encode(chunk).decode('utf-8')
            part_url = f"{base_url}/transcripciones_df_part_{i}.pkl"
            part_data = {
                "message": f"Actualizar DataFrame parte {i+1}/{n_parts} ({len(df)} filas)",
                "content": chunk_b64
            }
            resp = requests.get(part_url, headers=headers)
            if resp.status_code == 200:
                part_data["sha"] = resp.json().get("sha")
            put_resp = requests.put(part_url, headers=headers, json=part_data, timeout=60)
            if put_resp.status_code not in [200, 201]:
                return False, f"Error al guardar parte {i+1}: {put_resp.status_code} - {put_resp.text[:200]}"
        
        # Eliminar partes sobrantes si antes había más (ej. antes 3 partes, ahora 2)
        part_idx = n_parts
        while True:
            part_url = f"{base_url}/transcripciones_df_part_{part_idx}.pkl"
            resp = requests.get(part_url, headers=headers)
            if resp.status_code != 200:
                break
            sha = resp.json().get("sha")
            del_resp = requests.delete(part_url, headers=headers, json={"message": "Eliminar parte obsoleta", "sha": sha}, timeout=30)
            part_idx += 1
        
        # Índice con metadatos y número de partes
        index_data = {
            'file_index': file_index,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'total_files': len(file_index),
            'df_rows': len(df),
            'df_parts': n_parts
        }
        index_json = json.dumps(index_data, indent=2)
        index_base64 = base64.b64encode(index_json.encode('utf-8')).decode('utf-8')
        index_file_url = f"{base_url}/transcripciones_index.json"
        index_data_put = {
            "message": f"Actualizar índice ({len(file_index)} archivos, DataFrame en {n_parts} partes)",
            "content": index_base64
        }
        index_resp = requests.get(index_file_url, headers=headers)
        if index_resp.status_code == 200:
            index_data_put["sha"] = index_resp.json().get("sha")
        index_put = requests.put(index_file_url, headers=headers, json=index_data_put, timeout=30)
        if index_put.status_code not in [200, 201]:
            return False, f"Error al guardar índice: {index_put.status_code} - {index_put.text[:200]}"
        
        return True, ""
    except Exception as e:
        return False, f"Error al guardar DataFrame: {str(e)}"


def _load_settings_from_github(repo_url: str, path: str = "data/settings.json") -> tuple[dict, str]:
    """Carga settings (JSON) desde GitHub. Retorna (settings_dict, error_msg)."""
    owner, repo = _parse_repo_url(repo_url)
    if not owner or not repo:
        return {}, "URL de repositorio no válida"

    headers, token = _get_github_headers()
    if not token:
        return {}, "Se requiere GITHUB_TOKEN para leer settings"

    api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
    resp = requests.get(api_url, headers=headers)
    if resp.status_code == 404:
        return {}, ""
    if resp.status_code != 200:
        return {}, f"Error al cargar settings: {resp.status_code} - {resp.text[:200]}"

    try:
        data = resp.json()
        content_b64 = data.get("content", "")
        raw = base64.b64decode(content_b64).decode("utf-8", errors="ignore")
        settings = json.loads(raw) if raw.strip() else {}
        return settings if isinstance(settings, dict) else {}, ""
    except Exception as e:
        return {}, f"Error al parsear settings: {e}"


def _save_settings_to_github(repo_url: str, settings: dict, path: str = "data/settings.json") -> tuple[bool, str]:
    """Guarda settings (JSON) en GitHub. Retorna (éxito, error_msg)."""
    owner, repo = _parse_repo_url(repo_url)
    if not owner or not repo:
        return False, "URL de repositorio no válida"

    headers, token = _get_github_headers()
    if not token:
        return False, "Se requiere GITHUB_TOKEN para guardar settings"

    api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
    # Obtener SHA si existe
    sha = None
    existing = requests.get(api_url, headers=headers)
    if existing.status_code == 200:
        try:
            sha = existing.json().get("sha")
        except Exception:
            sha = None

    raw = json.dumps(settings or {}, ensure_ascii=False, indent=2)
    content_b64 = base64.b64encode(raw.encode("utf-8")).decode("utf-8")

    payload = {
        "message": "Update app settings",
        "content": content_b64,
    }
    if sha:
        payload["sha"] = sha

    put = requests.put(api_url, headers=headers, json=payload)
    if put.status_code in (200, 201):
        return True, ""
    return False, f"Error al guardar settings: {put.status_code} - {put.text[:200]}"


@st.cache_data(ttl=72000, max_entries=1, show_spinner=False)  # Cachear por 20 horas, máximo 1 entrada
def _load_dataframe_from_github(repo_url: str, path: str = "data") -> tuple[pd.DataFrame, dict, str]:
    """
    Carga el DataFrame y el índice desde GitHub.
    Retorna (DataFrame, file_index, mensaje_error)
    Si hay error, retorna (DataFrame vacío, {}, mensaje_error)
    
    NOTA: Esta función está cacheada con @st.cache_data para evitar descargar
    el DataFrame (17MB) en cada ejecución del script.
    El caché dura 20 horas ya que los cambios en transcripciones ocurren 1 vez al día.
    
    IMPORTANTE: En Streamlit Cloud, el caché puede limpiarse si el servidor se reinicia.
    Por eso también usamos session_state como respaldo.
    """
    import time
    start_time = time.time()
    
    owner, repo = _parse_repo_url(repo_url)
    if not owner or not repo:
        return pd.DataFrame(), {}, "URL de repositorio no válida"
    
    headers, token = _get_github_headers()
    
    try:
        import pickle
        import json
        
        base_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
        
        # Cargar índice primero con timeout
        index_file_url = f"{base_url}/transcripciones_index.json"
        index_start = time.time()
        index_resp = requests.get(index_file_url, headers=headers, timeout=30)
        index_time = time.time() - index_start
        
        if index_resp.status_code != 200:
            return pd.DataFrame(), {}, f"Índice no encontrado: {index_resp.status_code}"
        
        index_content = base64.b64decode(index_resp.json()["content"]).decode('utf-8')
        index_data = json.loads(index_content)
        file_index = index_data.get('file_index', {})
        df_parts = index_data.get('df_parts', 1)
        
        # Cargar DataFrame: varias partes o uno solo (legacy)
        df_start = time.time()
        parts_content = []
        if df_parts and int(df_parts) > 1:
            for i in range(int(df_parts)):
                part_url = f"{base_url}/transcripciones_df_part_{i}.pkl"
                part_resp = requests.get(part_url, headers=headers, timeout=120)
                if part_resp.status_code != 200:
                    return pd.DataFrame(), {}, f"Parte {i+1} no encontrada: {part_resp.status_code}"
                parts_content.append(base64.b64decode(part_resp.json()["content"]))
            df_content = b"".join(parts_content)
        else:
            part0_url = f"{base_url}/transcripciones_df_part_0.pkl"
            part0_resp = requests.get(part0_url, headers=headers, timeout=120)
            if part0_resp.status_code == 200:
                df_content = base64.b64decode(part0_resp.json()["content"])
            else:
                legacy_url = f"{base_url}/transcripciones_df.pkl"
                legacy_resp = requests.get(legacy_url, headers=headers, timeout=120)
                if legacy_resp.status_code != 200:
                    return pd.DataFrame(), {}, f"DataFrame no encontrado: {legacy_resp.status_code}"
                df_content = base64.b64decode(legacy_resp.json()["content"])
        
        df_download_time = time.time() - df_start
        decode_start = time.time()
        decode_time = time.time() - decode_start
        pickle_start = time.time()
        df = pickle.loads(df_content)
        pickle_time = time.time() - pickle_start
        
        total_time = time.time() - start_time
        # Guardar tiempos en session_state para debugging (solo si no está cacheado)
        if 'df_load_times' not in st.session_state:
            st.session_state['df_load_times'] = {
                'index_time': index_time,
                'download_time': df_download_time,
                'decode_time': decode_time,
                'pickle_time': pickle_time,
                'total_time': total_time,
                'files': len(file_index) if isinstance(file_index, dict) else 0,
            }
        
        return df, file_index, ""
        
    except Exception as e:
        return pd.DataFrame(), {}, f"Error al cargar: {str(e)}"


def _detect_changes_in_github(repo_url: str, current_file_index: dict, path: str = "transcripciones") -> tuple[bool, dict]:
    """
    Compara el índice actual con los archivos en GitHub para detectar cambios en una carpeta específica.
    Retorna (hay_cambios, nuevo_file_index)
    """
    owner, repo = _parse_repo_url(repo_url)
    if not owner or not repo:
        return False, {}  # Si no se puede parsear, asumir que no hay cambios (más seguro)
    
    headers, token = _get_github_headers()
    api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
    # Agregar timeout MUY corto para no bloquear la carga (1.5 segundos por carpeta)
    resp = requests.get(api_url, headers=headers, timeout=1.5)
    
    if resp.status_code != 200:
        # Si la carpeta no existe (404), no hay cambios en esa carpeta
        if resp.status_code == 404:
            return False, {}
        return True, {}  # Si hay otro error, asumir que hay cambios
    
    items = resp.json()
    if not isinstance(items, list):
        return True, {}
    
    new_file_index = {}
    has_changes = False
    
    # Filtrar el índice cacheado para solo archivos de esta carpeta (case-insensitive)
    path_lower = path.lower()
    cached_files_in_folder = {
        filename: info for filename, info in current_file_index.items()
        if info.get('folder', '').lower() == path_lower
    }
    
    # Crear índice de archivos actuales en GitHub
    for item in items:
        if item.get("type") == "file" and item.get("name", "").lower().endswith(".txt"):
            filename = item["name"]
            sha = item.get("sha", "")
            size = item.get("size", 0)
            
            new_file_index[filename] = {
                'sha': sha,
                'folder': path.lower(),  # Normalizar a minúsculas
                'size': size
            }
            
            # Comparar con el índice anterior (solo archivos de esta carpeta)
            if filename not in cached_files_in_folder:
                has_changes = True  # Archivo nuevo
            elif cached_files_in_folder[filename].get('sha') != sha:
                has_changes = True  # Archivo modificado
    
    # Verificar si se eliminaron archivos de esta carpeta
    if len(new_file_index) != len(cached_files_in_folder):
        has_changes = True
    
    return has_changes, new_file_index


def read_txt_files_from_github(
    repo_url: str,
    path: str = "transcripciones",
    status_cb=None,
    time_window_key: str | None = None,
) -> tuple[List[dict], str]:
    """
    Lee archivos .txt desde GitHub.
    Retorna (lista_de_archivos, mensaje_de_error_o_exito)
    """
    import re as _re
    from datetime import datetime, timedelta
    
    if repo_url.count("/") == 1 and "/" in repo_url:
        owner_repo = repo_url
    else:
        m = _re.match(r"https?://github.com/([^/]+)/([^/]+)", repo_url)
        if not m:
            return [], "URL de repo no válida."
        owner_repo = f"{m.group(1)}/{m.group(2).replace('.git','')}"
    
    # Verificar caché (válido por 30 minutos para reducir peticiones)
    # Incluir time_window_key para evitar mezclar resultados filtrados/no filtrados.
    cache_key = f"github_cache_{owner_repo}_{path}_{(time_window_key or 'all')}"
    if cache_key in st.session_state:
        cached_data, cached_time = st.session_state[cache_key]
        if datetime.now() - cached_time < timedelta(minutes=30):
            return cached_data, ""
    
    headers, token = _get_github_headers()
    api_url = f"https://api.github.com/repos/{owner_repo}/contents/{path}"
    resp = requests.get(api_url, headers=headers)
    
    # Verificar si es un error de rate limit
    if resp.status_code == 403 or resp.status_code == 429:
        try:
            error_json = resp.json()
            error_message = error_json.get("message", "")
            if "rate limit" in error_message.lower() or "API rate limit" in error_message:
                # Es un error de rate limit
                reset_time = resp.headers.get("X-RateLimit-Reset", "")
                remaining = resp.headers.get("X-RateLimit-Remaining", "0")
                limit = resp.headers.get("X-RateLimit-Limit", "60")
                
                # Verificar si el token se está usando correctamente
                token_being_used = "Authorization" in headers and headers["Authorization"].startswith("token ")
                token_status_msg = ""
                if token and not token_being_used:
                    token_status_msg = "\n\n⚠️ PROBLEMA DETECTADO: Tienes un token configurado pero parece que no se está usando correctamente en las peticiones.\nVerifica que el token esté correctamente configurado en los secrets."
                elif not token:
                    token_status_msg = "\n\n💡 CONSEJO: Configura un GITHUB_TOKEN en los secrets para tener 5,000 peticiones/hora en lugar de 60."
                elif limit == "60":
                    token_status_msg = "\n\n⚠️ ADVERTENCIA: El límite es 60, lo que sugiere que el token no se está usando. Verifica que el token esté correctamente configurado."
                
                reset_info = ""
                if reset_time:
                    try:
                        reset_timestamp = int(reset_time)
                        reset_datetime = timestamp_to_spain_time(reset_timestamp)
                        reset_info = f"\n\nEl límite se restablecerá aproximadamente a las: {reset_datetime.strftime('%Y-%m-%d %H:%M:%S')} (hora española)"
                    except:
                        pass
                
                # Mensaje mejorado según el límite alcanzado
                if limit == "5000":
                    solutions = """**Soluciones:**
- ⏰ **Espera hasta la hora indicada arriba** - El límite se restablecerá automáticamente
- 💾 **Usa el caché** - Los datos se guardan en caché por 30 minutos, evita recargar innecesariamente
- 🗑️ **Limpia el caché solo cuando sea necesario** - Usa el botón 'Limpiar Caché' solo si los archivos han cambiado
- 📊 **Monitorea tus peticiones** - Con 5000 peticiones/hora puedes hacer ~83 peticiones/minuto
- ⚠️ **Evita recargas repetidas** - Cada recarga de la página puede hacer múltiples peticiones"""
                else:
                    solutions = """**Soluciones:**
- Espera unos minutos antes de volver a intentar
- Verifica que el token esté correctamente configurado en los secrets
- Usa el botón '🧪 Probar Token' para verificar que funciona
- Evita recargar la página repetidamente"""
                
                return [], f"⚠️ Límite de tasa de la API de GitHub alcanzado.\n\n**Estado:**\n- Límite: {limit} peticiones/hora\n- Quedan: {remaining} peticiones disponibles\n- Usado: {int(limit) - int(remaining)}/{limit}{reset_info}{token_status_msg}\n\n{solutions}"
        except:
            pass
    
    if resp.status_code == 404:
        return [], f"La carpeta '{path}' no existe en el repositorio."
    elif resp.status_code == 403:
        # Intentar con formato Bearer si el formato token falló
        if token and headers.get("Authorization", "").startswith("token "):
            headers["Authorization"] = f"Bearer {token}"
            resp = requests.get(api_url, headers=headers)
            if resp.status_code == 200:
                # Si funciona con Bearer, continuar
                pass
            else:
                error_detail = ""
                try:
                    error_json = resp.json()
                    error_detail = error_json.get("message", "")
                except:
                    error_detail = resp.text[:200]
                
                token_info = "Token configurado" if token else "No hay token configurado"
                return [], f"Acceso denegado (403). {token_info}. Verifica que:\n- El token tenga permisos de lectura para repositorios\n- El repositorio sea accesible con este token\n- El token esté correctamente configurado en Streamlit Cloud Secrets\n\nDetalle: {error_detail}"
        else:
            error_detail = ""
            try:
                error_json = resp.json()
                error_detail = error_json.get("message", "")
            except:
                error_detail = resp.text[:200]
            
            token_info = "Token configurado" if token else "No hay token configurado"
            return [], f"Acceso denegado (403). {token_info}. Verifica que:\n- El token tenga permisos de lectura para repositorios\n- El repositorio sea accesible con este token\n- El token esté correctamente configurado en Streamlit Cloud Secrets\n\nDetalle: {error_detail}"
    elif resp.status_code != 200:
        return [], f"Error al acceder a GitHub (código {resp.status_code}): {resp.text[:200]}"
    
    try:
        items = resp.json()
    except Exception as e:
        return [], f"Error al procesar la respuesta de GitHub: {str(e)}"
    
    # Si items no es una lista, puede ser un archivo único o un error
    if not isinstance(items, list):
        if isinstance(items, dict) and items.get("type") == "file":
            if items.get("name", "").lower().endswith(".txt"):
                # Es un archivo .txt único
                try:
                    content_bytes = base64.b64decode(items.get("content", ""))
                    content = content_bytes.decode("utf-8", errors="ignore")
                    result = [{"name": items['name'], "content": content, "folder": path.lower()}]
                    # Guardar en caché
                    from datetime import datetime
                    st.session_state[cache_key] = (result, datetime.now())
                    return result, ""
                except Exception as e:
                    return [], f"Error al decodificar el archivo: {str(e)}"
        return [], f"La ruta '{path}' no es una carpeta o no contiene archivos .txt."
    
    data = []
    txt_files_found = 0
    txt_items = [
        f for f in items
        if f.get("type") == "file" and f.get("name", "").lower().endswith(".txt")
    ]

    # Si hay filtro temporal, precomputar qué ficheros entran para que el contador
    # (y por tanto el ETA) refleje el total resultante.
    allowed_names = None
    total_files = len(txt_items)
    if time_window_key:
        try:
            meta_list = [{"name": f["name"], "folder": path} for f in txt_items]
            kept_metas, _stats = filter_files_by_time_window(meta_list, time_window_key)
            allowed_names = set(m.get("name") for m in kept_metas)
            total_files = len(allowed_names)
        except Exception:
            # Si algo falla al filtrar, caemos al comportamiento anterior (sin filtrar)
            allowed_names = None
            total_files = len(txt_items)
    
    for idx, f in enumerate(items):
        if f.get("type") == "file" and f.get("name","").lower().endswith(".txt"):
            if allowed_names is not None and f.get("name") not in allowed_names:
                # Saltar archivos que no entran en el filtro temporal
                continue
            txt_files_found += 1
            file_api = f"https://api.github.com/repos/{owner_repo}/contents/{path}/{f['name']}"
            
            # Agregar un pequeño delay entre peticiones si hay muchos archivos (para evitar rate limit)
            # Solo delay si hay más de 10 archivos y no es el primero
            if total_files > 10 and idx > 0:
                import time
                time.sleep(0.1)  # 100ms de delay entre peticiones
            
            file_resp = requests.get(file_api, headers=headers)
            
            # Verificar rate limit en peticiones de archivos individuales
            if file_resp.status_code == 403 or file_resp.status_code == 429:
                try:
                    error_json = file_resp.json()
                    error_message = error_json.get("message", "")
                    if "rate limit" in error_message.lower() or "API rate limit" in error_message:
                        limit = file_resp.headers.get("X-RateLimit-Limit", "60")
                        remaining = file_resp.headers.get("X-RateLimit-Remaining", "0")
                        token_issue = ""
                        if limit == "60" and token:
                            token_issue = "\n\n⚠️ El límite es 60, lo que indica que el token NO se está usando correctamente. Verifica la configuración del token."
                        return [], f"⚠️ Límite de tasa alcanzado al obtener archivos.\n\nLímite: {limit}/hora\nQuedan: {remaining} peticiones{token_issue}"
                except:
                    pass
            
            if file_resp.status_code == 200:
                file_info = file_resp.json()
                try:
                    content_bytes = base64.b64decode(file_info.get("content", ""))
                    content = content_bytes.decode("utf-8", errors="ignore")
                    data.append({"name": f['name'], "content": content, "folder": path.lower()})
                    if status_cb:
                        try:
                            status_cb(path, txt_files_found, total_files)
                        except Exception:
                            pass
                except Exception as e:
                    return [], f"Error al decodificar el archivo {f['name']}: {str(e)}"
            elif file_resp.status_code != 200:
                # Si hay un error al obtener un archivo, continuar con los demás pero registrar el error
                error_msg = f"Error al obtener {f['name']}: código {file_resp.status_code}"
                if txt_files_found == 1:  # Si es el primer archivo y falla, retornar error
                    return [], error_msg
                # Si hay más archivos, continuar pero mostrar advertencia
                st.warning(error_msg)
    
    if txt_files_found == 0:
        if time_window_key:
            return [], f"No se encontraron archivos .txt en la carpeta '{path}' para el rango seleccionado."
        return [], f"No se encontraron archivos .txt en la carpeta '{path}'."
    
    # Guardar en caché los resultados exitosos
    from datetime import datetime
    st.session_state[cache_key] = (data, datetime.now())
    
    return data, ""


def force_regenerate_dataframe(
    repo_url: str,
    custom_path: str = "",
    time_window: str = "season",
    progress_cb=None,
    status_cb=None,
) -> tuple[pd.DataFrame, List[dict], str, str]:
    """
    Fuerza la regeneración del DataFrame ignorando el caché.
    Carga todos los archivos desde cero, construye el DataFrame y lo guarda en GitHub.
    
    Retorna (DataFrame, files, folder_used, error_message)
    """
    # Cargar archivos desde cero (ignorando caché)
    if progress_cb:
        progress_cb(0.05)
    files, folder_used, error_msg = load_transcriptions_from_github(
        repo_url,
        custom_path,
        status_cb=status_cb,
        time_window_key=time_window,
    )
    if not files:
        return pd.DataFrame(), [], "", error_msg if error_msg else "No se encontraron archivos"

    # Doble garantía: aplicar de nuevo el filtro temporal antes de construir el DataFrame.
    # Esto evita que cualquier desviación durante la lectura acabe metiendo archivos fuera de rango.
    files, filter_stats = filter_files_by_time_window(files, time_window)
    if not files:
        return pd.DataFrame(), [], "", "No quedaron archivos dentro del rango temporal seleccionado."
    
    # Construir DataFrame
    if progress_cb:
        progress_cb(0.3)
    df = build_transcriptions_dataframe(files)
    
    # Guardar en GitHub para próximas cargas
    if progress_cb:
        progress_cb(0.6)
    file_index = _get_file_sha(files)
    save_success, save_error = _save_dataframe_to_github(repo_url, df, file_index)
    if not save_success:
        # No es crítico si falla guardar, pero mostrar advertencia
        error_msg = f"⚠️ DataFrame regenerado pero no se pudo guardar en GitHub: {save_error}"

    # Mostrar en el nombre de carpeta usada la ventana temporal aplicada
    _label = ""
    if time_window == "last_1m":
        _label = "último mes"
    elif time_window == "last_2m":
        _label = "últimos 2 meses"
    elif time_window == "last_6m":
        _label = "últimos 6 meses"
    else:
        try:
            start, end = _season_date_range_for(datetime.now())
            _label = f"temporada {start.strftime('%d/%m/%Y')} - {end.strftime('%d/%m/%Y')}"
        except Exception:
            _label = "última temporada"
    if _label:
        folder_used = f"{folder_used} | {_label}"
    if filter_stats:
        folder_used = f"{folder_used} | filtrados: {filter_stats.get('kept', 0)}/{filter_stats.get('total_in', 0)}"
    return df, files, folder_used, error_msg if not save_success else ""


def load_transcriptions_from_github_optimized(
    repo_url: str,
    custom_path: str = "",
    progress_cb=None,
    status_cb=None,
) -> tuple[pd.DataFrame, List[dict], str, str, str]:
    """
    Carga transcripciones de forma optimizada:
    1. PRIMERO: Verifica si ya está en session_state (más rápido, persiste en la sesión)
    2. SEGUNDO: Intenta cargar DataFrame pre-construido desde GitHub (usa caché de Streamlit)
    3. Detecta cambios comparando SHA de archivos en ambas carpetas (transcripciones y spoti)
    4. Si hay cambios, regenera y guarda el DataFrame
    5. Si no hay cambios, usa el DataFrame pre-construido (RÁPIDO)
    
    Retorna (DataFrame, files, folder_used, status_message, error_message)
    status_message puede ser: "cached", "regenerated", "first_load", "error", "session_cached"
    """
    # OPTIMIZACIÓN: Primero verificar session_state (más rápido, persiste en la sesión)
    cache_key = f"df_cache_{repo_url}"
    if progress_cb:
        progress_cb(0.1)
    if cache_key in st.session_state:
        df_cached = st.session_state[cache_key].get('df')
        cached_index = st.session_state[cache_key].get('index', {})
        files_light = st.session_state[cache_key].get('files', [])
        folder_used = st.session_state[cache_key].get('folder', 'transcripciones')
        if df_cached is not None and not df_cached.empty:
            if progress_cb:
                progress_cb(1.0)
            return df_cached, files_light, folder_used, "session_cached", ""
    
    # Si no está en session_state, intentar cargar desde GitHub (usará caché de Streamlit si está disponible)
    df_cached, cached_index, load_error = _load_dataframe_from_github(repo_url)
    
    if df_cached.empty or load_error:
        # No hay DataFrame guardado o error al cargar, cargar desde cero
        files, folder_used, error_msg = load_transcriptions_from_github(
            repo_url, custom_path, status_cb=status_cb
        )
        if not files:
            if progress_cb:
                progress_cb(1.0)
            return pd.DataFrame(), [], "", "error", error_msg
        
        # Construir DataFrame
        df = build_transcriptions_dataframe(files)
        
        # Guardar en GitHub para próximas cargas
        file_index = _get_file_sha(files)
        save_success, save_error = _save_dataframe_to_github(repo_url, df, file_index)
        if not save_success:
            # No es crítico si falla guardar, solo mostrar advertencia
            pass
        
        # Guardar también en session_state como caché adicional
        cache_key = f"df_cache_{repo_url}"
        st.session_state[cache_key] = {
            'df': df,
            'index': file_index,
            'files': files,
            'folder': folder_used
        }
        if progress_cb:
            progress_cb(1.0)
        return df, files, folder_used, "first_load", ""
    
    # Hay DataFrame guardado
    # OPTIMIZACIÓN: Saltar la detección de cambios por defecto para carga rápida
    # El usuario puede usar "Forzar Regeneración" si necesita actualizar
    # Esto reduce el tiempo de carga de ~3 minutos a ~10-20 segundos
    has_changes = False
    new_file_index = cached_index.copy()  # Usar el índice cacheado directamente
    
    if not has_changes:
        # No hay cambios, usar DataFrame pre-construido (RÁPIDO)
        # Necesitamos los archivos para el calendario, pero no necesitamos cargar el contenido completo
        # Solo necesitamos la lista de nombres y metadatos
        files_light = []
        for filename, file_info in cached_index.items():
            files_light.append({
                'name': filename,
                'folder': file_info.get('folder', 'transcripciones'),
                'content': '',  # No cargamos contenido si no hay cambios (se carga bajo demanda si es necesario)
                'sha': file_info.get('sha', '')
            })
        
        # Determinar carpeta usada (mostrar ambas si hay archivos de ambas)
        has_transcripciones = any(f.get('folder', '').lower() == 'transcripciones' for f in files_light)
        has_spoti = any(f.get('folder', '').lower() == 'spoti' for f in files_light)
        
        if has_transcripciones and has_spoti:
            folder_used = "transcripciones y spoti"
        elif has_transcripciones:
            folder_used = "transcripciones"
        elif has_spoti:
            folder_used = "spoti"
        else:
            folder_used = "transcripciones"
        
        # Guardar en session_state como caché adicional (útil en Streamlit Cloud)
        cache_key = f"df_cache_{repo_url}"
        st.session_state[cache_key] = {
            'df': df_cached,
            'index': cached_index,
            'files': files_light,
            'folder': folder_used
        }
        if progress_cb:
            progress_cb(1.0)
        return df_cached, files_light, folder_used, "cached", ""
    
    # Hay cambios, regenerar DataFrame
    files, folder_used, error_msg = load_transcriptions_from_github(
        repo_url, custom_path, status_cb=status_cb
    )
    if not files:
        # Si falla cargar archivos, usar DataFrame cacheado como fallback
        return df_cached, [], "transcripciones", "error", f"Error al cargar archivos actualizados: {error_msg}"
    
    # Construir nuevo DataFrame
    df = build_transcriptions_dataframe(files)
    
    # Guardar nuevo DataFrame en GitHub
    file_index = _get_file_sha(files)
    save_success, save_error = _save_dataframe_to_github(repo_url, df, file_index)
    if not save_success:
        # No es crítico si falla guardar
        pass
    
    # Guardar también en session_state como caché adicional
    cache_key = f"df_cache_{repo_url}"
    st.session_state[cache_key] = {
        'df': df,
        'index': file_index,
        'files': files,
        'folder': folder_used
    }
    if progress_cb:
        progress_cb(1.0)
    return df, files, folder_used, "regenerated", ""


def load_transcriptions_from_github(
    repo_url: str,
    custom_path: str = "",
    status_cb=None,
    time_window_key: str | None = None,
) -> tuple[List[dict], str, str]:
    """
    Intenta cargar archivos desde una ruta personalizada, 'transcripciones' o 'spoti'.
    Si no se especifica ruta personalizada, carga archivos de AMBAS carpetas (transcripciones y spoti).
    Retorna (files, folder_used, error_message)
    """
    all_files = []
    folders_found = []
    
    # Si se especifica una ruta personalizada, intentar solo con esa
    if custom_path:
        files, error_msg = read_txt_files_from_github(
            repo_url,
            path=custom_path,
            status_cb=status_cb,
            time_window_key=time_window_key,
        )
        if files:
            return files, custom_path, ""
        elif error_msg and "404" not in error_msg:
            # Si hay un error real (no solo que no existe), retornarlo
            return [], "", error_msg
        # Si no encuentra nada, continuar para intentar las carpetas por defecto
    
    # Cargar archivos de "transcripciones"
    files_trans, error_msg_trans = read_txt_files_from_github(
        repo_url,
        path="transcripciones",
        status_cb=status_cb,
        time_window_key=time_window_key,
    )
    if files_trans:
        all_files.extend(files_trans)
        folders_found.append("transcripciones")
    elif error_msg_trans and "404" not in error_msg_trans:
        # Si hay un error real, guardarlo pero continuar
        pass
    
    # Cargar archivos de "spoti" (intentar también con mayúscula por si acaso)
    for spoti_variant in ["spoti", "Spoti", "SPOTI"]:
        files_spoti, error_msg_spoti = read_txt_files_from_github(
            repo_url,
            path=spoti_variant,
            status_cb=status_cb,
            time_window_key=time_window_key,
        )
        if files_spoti:
            all_files.extend(files_spoti)
            folders_found.append(spoti_variant.lower())  # Normalizar a minúsculas
            break  # Si encuentra con una variante, no intentar las demás
        elif error_msg_spoti and "404" not in error_msg_spoti:
            # Si hay un error real, guardarlo pero continuar
            pass
    
    # Si se encontraron archivos, retornarlos
    if all_files:
        folder_used = ", ".join(folders_found) if len(folders_found) > 1 else folders_found[0] if folders_found else "transcripciones"
        return all_files, folder_used, ""
    
    # Si no encuentra nada en ninguna carpeta
    return [], "", "No se encontraron archivos .txt en las carpetas 'transcripciones' ni 'spoti'."


def parse_transcription_text(name: str, text: str, folder: str = "") -> pd.DataFrame:
    pattern = re.compile(r"\[([^\]]+)\]\s*(.*?)((?=\[)|$)", re.S)
    rows = []
    for idx, m in enumerate(pattern.finditer(text)):
        speaker = m.group(1).strip()
        content = re.sub(r"\s+", " ", m.group(2).strip().replace('\r\n','\n')).strip()
        rows.append({"file": name, "speaker": speaker, "text": content, "block_index": idx, "folder": folder.lower()})
    if not rows:
        cleaned = re.sub(r"\s+"," ", text).strip()
        rows.append({"file": name, "speaker": "UNKNOWN", "text": cleaned, "block_index": 0, "folder": folder.lower()})
    return pd.DataFrame(rows)


def build_transcriptions_dataframe(files: List[dict]) -> pd.DataFrame:
    dfs = [parse_transcription_text(f['name'], f['content'], f.get('folder', '')) for f in files]
    if dfs:
        df = pd.concat(dfs, ignore_index=True)
        df["text_norm"] = df["text"].apply(normalize_text)  # 💥 normalizamos una vez
        return df
    else:
        return pd.DataFrame(columns=["file","speaker","text","block_index","folder"])


# --- Texto y búsqueda optimizados ---
def normalize_text(text: str) -> str:
    """Normaliza texto: minúsculas, sin tildes, sin puntuación, sin saltos ni espacios extra (para búsqueda)."""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    # Quitar puntuación: sustituir por espacio para no juntar palabras
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


# Palabras frecuentes que se ignoran al exigir "todas las palabras" (menor peso en la búsqueda)
_STOP_WORDS = frozenset({
    "el", "la", "los", "las", "un", "una", "unos", "unas", "al", "a", "de", "del", "en", "por", "para",
    "con", "sin", "sobre", "bajo", "entre", "hacia", "desde", "hasta", "durante", "mediante", "según",
    "que", "qué", "cual", "cuales", "quien", "quienes", "cuando", "donde", "como", "porque", "si", "no",
    "y", "e", "o", "u", "pero", "sino", "aunque", "porque", "pues", "ya", "también", "solo", "más",
    "menos", "muy", "tan", "esto", "esta", "ese", "esa", "aquel", "aquella", "su", "sus", "mi", "tu"
})


# --- Búsqueda optimizada con fuzzy ---
def search_transcriptions(
    df: pd.DataFrame,
    query: str,
    use_regex: bool = False,
    fuzzy_mode: str = "contextual",
    threshold: int = 86,
    use_stop_words: bool = True,
) -> pd.DataFrame:
    """
    Búsqueda flexible con texto normalizado.
    Siempre busca en 'transcripciones' y en 'spoti' y muestra los resultados de ambas carpetas.
    use_stop_words: si True, ignora artículos/preposiciones al exigir "todas las palabras".
    """
    if df.empty or not query:
        return pd.DataFrame(columns=["file", "speaker", "text", "block_index", "match_preview", "folder"])

    # --- Normalizar texto y consulta una sola vez ---
    if "text_norm" not in df.columns:
        df = df.copy()
        df["text_norm"] = df["text"].apply(normalize_text)
    
    # Asegurar que existe la columna 'folder'
    if "folder" not in df.columns:
        df = df.copy()
        df["folder"] = "transcripciones"  # Por defecto
    
    query_norm = normalize_text(query)

    # --- Separar DataFrame por carpeta ---
    df_transcripciones = df[df["folder"].str.lower() == "transcripciones"].copy()
    df_spoti = df[df["folder"].str.lower() == "spoti"].copy()

    # --- Función auxiliar para buscar en un DataFrame ---
    def search_in_dataframe(df_subset: pd.DataFrame) -> pd.DataFrame:
        if df_subset.empty:
            return pd.DataFrame()
        
        results = pd.DataFrame()

        # --- Búsqueda exacta / regex sobre texto normalizado ---
        if not use_regex:
            # 1️⃣ Primero busca la frase completa
            mask = df_subset["text_norm"].str.contains(re.escape(query_norm), na=False)

            # 2️⃣ Si no hay resultados, busca por palabras (opcional: ignorar artículos y preposiciones)
            if not mask.any():
                all_terms = [t for t in query_norm.split() if t]
                if use_stop_words:
                    terms = [t for t in all_terms if t not in _STOP_WORDS]
                    if not terms:
                        terms = all_terms
                else:
                    terms = all_terms
                if terms:
                    mask = df_subset["text_norm"].apply(lambda t: all(term in t for term in terms))
            results = df_subset.loc[mask].copy()
        else:
            try:
                mask = df_subset["text_norm"].str.contains(query_norm, flags=re.IGNORECASE, regex=True, na=False)
                results = df_subset.loc[mask].copy()
            except re.error:
                return pd.DataFrame()

        # --- Búsqueda fuzzy si no hay coincidencias exactas ---
        if results.empty and fuzzy_mode != "ninguno":
            matched_rows = []
            texts = df_subset["text_norm"].tolist()
            rows = df_subset.to_dict("records")

            if fuzzy_mode == "palabra":
                all_terms = [t for t in query_norm.split() if t]
                query_terms = ([t for t in all_terms if t not in _STOP_WORDS] or all_terms) if use_stop_words else all_terms
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
        
        return results

    # --- Buscar siempre en transcripciones y en spoti ---
    results_trans = search_in_dataframe(df_transcripciones)
    results_spoti = search_in_dataframe(df_spoti) if not df_spoti.empty else pd.DataFrame()
    
    if not results_trans.empty and not results_spoti.empty:
        results = pd.concat([results_trans, results_spoti], ignore_index=True)
    elif not results_trans.empty:
        results = results_trans
    elif not results_spoti.empty:
        results = results_spoti
    else:
        results = pd.DataFrame()

    if results.empty:
        return pd.DataFrame(columns=["file", "speaker", "text", "block_index", "match_preview", "folder"])

    # --- Ranking: ordenar por número de palabras de la consulta que aparecen en el bloque ---
    # El objetivo es que, para búsquedas tipo "frase", salgan primero los bloques que
    # contienen más palabras de la consulta (después de normalización y del filtro de stop-words).
    if not use_regex:
        all_terms = [t for t in query_norm.split() if t]
        if use_stop_words:
            terms = [t for t in all_terms if t not in _STOP_WORDS]
            if not terms:
                terms = all_terms
        else:
            terms = all_terms
        query_terms_set = set(terms)

        if query_terms_set:
            # match_word_count: cuántos términos de la consulta aparecen como palabras en el bloque
            # exact_phrase_match: si el bloque contiene la frase completa (substring) como "bonus" de orden
            results = results.copy()
            if "text_norm" in results.columns:
                def _count_query_terms_in_text_norm(t: str) -> int:
                    words = set(str(t).split())
                    return sum(1 for term in query_terms_set if term in words)

                results["match_word_count"] = results["text_norm"].apply(_count_query_terms_in_text_norm)
                results["exact_phrase_match"] = results["text_norm"].str.contains(
                    re.escape(query_norm), na=False
                )
                results["exact_phrase_match"] = results["exact_phrase_match"].astype(int)
                results = results.sort_values(
                    by=["exact_phrase_match", "match_word_count", "block_index"],
                    ascending=[False, False, True],
                    kind="mergesort",
                )

    # --- Crear vista previa con resaltado (snippet largo para usar mejor el ancho disponible) ---
    _preview_len = 480  # caracteres a mostrar en vista previa
    _context_before = 60

    def make_preview(text):
        tnorm = normalize_text(text)
        idx = tnorm.find(query_norm)
        if idx != -1:
            start = max(0, idx - _context_before)
            end = min(len(text), start + _preview_len)
            snippet = text[start:end]
            preview_text = ("..." if start > 0 else "") + snippet + ("..." if end < len(text) else "")
        else:
            preview_text = text[:_preview_len] + ("..." if len(text) > _preview_len else "")
        return highlight_matching_words(preview_text, query)

    results["match_preview"] = results["text"].apply(make_preview)
    
    # Asegurar que la columna folder esté presente
    if "folder" not in results.columns:
        results["folder"] = "transcripciones"
    
    return results[["file", "speaker", "text", "block_index", "match_preview", "folder"]]
    
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


# --- Mostrar tabla de resultados con colores por orador y resaltado de coincidencias ---
def display_results_table(results_df: pd.DataFrame):
    """Muestra los resultados en tabla HTML: fondo por orador y palabras encontradas resaltadas."""
    if results_df.empty:
        return
    rows_html = []
    for i, row in results_df.iterrows():
        bg_color = get_speaker_bg_color(row["speaker"])
        text_color = "white" if bg_color.lower() not in ["#f0f0f0", "salmon", "#ff8c00"] else "black"
        file_name = str(row["file"]).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        speaker_name = str(row["speaker"]).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        preview = row["match_preview"]
        rows_html.append(
            f'<tr style="background-color: {bg_color}; color: {text_color};"><td>{file_name}</td><td><b>{speaker_name}</b></td><td>{preview}</td></tr>'
        )
    html_content = f"""
<style>
.results-table {{ width: 100%; border-collapse: collapse; margin: 10px 0; table-layout: fixed; }}
.results-table th {{ background-color: #4C98AF; color: white; padding: 8px; text-align: left; border: 1px solid #ddd; }}
.results-table td {{ padding: 8px; border: 1px solid #ddd; }}
.results-table col.archivo {{ width: 12ch; }}
.results-table col.orador {{ width: 12ch; }}
.results-table col.vista {{ width: auto; min-width: 0; }}
.results-table td:nth-child(3) {{ word-wrap: break-word; overflow-wrap: break-word; white-space: normal; }}
</style>
<table class="results-table">
<colgroup>
<col class="archivo">
<col class="orador">
<col class="vista">
</colgroup>
<thead><tr><th>Archivo</th><th>Orador</th><th>Vista Previa</th></tr></thead>
<tbody>{''.join(rows_html)}</tbody>
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


# --- Procesar fechas de archivos ---
def extract_date_from_filename(filename: str, folder: str = "") -> tuple:
    """
    Extrae la fecha del nombre de archivo.
    Formato esperado: DDMMYYYY seguido de part1, part2, etc.
    También detecta formato DDMM (4 dígitos) y añade 2025 si es de la carpeta spoti.
    Retorna (fecha_datetime, fecha_str, es_spoti) o (None, None, False) si no se puede extraer
    """
    try:
        # Remover extensión .txt si existe
        name_without_ext = str(filename).replace('.txt', '').strip()
        # Limpieza defensiva: quita caracteres invisibles y espacios al inicio
        name_without_ext = name_without_ext.lstrip("\ufeff").strip()
        es_spoti = folder.lower() == "spoti" or "spoti" in filename.lower()
        
        # Buscar patrón de fecha al inicio: DDMMYYYY (8 dígitos)
        match = re.match(r'^(\d{8})', name_without_ext)
        if match:
            date_str = match.group(1)
            # Parsear DDMMYYYY
            day = int(date_str[0:2])
            month = int(date_str[2:4])
            year = int(date_str[4:8])
            
            try:
                date_obj = datetime(year, month, day)
                return date_obj, date_str, es_spoti
            except ValueError:
                return None, None, False
        
        # Si no encuentra formato DDMMYYYY, buscar DDMM (4 dígitos) para archivos de spoti
        if es_spoti:
            match = re.match(r'^(\d{4})', name_without_ext)
            if match:
                date_str_short = match.group(1)
                # Parsear DDMM y añadir 2025
                day = int(date_str_short[0:2])
                month = int(date_str_short[2:4])
                year = 2025  # Añadir año 2025 para archivos de spoti
                date_str = f"{date_str_short}2025"
                
                try:
                    date_obj = datetime(year, month, day)
                    return date_obj, date_str, True
                except ValueError:
                    return None, None, False
    except Exception:
        pass
    
    return None, None, False


def _season_date_range_for(dt: datetime) -> tuple[datetime, datetime]:
    """
    Temporadas: del 1 de septiembre al 15 de julio (del año siguiente).
    Si dt está entre enero y 15/jul -> temporada empezó el 1/sep del año anterior.
    Si dt está entre 16/jul y 31/ago -> consideramos la última temporada acabada.
    Si dt está entre 1/sep y 31/dic -> temporada empezó el 1/sep del mismo año.
    """
    y = dt.year
    if dt.month >= 9:
        start = datetime(y, 9, 1)
        end = datetime(y + 1, 7, 15, 23, 59, 59)
        return start, end
    if dt.month < 7 or (dt.month == 7 and dt.day <= 15):
        start = datetime(y - 1, 9, 1)
        end = datetime(y, 7, 15, 23, 59, 59)
        return start, end
    # 16/jul - 31/ago -> última temporada acabada (sep del año anterior - 15/jul del año actual)
    start = datetime(y - 1, 9, 1)
    end = datetime(y, 7, 15, 23, 59, 59)
    return start, end


def filter_files_by_time_window(files: List[dict], window_key: str) -> tuple[List[dict], dict]:
    """
    Filtra archivos según ventana temporal basada en fecha en el nombre.
    - window_key: "last_1m" | "last_2m" | "last_6m" | "season"
    Retorna (files_filtrados, stats)
    """
    import time as _time
    now = datetime.now()

    if window_key == "last_1m":
        start = now - timedelta(days=30)
        end = now
        window_label = "último mes"
    elif window_key == "last_2m":
        start = now - timedelta(days=60)
        end = now
        window_label = "últimos 2 meses"
    elif window_key == "last_6m":
        start = now - timedelta(days=180)
        end = now
        window_label = "últimos 6 meses"
    else:
        start, end = _season_date_range_for(now)
        window_label = f"temporada {start.strftime('%d/%m/%Y')} - {end.strftime('%d/%m/%Y')}"

    kept: List[dict] = []
    dropped = 0
    no_date = 0
    for f in files:
        filename = f.get("name", "")
        folder = f.get("folder", "")
        date_obj, _, _ = extract_date_from_filename(filename, folder)
        if not date_obj:
            # Filtrado estricto por fecha: si no se puede extraer fecha, no entra.
            no_date += 1
            continue
        if start <= date_obj <= end:
            kept.append(f)
        else:
            dropped += 1

    stats = {
        "window_key": window_key,
        "window_label": window_label,
        "start": start,
        "end": end,
        "kept": len(kept),
        "dropped": dropped,
        "no_date_kept": no_date,
        "total_in": len(files),
        "ts": _time.time(),
    }
    return kept, stats


def get_files_with_invalid_dates(files: List[dict]) -> List[dict]:
    """
    Devuelve listado de archivos cuyo nombre no permite extraer una fecha válida.
    Cada item: {"folder": ..., "name": ...}
    """
    invalid = []
    for f in files:
        name = f.get("name", "")
        folder = f.get("folder", "")
        date_obj, _, _ = extract_date_from_filename(name, folder)
        if not date_obj:
            invalid.append({"folder": str(folder), "name": str(name)})
    return invalid


def get_files_by_date(files: List[dict]) -> dict:
    """
    Agrupa los archivos por fecha.
    Retorna un diccionario: {fecha_datetime: [lista_de_archivos]}
    Cada archivo incluye información sobre su carpeta de origen.
    """
    files_by_date = {}
    
    for file_info in files:
        filename = file_info.get('name', '')
        folder = file_info.get('folder', '')
        date_obj, date_str, es_spoti = extract_date_from_filename(filename, folder)
        
        if date_obj:
            # Añadir información sobre si es de spoti
            file_info_with_folder = file_info.copy()
            file_info_with_folder['es_spoti'] = es_spoti
            file_info_with_folder['folder'] = folder
            
            if date_obj not in files_by_date:
                files_by_date[date_obj] = []
            files_by_date[date_obj].append(file_info_with_folder)
    
    return files_by_date


def display_calendar(files_by_date: dict, show_transcripciones: bool = True, show_spoti: bool = True):
    """
    Muestra un calendario visual con las fechas y número de archivos.
    Permite filtrar por origen (transcripciones o spoti) y muestra colores diferentes.
    """
    if not files_by_date:
        st.warning("No se encontraron archivos con fechas válidas en los nombres.")
        return
    
    # Filtrar archivos según los checkboxes
    filtered_files_by_date = {}
    for date_obj, files_list in files_by_date.items():
        filtered_files = []
        for file_info in files_list:
            folder = file_info.get('folder', '').lower()
            es_spoti = file_info.get('es_spoti', False) or 'spoti' in folder
            
            if es_spoti and show_spoti:
                filtered_files.append(file_info)
            elif not es_spoti and show_transcripciones:
                filtered_files.append(file_info)
        
        if filtered_files:
            filtered_files_by_date[date_obj] = filtered_files
    
    if not filtered_files_by_date:
        st.warning("No hay archivos que coincidan con los filtros seleccionados.")
        return
    
    # Ordenar fechas
    sorted_dates = sorted(filtered_files_by_date.keys())
    min_date = sorted_dates[0]
    max_date = sorted_dates[-1]
    
    # Traducción de días de la semana
    dias_semana = {
        'Monday': 'Lunes',
        'Tuesday': 'Martes',
        'Wednesday': 'Miércoles',
        'Thursday': 'Jueves',
        'Friday': 'Viernes',
        'Saturday': 'Sábado',
        'Sunday': 'Domingo'
    }
    
    # Contar archivos por origen
    total_transcripciones = sum(len([f for f in files if not (f.get('es_spoti', False) or 'spoti' in f.get('folder', '').lower())]) 
                                for files in filtered_files_by_date.values())
    total_spoti = sum(len([f for f in files if f.get('es_spoti', False) or 'spoti' in f.get('folder', '').lower()]) 
                      for files in filtered_files_by_date.values())
    
    st.markdown(f"**📅 Rango de fechas:** {min_date.strftime('%d/%m/%Y')} - {max_date.strftime('%d/%m/%Y')}")
    st.markdown(f"**📊 Total de días con archivos:** {len(filtered_files_by_date)}")
    st.markdown(f"**📁 Total de archivos:** {total_transcripciones + total_spoti}")
    if show_transcripciones and show_spoti:
        st.markdown(f"   - 📝 Transcripciones: {total_transcripciones} archivos")
        st.markdown(f"   - 🎵 Spoti: {total_spoti} archivos")
    
    # Crear un DataFrame para mostrar el calendario con información de origen
    calendar_data = []
    for date_obj in sorted_dates:
        files_list = filtered_files_by_date[date_obj]
        
        # Separar archivos por origen
        files_transcripciones = [f for f in files_list if not (f.get('es_spoti', False) or 'spoti' in f.get('folder', '').lower())]
        files_spoti = [f for f in files_list if f.get('es_spoti', False) or 'spoti' in f.get('folder', '').lower()]
        
        # Crear lista de nombres con colores HTML (escapando caracteres especiales)
        file_names_html = []
        for f in files_list:
            name = f['name']
            # Escapar caracteres especiales HTML
            name_escaped = html.escape(name)
            es_spoti_file = f.get('es_spoti', False) or 'spoti' in f.get('folder', '').lower()
            if es_spoti_file:
                file_names_html.append(f'<span style="color: #FF6B6B; font-weight: bold;">{name_escaped}</span>')
            else:
                file_names_html.append(f'<span style="color: #4ECDC4; font-weight: bold;">{name_escaped}</span>')
        
        dia_semana_en = date_obj.strftime('%A')
        dia_semana_es = dias_semana.get(dia_semana_en, dia_semana_en)
        
        calendar_data.append({
            'Fecha': date_obj.strftime('%d/%m/%Y'),
            'Día': dia_semana_es,
            'Total': len(files_list),
            'Transcripciones': len(files_transcripciones) if show_transcripciones else 0,
            'Spoti': len(files_spoti) if show_spoti else 0,
            'Archivos': ', '.join([f['name'] for f in files_list])
        })
    
    df_calendar = pd.DataFrame(calendar_data)
    
    # Mostrar tabla con colores
    st.markdown("### 📋 Tabla de archivos por fecha")
    st.markdown("""
    <style>
    .legend {
        margin: 10px 0;
        padding: 10px;
        background-color: #f0f0f0;
        border-radius: 5px;
    }
    .calendar-table {
        width: 100%;
        border-collapse: collapse;
        margin: 10px 0;
    }
    .calendar-table th {
        background-color: #4CAF50;
        color: white;
        padding: 8px;
        text-align: left;
        border: 1px solid #ddd;
    }
    .calendar-table td {
        padding: 8px;
        border: 1px solid #ddd;
    }
    </style>
    <div class="legend">
        <strong>Leyenda:</strong><br>
        <span style="color: #4ECDC4; font-weight: bold;">📝 Azul/Turquesa = Transcripciones</span><br>
        <span style="color: #FF6B6B; font-weight: bold;">🎵 Rojo = Spoti</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Crear tabla HTML personalizada para mostrar archivos con colores
    html_table_rows = []
    
    for date_obj in sorted_dates:
        files_list = filtered_files_by_date[date_obj]
        files_transcripciones = [f for f in files_list if not (f.get('es_spoti', False) or 'spoti' in f.get('folder', '').lower())]
        files_spoti = [f for f in files_list if f.get('es_spoti', False) or 'spoti' in f.get('folder', '').lower()]
        
        dia_semana_en = date_obj.strftime('%A')
        dia_semana_es = dias_semana.get(dia_semana_en, dia_semana_en)
        
        # Crear lista de nombres con colores HTML (escapando caracteres especiales)
        file_names_html = []
        for f in files_list:
            name = f['name']
            # Escapar caracteres especiales HTML
            name_escaped = html.escape(name)
            es_spoti_file = f.get('es_spoti', False) or 'spoti' in f.get('folder', '').lower()
            if es_spoti_file:
                file_names_html.append(f'<span style="color: #FF6B6B; font-weight: bold;">{name_escaped}</span>')
            else:
                file_names_html.append(f'<span style="color: #4ECDC4; font-weight: bold;">{name_escaped}</span>')
        
        # Escapar también el día de la semana por si acaso
        dia_semana_escaped = html.escape(dia_semana_es)
        fecha_str = date_obj.strftime('%d/%m/%Y')
        
        html_table_rows.append(f'<tr><td>{fecha_str}</td><td>{dia_semana_escaped}</td><td><strong>{len(files_list)}</strong></td><td style="color: #4ECDC4; font-weight: bold;">{len(files_transcripciones) if show_transcripciones else 0}</td><td style="color: #FF6B6B; font-weight: bold;">{len(files_spoti) if show_spoti else 0}</td><td>{", ".join(file_names_html)}</td></tr>')
    
    # Construir el HTML completo de forma más limpia
    html_table_content = ''.join(html_table_rows)
    
    # Construir el HTML completo con estilos CSS incluidos
    html_table = f"""
    <style>
    .calendar-table {{
        width: 100%;
        border-collapse: collapse;
        margin: 10px 0;
        font-family: Arial, sans-serif;
    }}
    .calendar-table th {{
        background-color: #4CAF50;
        color: white;
        padding: 8px;
        text-align: left;
        border: 1px solid #ddd;
    }}
    .calendar-table td {{
        padding: 8px;
        border: 1px solid #ddd;
        color: #333;
        background-color: #ffffff;
    }}
    .calendar-table tbody tr:nth-child(even) {{
        background-color: #f9f9f9;
    }}
    .calendar-table tbody tr:nth-child(even) td {{
        background-color: #f9f9f9;
        color: #333;
    }}
    .calendar-table tbody tr:nth-child(odd) td {{
        background-color: #ffffff;
        color: #333;
    }}
    </style>
    <div style="overflow-x: auto;">
    <table class="calendar-table">
    <thead>
    <tr>
    <th>Fecha</th><th>Día</th><th>Total</th><th>Transcripciones</th><th>Spoti</th><th>Archivos</th>
    </tr>
    </thead>
    <tbody>
    {html_table_content}
    </tbody>
    </table>
    </div>
    """
    
    # Usar st.components.v1.html() para renderizar HTML de forma más confiable
    try:
        import streamlit.components.v1 as components
        components.html(html_table, height=600, scrolling=True)
    except ImportError:
        # Fallback a st.markdown si components no está disponible
        st.markdown(html_table, unsafe_allow_html=True)
    
    # Mostrar gráfico de barras con colores diferentes
    st.markdown("### 📊 Gráfico de archivos por fecha")
    if show_transcripciones and show_spoti:
        # Gráfico apilado mostrando ambos orígenes
        chart_data = pd.DataFrame({
            'Fecha': [d.strftime('%d/%m/%Y') for d in sorted_dates],
            'Transcripciones': [len([f for f in filtered_files_by_date[d] if not (f.get('es_spoti', False) or 'spoti' in f.get('folder', '').lower())]) for d in sorted_dates],
            'Spoti': [len([f for f in filtered_files_by_date[d] if f.get('es_spoti', False) or 'spoti' in f.get('folder', '').lower()]) for d in sorted_dates]
        })
        st.bar_chart(chart_data.set_index('Fecha'), color=['#4ECDC4', '#FF6B6B'])
    else:
        # Gráfico simple si solo se muestra un origen
        chart_data = pd.DataFrame({
            'Fecha': [d.strftime('%d/%m/%Y') for d in sorted_dates],
            'Número de archivos': [len(filtered_files_by_date[d]) for d in sorted_dates]
        })
        color = '#FF6B6B' if show_spoti else '#4ECDC4'
        st.bar_chart(chart_data.set_index('Fecha'))
    
    # Mostrar detalles por fecha
    st.markdown("### 📅 Detalles por fecha")
    selected_date = st.selectbox(
        "Selecciona una fecha para ver detalles:",
        options=[d.strftime('%d/%m/%Y') for d in sorted_dates]
    )
    
    if selected_date:
        # Encontrar la fecha correspondiente
        selected_date_obj = None
        for date_obj in sorted_dates:
            if date_obj.strftime('%d/%m/%Y') == selected_date:
                selected_date_obj = date_obj
                break
        
        if selected_date_obj:
            files_list = filtered_files_by_date[selected_date_obj]
            
            # Separar por origen para mostrarlos agrupados
            files_transcripciones = [f for f in files_list if not (f.get('es_spoti', False) or 'spoti' in f.get('folder', '').lower())]
            files_spoti = [f for f in files_list if f.get('es_spoti', False) or 'spoti' in f.get('folder', '').lower()]
            
            st.markdown(f"**Archivos del {selected_date}:**")
            
            if files_transcripciones:
                st.markdown(f"<h4 style='color: #4ECDC4;'>📝 Transcripciones ({len(files_transcripciones)} archivos)</h4>", unsafe_allow_html=True)
                for file_info in files_transcripciones:
                    with st.expander(f"📄 {file_info['name']}", expanded=False):
                        content = file_info.get('content', '')
                        # Si no hay contenido (cargado desde caché), intentar cargarlo bajo demanda
                        if not content and 'gh_url' in st.session_state:
                            # Cargar contenido desde GitHub
                            folder = file_info.get('folder', 'transcripciones')
                            owner, repo = _parse_repo_url(st.session_state['gh_url'])
                            if owner and repo:
                                headers, _ = _get_github_headers()
                                file_api = f"https://api.github.com/repos/{owner}/{repo}/contents/{folder}/{file_info['name']}"
                                file_resp = requests.get(file_api, headers=headers)
                                if file_resp.status_code == 200:
                                    file_data = file_resp.json()
                                    content_bytes = base64.b64decode(file_data.get("content", ""))
                                    content = content_bytes.decode("utf-8", errors="ignore")
                                    file_info['content'] = content  # Guardar en memoria
                        
                        preview = content[:500] + "..." if len(content) > 500 else content
                        if preview:
                            st.text(preview)
                        else:
                            st.info("Contenido no disponible")
            
            if files_spoti:
                st.markdown(f"<h4 style='color: #FF6B6B;'>🎵 Spoti ({len(files_spoti)} archivos)</h4>", unsafe_allow_html=True)
                for file_info in files_spoti:
                    with st.expander(f"📄 {file_info['name']}", expanded=False):
                        content = file_info.get('content', '')
                        # Si no hay contenido (cargado desde caché), intentar cargarlo bajo demanda
                        if not content and 'gh_url' in st.session_state:
                            # Cargar contenido desde GitHub
                            folder = file_info.get('folder', 'spoti')
                            owner, repo = _parse_repo_url(st.session_state['gh_url'])
                            if owner and repo:
                                headers, _ = _get_github_headers()
                                file_api = f"https://api.github.com/repos/{owner}/{repo}/contents/{folder}/{file_info['name']}"
                                file_resp = requests.get(file_api, headers=headers)
                                if file_resp.status_code == 200:
                                    file_data = file_resp.json()
                                    content_bytes = base64.b64decode(file_data.get("content", ""))
                                    content = content_bytes.decode("utf-8", errors="ignore")
                                    file_info['content'] = content  # Guardar en memoria
                        
                        preview = content[:500] + "..." if len(content) > 500 else content
                        if preview:
                            st.text(preview)
                        else:
                            st.info("Contenido no disponible")


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
        with st.spinner("Cortando audio... (puede tardar 1–2 min)"):
            try:
                segments = split_audio(audio_bytes, uploaded.name, segment_seconds=int(segment_minutes*60))
                st.session_state['audio_segments'] = segments
                st.success(f"Generados {len(segments)} fragmentos")
            except ImportError as e:
                st.error(f"❌ Error de importación: {str(e)}")
            except Exception as e:
                st.error(f"❌ Error al procesar el audio: {type(e).__name__}: {str(e)}")
                st.caption("Si el archivo es largo o muy pesado, prueba con uno más corto o hazlo en local con la app.")
    if 'audio_segments' in st.session_state:
        st.markdown("### Descargar fragmentos")
        for seg in st.session_state['audio_segments']:
            st.download_button(f"Descargar {seg['name']}", data=seg['bytes'], file_name=seg['name'])

with col2:
    st.markdown("""
    **Importante**:
    - Soporta archivos de hasta ~100 MB (p. ej. 96 MB).
    - [Cortar audio online](https://mp3cut.net/es)
    - [Transcripción automática](https://turboscribe.ai/)
    - [ffmpeg](https://www.gyan.dev/ffmpeg/builds)
    """, unsafe_allow_html=True)

st.markdown("---")


# --- UI: Transcriptions loader ---
st.header("2) Leer transcripciones")

# Mostrar estado del token
headers_token, token_value = _get_github_headers()
token_status = "✅ Configurado" if token_value else "❌ No configurado"
with st.expander(f"🔑 Estado del Token GitHub: {token_status}", expanded=False):
    if token_value:
        st.success("Token GitHub detectado correctamente")
        # Mostrar información del token (solo primeros y últimos caracteres por seguridad)
        token_preview = f"{token_value[:7]}...{token_value[-4:]}" if len(token_value) > 11 else "***"
        st.code(f"Token: {token_preview} (longitud: {len(token_value)} caracteres)")
        
        # Botón para probar el token
        if st.button("🧪 Probar Token", key="test_token"):
            with st.spinner("Probando token con la API de GitHub..."):
                try:
                    test_resp = requests.get("https://api.github.com/user", headers=headers_token)
                    if test_resp.status_code == 200:
                        user_info = test_resp.json()
                        st.success(f"✅ Token válido! Conectado como: {user_info.get('login', 'Usuario')}")
                        st.json({
                            "Usuario": user_info.get('login', 'N/A'),
                            "Nombre": user_info.get('name', 'N/A'),
                            "Email": user_info.get('email', 'N/A'),
                            "Tipo": user_info.get('type', 'N/A')
                        })
                        
                        # Mostrar información de rate limits
                        limit = test_resp.headers.get("X-RateLimit-Limit", "N/A")
                        remaining = test_resp.headers.get("X-RateLimit-Remaining", "N/A")
                        reset_time = test_resp.headers.get("X-RateLimit-Reset", "")
                        
                        st.markdown("**📊 Estado de Rate Limits:**")
                        if limit != "N/A":
                            if limit == "60":
                                st.warning(f"⚠️ Límite: {limit}/hora - El token NO se está usando correctamente")
                                st.info("💡 Verifica que el token esté correctamente configurado en los secrets")
                            elif limit == "5000":
                                st.success(f"✅ Límite: {limit}/hora - Token funcionando correctamente")
                            else:
                                st.info(f"ℹ️ Límite: {limit}/hora")
                            
                            if remaining != "N/A":
                                st.info(f"Peticiones restantes: {remaining}/{limit}")
                            
                            if reset_time:
                                try:
                                    from datetime import datetime
                                    reset_timestamp = int(reset_time)
                                    reset_datetime = datetime.fromtimestamp(reset_timestamp)
                                    st.caption(f"Se restablece: {reset_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
                                except:
                                    pass
                    elif test_resp.status_code == 401:
                        st.error("❌ Token inválido o expirado. Verifica que el token sea correcto.")
                    else:
                        st.warning(f"⚠️ Respuesta inesperada: {test_resp.status_code}")
                        st.text(test_resp.text[:200])
                except Exception as e:
                    st.error(f"❌ Error al probar el token: {str(e)}")
        
        st.info("💡 Si tienes problemas de acceso, verifica que el token tenga permisos de **repo** para repositorios privados.")
        
        # Información sobre rate limits
        st.markdown("---")
        st.markdown("**📊 Límites de la API de GitHub:**")
        st.markdown("- Sin token: 60 peticiones/hora")
        st.markdown("- Con token: 5,000 peticiones/hora (~83 peticiones/minuto)")
        st.markdown("- La aplicación usa caché (30 minutos) para reducir peticiones")
        st.markdown("- Delay automático entre peticiones cuando hay muchos archivos")
        st.warning("⚠️ **Importante:** Con 5000 peticiones/hora, evita recargar la página repetidamente. El caché ayuda pero cada recarga puede hacer múltiples peticiones.")
        
        # Botón para limpiar caché
        if st.button("🗑️ Limpiar Caché", key="clear_cache"):
            # Limpiar todos los cachés de GitHub en session_state
            keys_to_remove = [k for k in st.session_state.keys() if k.startswith("github_cache_")]
            for key in keys_to_remove:
                del st.session_state[key]
            # Limpiar también el caché del DataFrame de Streamlit
            try:
                _load_dataframe_from_github.clear()
            except:
                pass
            st.success("✅ Caché limpiado. Las próximas peticiones serán frescas.")
    else:
        st.warning("No se encontró GITHUB_TOKEN en los secrets ni en variables de entorno.")
        st.markdown("""
        **Para configurar el token en Streamlit Cloud:**
        1. Ve a tu aplicación en Streamlit Cloud
        2. Haz clic en "Settings" → "Secrets"
        3. Agrega:
           ```toml
           [default]
           GITHUB_TOKEN = "tu_token_aqui"
           ```
           O sin [default]:
           ```toml
           GITHUB_TOKEN = "tu_token_aqui"
           ```
        4. Guarda los cambios
        
        **Para crear un token en GitHub:**
        1. Ve a GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
        2. Genera un nuevo token con permisos de **repo** (para repositorios privados)
        3. Copia el token y pégalo en los secrets de Streamlit Cloud
        """)

repo_col, path_col = st.columns([2, 1])
with repo_col:
    gh_url = st.text_input("Repo público GitHub", value="https://github.com/jarconett/c_especiales/")
with path_col:
    custom_path = st.text_input("Ruta personalizada (opcional)", placeholder="ej: transcripciones, spoti, docs", help="Deja vacío para buscar en 'transcripciones' y 'spoti' automáticamente")

# Guardar URL del repositorio en session_state para uso posterior
if gh_url:
    st.session_state['gh_url'] = gh_url
    # Cargar preferencia persistente del filtro temporal (una vez por sesión)
    if "df_time_window_radio" not in st.session_state:
        settings, _settings_err = _load_settings_from_github(gh_url)
        saved_key = ""
        try:
            saved_key = str((settings or {}).get("df_time_window", "")).strip()
        except Exception:
            saved_key = ""
        _key_to_label = {
            "last_1m": "Último mes",
            "last_2m": "Últimos 2 meses",
            "last_6m": "Últimos 6 meses",
            "season": "Última temporada",
        }
        st.session_state["df_time_window_radio"] = _key_to_label.get(saved_key, "Última temporada")

# Carga automática al inicio si no hay datos (optimizada)
# PRIORIDAD: 1) session_state (trans_df/trans_files), 2) caché adicional (df_cache), 3) caché Streamlit, 4) GitHub
if gh_url:
    # Nivel 1: Si ya tenemos los datos en session_state estándar, no hacer nada (más rápido)
    if 'trans_df' in st.session_state and 'trans_files' in st.session_state:
        df_existing = st.session_state.get('trans_df')
        files_existing = st.session_state.get('trans_files')
        if not df_existing.empty and files_existing:
            # Los datos ya están cargados, no hacer nada
            pass
    # Nivel 2: Verificar caché adicional en session_state antes de descargar
    elif f"df_cache_{gh_url}" in st.session_state:
        cache_data = st.session_state[f"df_cache_{gh_url}"]
        df_cached = cache_data.get('df')
        files_cached = cache_data.get('files', [])
        folder_cached = cache_data.get('folder', 'transcripciones')
        if df_cached is not None and not df_cached.empty:
            # Restaurar desde caché adicional (muy rápido, sin descargar)
            st.session_state['trans_df'] = df_cached
            st.session_state['trans_files'] = files_cached
            st.session_state['dataframe_loaded'] = True
            st.success(f"⚡⚡ Carga instantánea desde caché de sesión: {len(df_cached)} bloques desde carpeta '{folder_cached}' ({len(files_cached)} archivos)")
    # Nivel 3 y 4: Cargar desde caché de Streamlit o GitHub
    elif 'trans_files' not in st.session_state or 'trans_df' not in st.session_state:
        # Verificar si ya se está cargando para evitar ejecuciones múltiples
        if 'loading_dataframe' not in st.session_state:
            st.session_state['loading_dataframe'] = True
            import time
            start_time = time.time()
            with st.spinner("Cargando transcripciones desde GitHub (optimizado)..."):
                progress_bar = st.progress(0)
                status_placeholder = st.empty()

                def _df_load_progress(p: float):
                    try:
                        progress_bar.progress(int(max(0, min(1, p)) * 100))
                    except Exception:
                        pass

                def _df_load_status(folder: str, current: int, total: int):
                    try:
                        # ETA basado en tiempo actual y, si existe, tiempos históricos
                        elapsed = time.time() - start_time
                        eta_str = ""
                        if current > 0 and total:
                            avg_cur = elapsed / current
                            base_avg = None
                            prev = st.session_state.get("df_load_times")
                            if prev and prev.get("files"):
                                try:
                                    base_avg = prev.get("total_time", 0) / max(1, prev["files"])
                                except Exception:
                                    base_avg = None
                            avg = (0.5 * base_avg + 0.5 * avg_cur) if base_avg else avg_cur
                            remaining = max(0.0, avg * (total - current))
                            if remaining >= 60:
                                m = int(remaining // 60)
                                s = int(remaining % 60)
                                eta_str = f" | ETA ~ {m}m {s}s"
                            else:
                                eta_str = f" | ETA ~ {int(remaining)}s"
                        status_placeholder.text(
                            f"Incorporando archivo {current} de {total} en carpeta '{folder}' de GitHub...{eta_str}"
                        )
                    except Exception:
                        pass

                df, files, folder_used, status, error_msg = load_transcriptions_from_github_optimized(
                    gh_url,
                    custom_path.strip() if custom_path else "",
                    progress_cb=_df_load_progress,
                    status_cb=_df_load_status,
                )
                progress_bar.empty()
                elapsed_time = time.time() - start_time
                if not df.empty:
                    st.session_state['trans_files'] = files
                    st.session_state['trans_df'] = df
                    st.session_state['dataframe_loaded'] = True
                    
                    # Mostrar información de tiempos si está disponible
                    time_info = ""
                    if 'df_load_times' in st.session_state:
                        times = st.session_state['df_load_times']
                        time_info = f" | Descarga: {times.get('download_time', 0):.1f}s, Despickle: {times.get('pickle_time', 0):.1f}s"
                    
                    if status == "session_cached":
                        st.success(f"⚡⚡ Carga instantánea desde session_state: {len(df)} bloques desde carpeta '{folder_used}' ({len(files)} archivos) - Tiempo: {elapsed_time:.2f}s")
                    elif status == "cached":
                        # Si viene del caché de Streamlit, debería ser muy rápido (< 5s)
                        cache_source = "caché de Streamlit" if elapsed_time < 5 else "caché (pero tardó descargando desde GitHub)"
                        st.success(f"⚡ Carga desde {cache_source}: {len(df)} bloques desde carpeta '{folder_used}' ({len(files)} archivos) - Tiempo total: {elapsed_time:.1f}s{time_info}")
                    elif status == "regenerated":
                        st.success(f"🔄 DataFrame regenerado: {len(df)} bloques desde carpeta '{folder_used}' ({len(files)} archivos) - Tiempo: {elapsed_time:.1f}s{time_info}")
                    elif status == "first_load":
                        st.success(f"📥 Primera carga desde GitHub: {len(df)} bloques desde carpeta '{folder_used}' ({len(files)} archivos) - Tiempo: {elapsed_time:.1f}s{time_info}")
                    else:
                        st.success(f"Cargados {len(files)} archivos desde carpeta '{folder_used}' y DataFrame con {len(df)} bloques - Tiempo: {elapsed_time:.1f}s{time_info}")
                else:
                    if error_msg:
                        st.error(f"❌ {error_msg}")
                    else:
                        st.warning("No se encontraron archivos .txt en las carpetas 'transcripciones' ni 'spoti'")
            # Limpiar la bandera después de cargar
            if 'loading_dataframe' in st.session_state:
                del st.session_state['loading_dataframe']

# Botones para recargar
button_col1, button_col2 = st.columns([2, 1])

with button_col1:
    if st.button("🔄 Recargar archivos .txt desde GitHub", key="reload_transcriptions"):
        if gh_url:
            st.session_state['gh_url'] = gh_url  # Guardar URL
            # Limpiar caché de Streamlit para forzar recarga desde GitHub
            _load_dataframe_from_github.clear()
            import time
            start_time = time.time()
            with st.spinner("Recargando transcripciones desde GitHub (optimizado)..."):
                progress_bar = st.progress(0)
                status_placeholder = st.empty()

                def _df_reload_progress(p: float):
                    try:
                        progress_bar.progress(int(max(0, min(1, p)) * 100))
                    except Exception:
                        pass

                def _df_reload_status(folder: str, current: int, total: int):
                    try:
                        elapsed = time.time() - start_time
                        eta_str = ""
                        if current > 0 and total:
                            avg_cur = elapsed / current
                            base_avg = None
                            prev = st.session_state.get("df_load_times")
                            if prev and prev.get("files"):
                                try:
                                    base_avg = prev.get("total_time", 0) / max(1, prev["files"])
                                except Exception:
                                    base_avg = None
                            avg = (0.5 * base_avg + 0.5 * avg_cur) if base_avg else avg_cur
                            remaining = max(0.0, avg * (total - current))
                            if remaining >= 60:
                                m = int(remaining // 60)
                                s = int(remaining % 60)
                                eta_str = f" | ETA ~ {m}m {s}s"
                            else:
                                eta_str = f" | ETA ~ {int(remaining)}s"
                        status_placeholder.text(
                            f"Incorporando archivo {current} de {total} en carpeta '{folder}' de GitHub...{eta_str}"
                        )
                    except Exception:
                        pass

                df, files, folder_used, status, error_msg = load_transcriptions_from_github_optimized(
                    gh_url,
                    custom_path.strip() if custom_path else "",
                    progress_cb=_df_reload_progress,
                    status_cb=_df_reload_status,
                )
                progress_bar.empty()
                if not df.empty:
                    st.session_state['trans_files'] = files
                    st.session_state['trans_df'] = df
                    if status == "cached":
                        st.success(f"⚡ Carga rápida desde caché: {len(df)} bloques desde carpeta '{folder_used}' ({len(files)} archivos)")
                    elif status == "regenerated":
                        st.success(f"🔄 DataFrame regenerado: {len(df)} bloques desde carpeta '{folder_used}' ({len(files)} archivos)")
                    elif status == "first_load":
                        st.success(f"📥 Primera carga: {len(df)} bloques desde carpeta '{folder_used}' ({len(files)} archivos)")
                    else:
                        st.success(f"Recargados {len(files)} archivos desde carpeta '{folder_used}' y DataFrame con {len(df)} bloques")
                else:
                    if error_msg:
                        st.error(f"❌ {error_msg}")
                    else:
                        st.warning("No se encontraron archivos .txt en las carpetas 'transcripciones' ni 'spoti'")
        else:
            st.error("Por favor, ingresa una URL de repositorio GitHub válida")

with button_col2:
    time_window_label = st.radio(
        "Incluir archivos en el DataFrame (regeneración):",
        options=["Último mes", "Últimos 2 meses", "Últimos 6 meses", "Última temporada"],
        index=3,
        key="df_time_window_radio",
        help="Filtra qué archivos de 'transcripciones' y 'spoti' se incluyen al regenerar el DataFrame.",
    )
    _time_window_map = {
        "Último mes": "last_1m",
        "Últimos 2 meses": "last_2m",
        "Últimos 6 meses": "last_6m",
        "Última temporada": "season",
    }
    time_window_key = _time_window_map.get(time_window_label, "season")

    if st.button("🔧 Forzar Regeneración", key="force_regenerate", help="Fuerza la regeneración completa del DataFrame ignorando el caché. Útil si la detección automática de cambios falla."):
        if DF_REGEN_LOCK:
            st.info("🔒 Ya se está regenerando el DataFrame en otra sesión. Espera a que termine antes de lanzar otra regeneración.")
        elif gh_url:
            DF_REGEN_LOCK = True
            try:
                st.session_state['gh_url'] = gh_url  # Guardar URL
                # Persistir preferencia del rango temporal en GitHub (entre sesiones)
                try:
                    _save_settings_to_github(gh_url, {"df_time_window": time_window_key})
                except Exception:
                    pass
                # Limpiar caché de Streamlit antes de regenerar
                _load_dataframe_from_github.clear()
                import time
                start_time = time.time()
                with st.spinner("🔄 Forzando regeneración completa del DataFrame (esto puede tardar varios minutos con 300+ archivos)..."):
                    progress_bar = st.progress(0)
                    status_placeholder = st.empty()

                    def _df_regen_progress(p: float):
                        try:
                            progress_bar.progress(int(max(0, min(1, p)) * 100))
                        except Exception:
                            pass

                    def _df_regen_status(folder: str, current: int, total: int):
                        try:
                            elapsed = time.time() - start_time
                            eta_str = ""
                            if current > 0 and total:
                                avg_cur = elapsed / current
                                base_avg = None
                                prev = st.session_state.get("df_load_times")
                                if prev and prev.get("files"):
                                    try:
                                        base_avg = prev.get("total_time", 0) / max(1, prev["files"])
                                    except Exception:
                                        base_avg = None
                                avg = (0.5 * base_avg + 0.5 * avg_cur) if base_avg else avg_cur
                                remaining = max(0.0, avg * (total - current))
                                if remaining >= 60:
                                    m = int(remaining // 60)
                                    s = int(remaining % 60)
                                    eta_str = f" | ETA ~ {m}m {s}s"
                                else:
                                    eta_str = f" | ETA ~ {int(remaining)}s"
                            status_placeholder.text(
                                f"Incorporando archivo {current} de {total} en carpeta '{folder}' de GitHub...{eta_str}"
                            )
                        except Exception:
                            pass

                    df, files, folder_used, error_msg = force_regenerate_dataframe(
                        gh_url,
                        custom_path.strip() if custom_path else "",
                        time_window=time_window_key,
                        progress_cb=_df_regen_progress,
                        status_cb=_df_regen_status,
                    )
                    progress_bar.empty()
                    if not df.empty:
                        st.session_state['trans_files'] = files
                        st.session_state['trans_df'] = df
                        # Limpiar caché después de regenerar para que use el nuevo DataFrame
                        _load_dataframe_from_github.clear()
                        if error_msg:
                            # Regeneración correcta pero fallo al guardar en GitHub
                            st.warning(f"{error_msg}")
                            st.info(f"El DataFrame se ha regenerado correctamente en esta sesión: {len(df)} bloques desde carpeta '{folder_used}' ({len(files)} archivos), pero no se pudo guardar en GitHub.")
                        else:
                            st.success(f"✅ DataFrame regenerado manualmente: {len(df)} bloques desde carpeta '{folder_used}' ({len(files)} archivos)")
                    else:
                        if error_msg:
                            st.error(f"❌ {error_msg}")
                        else:
                            st.warning("No se encontraron archivos .txt en las carpetas 'transcripciones' ni 'spoti'")
            finally:
                DF_REGEN_LOCK = False
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
        use_stop_words = st.checkbox(
            "Ignorar palabras frecuentes (art., prep.)",
            value=True,
            help="Si está activado, no se exige coincidir artículos ni preposiciones (el, la, de, en…)."
        )
        speaker_filter = st.selectbox(
            "Filtrar por orador",
            options=["(todos)"] + sorted(df['speaker'].unique().tolist())
        )
        fuzzy_mode = st.radio("Modo fuzzy", options=["ninguno", "palabra", "contextual"], index=2)
        threshold = st.slider("Umbral similitud (%)", 60, 95, 86)

    if st.button("Buscar"):
        res = search_transcriptions(df, query, use_regex, fuzzy_mode, threshold, use_stop_words=use_stop_words)
        if speaker_filter != "(todos)":
            res = res[res['speaker'] == speaker_filter]

        if res.empty:
            st.warning("No se encontraron coincidencias en transcripciones ni en spoti.")
        else:
            # Contar resultados por carpeta
            if 'folder' in res.columns:
                res_trans = res[res['folder'].str.lower() == 'transcripciones'] if res['folder'].dtype == 'object' else pd.DataFrame()
                res_spoti = res[res['folder'].str.lower() == 'spoti'] if res['folder'].dtype == 'object' else pd.DataFrame()
            else:
                res_trans = pd.DataFrame()
                res_spoti = pd.DataFrame()
            
            if not res_trans.empty and not res_spoti.empty:
                st.success(f"Encontradas {len(res)} coincidencias: {len(res_trans)} en transcripciones, {len(res_spoti)} en spoti")
            elif not res_trans.empty:
                st.success(f"Encontradas {len(res)} coincidencias en transcripciones")
            elif not res_spoti.empty:
                st.success(f"Encontradas {len(res)} coincidencias en spoti (no se encontraron en transcripciones)")
            else:
                st.success(f"Encontradas {len(res)} coincidencias")
            
            display_results_table(res[['file', 'speaker', 'match_preview']])
            for i, row in res.iterrows():
                folder_info = ""
                if 'folder' in row and pd.notna(row.get('folder')):
                    folder_name = str(row['folder']).upper()
                    folder_info = f" [{folder_name}]"
                with st.expander(f"{i+1}. {row['speaker']} — {row['file']}{folder_info} (bloque {row['block_index']})", expanded=False):
                    show_context(df, row['file'], row['block_index'], query=query, context=4)

st.markdown("---")

# --- UI: Validación de nombres de archivos ---
st.header("4) Validar nombres de archivo")
if 'trans_files' in st.session_state and st.session_state['trans_files']:
    if st.button("🧪 Validar fechas en nombres", key="validate_file_dates"):
        files_loaded = st.session_state.get('trans_files', [])
        invalid_files = get_files_with_invalid_dates(files_loaded)
        if not invalid_files:
            st.success("✅ Todos los archivos cargados tienen fecha válida al inicio del nombre (DDMMYYYY).")
        else:
            st.warning(f"⚠️ Se detectaron {len(invalid_files)} archivos con nombre inválido para fecha.")
            invalid_df = pd.DataFrame(invalid_files)
            invalid_df = invalid_df.rename(columns={"folder": "Carpeta", "name": "Archivo"})
            st.dataframe(invalid_df, use_container_width=True)
else:
    st.info("ℹ️ Primero carga/regenera el DataFrame para validar nombres de archivo.")

st.markdown("---")

# --- UI: Calendario de archivos ---
st.header("5) 📅 Calendario de transcripciones")
if 'trans_files' in st.session_state and st.session_state['trans_files']:
    files = st.session_state['trans_files']
    files_by_date = get_files_by_date(files)
    
    if files_by_date:
        st.info(f"📊 Se encontraron archivos en {len(files_by_date)} fechas diferentes")
        
        # Filtros por origen
        st.markdown("### 🔍 Filtros")
        filter_col1, filter_col2 = st.columns(2)
        with filter_col1:
            show_transcripciones = st.checkbox(
                "📝 Mostrar archivos de Transcripciones", 
                value=True, 
                key="filter_transcripciones",
                help="Archivos de la carpeta 'transcripciones' de GitHub"
            )
        with filter_col2:
            show_spoti = st.checkbox(
                "🎵 Mostrar archivos de Spoti", 
                value=True, 
                key="filter_spoti",
                help="Archivos de la carpeta 'spoti' de GitHub"
            )
        
        if not show_transcripciones and not show_spoti:
            st.warning("⚠️ Debes seleccionar al menos un origen para mostrar.")
        else:
            display_calendar(files_by_date, show_transcripciones, show_spoti)
    else:
        st.warning("⚠️ No se pudieron extraer fechas de los nombres de archivos. Verifica que sigan el formato DDMMYYYY (ej: 30012025 part1.txt)")
else:
    st.info("ℹ️ Primero carga los archivos de transcripciones desde GitHub en la sección 2)")