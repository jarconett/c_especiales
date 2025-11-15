# Sistema de Automatización de Instagram

Sistema automatizado para responder a preguntas en stories de Instagram en múltiples dispositivos Android de forma simultánea.

## 📋 Descripción

Este sistema permite:
- **Responder automáticamente** a preguntas en stories de Instagram
- **Ejecutar en paralelo** en múltiples dispositivos Android
- **Marcar stories como vistos** antes de responder
- **Mantener sesiones persistentes** para múltiples respuestas

## 🔧 Requisitos Previos

### Software Necesario:
1. **Node.js** (versión 14 o superior)
2. **Appium** (instalado globalmente)
3. **Android SDK Platform Tools (ADB)**
4. **PowerShell** (incluido en Windows 10/11)

### Configuración de Dispositivos:
- Dispositivos Android conectados vía USB
- **Depuración USB activada** en cada dispositivo
- **Permisos de depuración USB** otorgados
- Instagram instalado y configurado en cada dispositivo

## 📦 Instalación Completa desde Cero

### Paso 1: Instalar Node.js

1. **Descargar Node.js:**
   - Visita: https://nodejs.org/
   - Descarga la versión **LTS** (Long Term Support)
   - Ejecuta el instalador y sigue las instrucciones
   - Asegúrate de marcar la opción "Add to PATH" durante la instalación

2. **Verificar instalación:**
   Abre PowerShell o CMD y ejecuta:
   ```bash
   node --version
   npm --version
   ```
   Debe mostrar las versiones instaladas (ej: `v18.17.0` y `9.6.7`)

### Paso 2: Instalar Android SDK Platform Tools (ADB)

1. **Descargar Platform Tools:**
   - Visita: https://developer.android.com/tools/releases/platform-tools
   - Descarga "SDK Platform-Tools for Windows"
   - O descarga directa: https://dl.google.com/android/repository/platform-tools-latest-windows.zip

2. **Extraer y configurar:**
   - Extrae el archivo ZIP en una carpeta (ej: `C:\android-sdk\platform-tools`)
   - **Agregar al PATH del sistema:**
     - Presiona `Win + X` y selecciona "Sistema"
     - Haz clic en "Configuración avanzada del sistema"
     - Haz clic en "Variables de entorno"
     - En "Variables del sistema", busca "Path" y haz clic en "Editar"
     - Haz clic en "Nuevo" y agrega la ruta: `C:\android-sdk\platform-tools`
     - Haz clic en "Aceptar" en todas las ventanas

3. **Verificar instalación:**
   Abre una **nueva** ventana de PowerShell/CMD y ejecuta:
   ```bash
   adb version
   ```
   Debe mostrar la versión de ADB (ej: `Android Debug Bridge version 1.0.41`)

### Paso 3: Instalar Appium

1. **Instalar Appium globalmente:**
   Abre PowerShell o CMD y ejecuta:
   ```bash
   npm install -g appium
   ```
   ⚠️ **Nota:** Esto puede tardar varios minutos. Espera a que termine completamente.

2. **Instalar el driver de UiAutomator2:**
   ```bash
   appium driver install uiautomator2
   ```

3. **Verificar instalación:**
   ```bash
   appium --version
   ```
   Debe mostrar la versión de Appium (ej: `3.1.0`)

### Paso 4: Configurar Dispositivos Android

1. **Activar Depuración USB:**
   - En tu dispositivo Android, ve a **Configuración** → **Acerca del teléfono**
   - Toca **Número de compilación** 7 veces para activar "Opciones de desarrollador"
   - Vuelve a **Configuración** → **Opciones de desarrollador**
   - Activa **Depuración USB**

2. **Conectar dispositivo:**
   - Conecta el dispositivo Android a la PC con un cable USB
   - En el dispositivo, aparecerá un diálogo: "¿Permitir depuración USB?"
   - Marca **"Permitir siempre desde este equipo"** y toca **"Permitir"**

3. **Verificar conexión:**
   ```bash
   adb devices
   ```
   Debe mostrar tu dispositivo con estado "device":
   ```
   List of devices attached
   ABC123XYZ    device
   ```

### Paso 5: Instalar Dependencias del Proyecto

1. **Navegar a la carpeta del proyecto:**
   ```bash
   cd C:\Users\javir\Desktop\CESP_APPIUM
   ```

2. **Instalar dependencias del proyecto:**
   ```bash
   npm install
   ```
   Esto instalará:
   - `webdriverio` - Framework de automatización
   - `readline-sync` - Para entrada de usuario en consola

3. **Verificar instalación:**
   Verifica que se haya creado la carpeta `node_modules` y que contenga las dependencias.

### Paso 6: Verificar que Todo Funciona

1. **Verificar dispositivos conectados:**
   ```bash
   adb devices
   ```
   Debe mostrar al menos un dispositivo con estado "device"

2. **Verificar Appium:**
   ```bash
   appium --version
   ```
   Debe mostrar la versión sin errores

3. **Verificar Node.js:**
   ```bash
   node --version
   npm --version
   ```
   Debe mostrar las versiones instaladas

### ✅ Instalación Completa

Una vez completados todos los pasos, ya puedes usar el sistema. Ve a la sección **🚀 Uso** para comenzar.

## 🚀 Uso

### Opción 1: Perfil "pruebas"
Doble clic en: **`iniciar-test.bat`**

### Opción 2: Perfil "c_especiales"
Doble clic en: **`iniciar-c_especiales.bat`**

### Proceso Automático:
1. El script detecta automáticamente todos los dispositivos Android conectados
2. Inicia un servidor Appium por cada dispositivo (en ventanas separadas)
3. Espera 15 segundos para que los servidores estén listos
4. Navega al perfil objetivo en cada dispositivo
5. Marca todos los stories como vistos
6. Espera tu respuesta

### Uso Interactivo:
Una vez iniciado, el sistema te pedirá:
```
📝 Ingresa tu respuesta (o "salir" para terminar):
```

- Escribe tu respuesta y presiona Enter
- La respuesta se enviará a **todos los dispositivos simultáneamente**
- Puedes enviar múltiples respuestas sin reiniciar
- Escribe `salir` para terminar el programa

## 📁 Estructura del Proyecto

```
CESP_APPIUM/
├── iniciar-test.bat                   # Inicia automatización para "pruebas"
├── iniciar-c_especiales.bat           # Inicia automatización para "c_especiales"
├── start-all.ps1                      # Script PowerShell para "pruebas"
├── start-all-c_especiales.ps1          # Script PowerShell para "c_especiales"
├── instagram-test-persistent.js       # Script principal para "pruebas"
├── instagram-c_especiales_persistent.js # Script principal para "c_especiales"
├── package.json                       # Dependencias del proyecto
├── package-lock.json                  # Versiones exactas de dependencias
└── node_modules/                      # Dependencias instaladas (no subir a Git)
```

## ⚙️ Configuración

### Cambiar Perfil Objetivo:
Edita el archivo JavaScript correspondiente y busca:
```javascript
await searchInput.setValue('NOMBRE_DEL_PERFIL');
```
Reemplaza `'NOMBRE_DEL_PERFIL'` con el perfil deseado.

### Cambiar Puerto Base de Appium:
Por defecto usa puertos desde `4723`. Para cambiar, edita:
- En `start-all.ps1` o `start-all-c_especiales.ps1`:
  ```powershell
  $startPort = 4723  # Cambiar este valor
  ```
- En el archivo JavaScript:
  ```javascript
  const basePort = 4723;  // Cambiar este valor
  ```

## 🌍 Soporte Multi-idioma

El sistema detecta automáticamente el idioma del dispositivo:
- **Español**: Busca "Escribe algo" y "Enviar"
- **Inglés**: Busca "Type something" y "Send"

Para agregar más idiomas, edita la sección correspondiente en el archivo JavaScript.

## ⚠️ Notas Importantes

1. **Mantén las ventanas de Appium abiertas** mientras el script esté ejecutándose
2. **No desconectes los dispositivos** durante la ejecución
3. **Asegúrate de tener suficiente batería** en los dispositivos
4. **El script mantiene sesiones activas** - puedes enviar múltiples respuestas sin reiniciar
5. **Si un dispositivo falla**, el script continuará con los demás

## 🐛 Solución de Problemas

### Error: "No se encontraron dispositivos Android conectados"
- Verifica que los dispositivos estén conectados: `adb devices`
- Asegúrate de que la depuración USB esté activada
- Revisa que los permisos de depuración estén otorgados

### Error: "No se encontró appium.cmd"
- Instala Appium globalmente: `npm install -g appium`
- Verifica la ruta en el script PowerShell

### Error: "ECONNREFUSED"
- Espera más tiempo para que los servidores Appium inicien
- Verifica que los puertos no estén en uso
- Cierra otras instancias de Appium

### El script no encuentra el perfil
- Verifica que el nombre del perfil sea correcto
- Asegúrate de estar en la pantalla de búsqueda de Instagram
- Espera unos segundos más para que carguen los resultados

## 📝 Licencia

Este proyecto es de uso personal.

## 👤 Autor

Desarrollado para automatización de respuestas en Instagram.

---

**Última actualización:** 2025-01-15


