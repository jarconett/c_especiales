# Script maestro: Inicia servidores Appium en ventanas separadas y ejecuta el script principal

Write-Host "INICIANDO SISTEMA DE AUTOMATIZACION INSTAGRAM - CUERPOS ESPECIALES" -ForegroundColor Cyan
Write-Host ""

# Detectar dispositivos
$devices = @()
try {
    $output = adb devices 2>&1
    $lines = $output | Where-Object { $_ -match "device$" -and $_ -notmatch "List of devices" }
    $devices = $lines | ForEach-Object {
        if ($_ -match "^(\S+)\s+device$") {
            $matches[1]
        }
    }
} catch {
    Write-Host "ERROR: No se pudo ejecutar adb devices" -ForegroundColor Red
    exit 1
}

if ($devices.Count -eq 0) {
    Write-Host "ERROR: No se encontraron dispositivos Android conectados" -ForegroundColor Red
    exit 1
}

Write-Host "Detectados $($devices.Count) dispositivo(s)" -ForegroundColor Green
Write-Host ""

# Iniciar servidores Appium en ventanas separadas
$appiumPath = "C:\Users\javir\AppData\Roaming\npm\appium.cmd"

if (-not (Test-Path $appiumPath)) {
    Write-Host "ERROR: No se encontro appium.cmd en $appiumPath" -ForegroundColor Red
    Write-Host "Verifica que Appium este instalado globalmente" -ForegroundColor Yellow
    exit 1
}

Write-Host "Iniciando servidores Appium en ventanas separadas..." -ForegroundColor Yellow

# Iniciar un servidor Appium por cada dispositivo detectado
$startPort = 4723
for ($i = 0; $i -lt $devices.Count; $i++) {
    $port = $startPort + $i
    Write-Host "   Iniciando servidor en puerto $port para dispositivo $($devices[$i])..." -ForegroundColor Gray
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PWD'; & '$appiumPath' -p $port"
    Start-Sleep -Seconds 2
}

Write-Host ""
Write-Host "Esperando 15 segundos para que los servidores esten listos..." -ForegroundColor Yellow
Start-Sleep -Seconds 15
Write-Host ""

# Ejecutar script principal
Write-Host "Ejecutando script principal..." -ForegroundColor Green
Write-Host ""

if (Test-Path "instagram-c_especiales_persistent.js") {
    node instagram-c_especiales_persistent.js
} else {
    Write-Host "ERROR: No se encontro el archivo instagram-c_especiales_persistent.js" -ForegroundColor Red
    exit 1
}

