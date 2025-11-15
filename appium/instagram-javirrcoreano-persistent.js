const { remote } = require('webdriverio');
const readlineSync = require('readline-sync');
const { execSync } = require('child_process');

// Almacenar drivers para cada dispositivo
const deviceDrivers = new Map(); // Map<udid, driver>
const devicePorts = new Map(); // Map<udid, port>
const basePort = 4723; // Puerto base para Appium
const keepAliveIntervals = new Map(); // Map<udid, intervalId> para keep-alive

// Manejadores de errores no capturados para evitar que el proceso termine
process.on('uncaughtException', (error) => {
    console.error('⚠️  Error no capturado:', error.message);
    console.log('🔄 Continuando ejecución...\n');
    // No terminar el proceso, solo registrar el error
});

process.on('unhandledRejection', (reason, promise) => {
    console.error('⚠️  Promesa rechazada no manejada:', reason);
    console.log('🔄 Continuando ejecución...\n');
    // No terminar el proceso, solo registrar el error
});

// Manejador de señales para cerrar limpiamente solo cuando sea necesario
let isShuttingDown = false;
process.on('SIGINT', () => {
    if (!isShuttingDown) {
        isShuttingDown = true;
        console.log('\n👋 Cerrando sesiones de forma controlada...');
        closeAllSessions().then(() => {
            process.exit(0);
        });
    }
});

process.on('SIGTERM', () => {
    if (!isShuttingDown) {
        isShuttingDown = true;
        console.log('\n👋 Cerrando sesiones de forma controlada...');
        closeAllSessions().then(() => {
            process.exit(0);
        });
    }
});

// Función para cerrar todas las sesiones
async function closeAllSessions() {
    // Limpiar todos los intervalos de keep-alive
    for (const [udid, intervalId] of keepAliveIntervals.entries()) {
        clearInterval(intervalId);
    }
    keepAliveIntervals.clear();
    
    const closePromises = Array.from(deviceDrivers.values()).map(async (driver) => {
        try {
            await driver.deleteSession();
        } catch (e) {
            // Ignorar errores al cerrar
        }
    });
    await Promise.all(closePromises);
    console.log('✅ Todas las sesiones cerradas.');
}

// Función para detectar dispositivos Android conectados
function getConnectedDevices() {
    try {
        const output = execSync('adb devices', { encoding: 'utf-8' });
        const lines = output.split('\n').filter(line => line.trim() && !line.includes('List of devices'));
        const devices = lines
            .map(line => {
                const parts = line.trim().split('\t');
                if (parts.length >= 2 && parts[1] === 'device') {
                    return parts[0];
                }
                return null;
            })
            .filter(udid => udid !== null);
        return devices;
    } catch (error) {
        console.error('❌ Error ejecutando adb devices:', error.message);
        return [];
    }
}

// Función para verificar si el servidor UiAutomator2 está instalado
function isUiautomator2Installed(udid) {
    try {
        const output = execSync(`adb -s ${udid} shell pm list packages io.appium.uiautomator2`, { encoding: 'utf-8' });
        const hasServer = output.includes('io.appium.uiautomator2.server');
        const hasServerTest = output.includes('io.appium.uiautomator2.server.test');
        return hasServer && hasServerTest;
    } catch (error) {
        return false;
    }
}

// Función para crear capabilities dinámicamente
function createCapabilities(udid) {
    const isInstalled = isUiautomator2Installed(udid);

const capabilities = {
    platformName: 'Android',
        'appium:udid': udid,
        'appium:deviceName': udid,
    'appium:automationName': 'UiAutomator2',
    'appium:appPackage': 'com.instagram.android',
    'appium:appActivity': 'com.instagram.mainactivity.MainActivity',
    'appium:noReset': true,
    'appium:fullReset': false,
        'appium:newCommandTimeout': 300, // 5 minutos para evitar cierres por inactividad
    'appium:autoGrantPermissions': true,
    'appium:skipDeviceInitialization': true,
        'appium:disableWindowAnimation': true
    };
    
    if (isInstalled) {
        capabilities['appium:skipServerInstallation'] = true;
    }
    
    return capabilities;
}

async function waitForElementMaxSpeed(driver, selector, timeout = 100) {
    try {
        const element = await driver.$(selector);
        await element.waitForExist({ timeout: timeout });
        return element;
    } catch (error) {
        return null;
    }
}

// Keep-alive para mantener sesiones activas
function startKeepAlive(udid, driver) {
    // Limpiar intervalo anterior si existe
    if (keepAliveIntervals.has(udid)) {
        clearInterval(keepAliveIntervals.get(udid));
    }
    
    // Ejecutar comando simple cada 30 segundos para mantener la sesión activa
    const intervalId = setInterval(async () => {
        try {
            await driver.getWindowSize(); // Comando simple que no afecta la UI
        } catch (e) {
            // Si falla, la sesión probablemente se cerró
            clearInterval(intervalId);
            keepAliveIntervals.delete(udid);
        }
    }, 30000); // Cada 30 segundos
    
    keepAliveIntervals.set(udid, intervalId);
}

// Inicializar conexión para un dispositivo específico
async function initializeDeviceConnection(udid, port) {
    try {
        console.log(`🔌 Iniciando conexión con dispositivo ${udid} en puerto ${port}...`);
        
        const capabilities = createCapabilities(udid);
        const isInstalled = isUiautomator2Installed(udid);
        
        if (isInstalled) {
            console.log(`  ✅ Servidor UiAutomator2 detectado - Saltando instalación`);
        } else {
            console.log(`  📦 Servidor UiAutomator2 no encontrado - Se instalará automáticamente`);
        }
        
        const driver = await remote({
            hostname: 'localhost',
            port: port,
            path: '/',
            capabilities: capabilities
        });
        
        console.log(`  ✅ Conexión establecida para ${udid}`);
        
        // Iniciar keep-alive para mantener la sesión activa
        startKeepAlive(udid, driver);
        
        await driver.pause(500);
        
        // ADELANTAR NAVEGACIÓN AL PERFIL
        console.log(`  🔍 Navegando al perfil en ${udid}...`);
        
        // Verificar si ya estamos en el perfil
        const alreadyInProfile = await waitForElementMaxSpeed(driver, '//*[@resource-id="com.instagram.android:id/row_profile_header_imageview"]', 500);
        if (alreadyInProfile) {
            console.log(`  ✅ Ya estamos en el perfil en ${udid}`);
            return driver;
        }
        
        // Ir a la pestaña de búsqueda
        const searchButton = await waitForElementMaxSpeed(driver, '//*[@resource-id="com.instagram.android:id/search_tab"]', 1000);
        if (searchButton) {
            await searchButton.click();
            await driver.pause(300); // Esperar a que se cargue la pantalla de búsqueda
        }

        // Buscar el perfil
        const searchInput = await waitForElementMaxSpeed(driver, '//android.widget.EditText', 500);
        if (searchInput) {
            await searchInput.click();
            await driver.pause(200);
            await searchInput.setValue('javirrcoreano');
            await driver.pause(800); // Esperar a que aparezcan los resultados
        }

        // Buscar el resultado del perfil (evitar información de cuenta)
        // Buscar por el texto del perfil y verificar que sea clickeable y esté en la lista de resultados
        const profileResults = await driver.$$('//android.widget.TextView[@text="javirrcoreano"]');
        
        // Función auxiliar para verificar si estamos en el perfil usando múltiples selectores
        async function verifyInProfile() {
            const selectors = [
                '//*[@resource-id="com.instagram.android:id/row_profile_header_imageview"]',
                '//*[@resource-id="com.instagram.android:id/profile_header_container"]',
                '//*[contains(@resource-id, "profile_header")]',
                '//*[@resource-id="com.instagram.android:id/row_profile_header_username"]',
                '//android.widget.TextView[@text="javirrcoreano" and @resource-id="com.instagram.android:id/row_profile_header_username"]'
            ];
            
            for (const selector of selectors) {
                const element = await waitForElementMaxSpeed(driver, selector, 300);
                if (element) {
                    return true;
                }
            }
            return false;
        }
        
        // Buscar el resultado correcto: debe ser clickeable y estar en un contenedor de perfil
        // Intentar buscar el resultado que esté dentro de un contenedor de perfil (no información)
        let profileFound = false;
        let clickedResult = null;
        
        // Primero intentar buscar resultados que estén dentro de contenedores de perfil
        try {
            // Buscar resultados que estén dentro de un contenedor de perfil (más específico)
            const profileContainers = await driver.$$('//*[contains(@resource-id, "row_search") or contains(@resource-id, "search_result")]//android.widget.TextView[@text="javirrcoreano"]');
            
            for (const result of profileContainers) {
                try {
                    const clickable = await result.getAttribute('clickable').catch(() => null);
                    const parent = await result.$('..').catch(() => null);
                    const parentClickable = parent ? await parent.getAttribute('clickable').catch(() => null) : null;
                    
                    // El perfil debe ser clickeable (directamente o su contenedor padre)
                    if (clickable === 'true' || parentClickable === 'true') {
                        clickedResult = clickable === 'true' ? result : parent;
                        await clickedResult.click();
                        await driver.pause(1000);
                        profileFound = true;
                        
                        const inProfile = await verifyInProfile();
                        if (inProfile) {
                            console.log(`  ✅ Perfil cargado en ${udid}`);
                            break;
                        } else {
                            await driver.pressKeyCode(4);
                            await driver.pause(500);
                            profileFound = false;
                            clickedResult = null;
                            continue;
                        }
                    }
                } catch (e) {
                    continue;
                }
            }
        } catch (e) {
            // Si falla, continuar con el método anterior
        }
        
        // Si no se encontró con el método anterior, buscar resultados clickeables directamente
        if (!profileFound) {
            for (let i = 0; i < profileResults.length; i++) {
                try {
                    const result = profileResults[i];
                    const clickable = await result.getAttribute('clickable').catch(() => null);
                    
                    // Solo intentar con resultados que sean clickeables
                    if (clickable === 'true') {
                        // Verificar que el elemento padre también sea clickeable (más confiable)
                        try {
                            const parent = await result.$('..').catch(() => null);
                            if (parent) {
                                const parentClickable = await parent.getAttribute('clickable').catch(() => null);
                                if (parentClickable === 'true') {
                                    await parent.click();
                                } else {
                                    await result.click();
                                }
                            } else {
                                await result.click();
                            }
                        } catch (e) {
                            await result.click();
                        }
                        
                        await driver.pause(1000);
                        profileFound = true;
                        
                        const inProfile = await verifyInProfile();
                        if (inProfile) {
                            console.log(`  ✅ Perfil cargado en ${udid}`);
                            break;
                        } else {
                            await driver.pressKeyCode(4);
                            await driver.pause(500);
                            profileFound = false;
                            continue;
                        }
                    }
                } catch (e) {
                    continue;
                }
            }
        }
        
        // Verificación final: asegurarse de que estamos en el perfil antes de continuar
        const finalCheck = await verifyInProfile();
        if (!finalCheck) {
            console.log(`  ⚠️  Advertencia: No se pudo confirmar que estamos en el perfil en ${udid}, pero continuando...`);
        }
        
        return driver;
        
    } catch (error) {
        console.error(`  ❌ Error conectando dispositivo ${udid}:`, error.message);
        return null;
    }
}

// Inicializar todas las conexiones en paralelo
async function initializeAllConnections() {
    const devices = getConnectedDevices();
    
    if (devices.length === 0) {
        console.error('❌ No se encontraron dispositivos Android conectados');
        return false;
    }
    
    console.log(`📱 Detectados ${devices.length} dispositivo(s):`);
    devices.forEach((udid, index) => {
        console.log(`  ${index + 1}. ${udid} → Puerto ${basePort + index}`);
    });
    console.log('');
    
    // Inicializar todas las conexiones en paralelo
    const connectionPromises = devices.map((udid, index) => {
        const port = basePort + index;
        devicePorts.set(udid, port);
        return initializeDeviceConnection(udid, port).then(driver => {
            if (driver) {
                deviceDrivers.set(udid, driver);
            }
            return { udid, driver, success: driver !== null };
        });
    });
    
    const results = await Promise.all(connectionPromises);
    const successful = results.filter(r => r.success);
    
    if (successful.length === 0) {
        console.error('❌ No se pudo conectar a ningún dispositivo');
        return false;
    }
    
    console.log(`\n✅ ${successful.length} dispositivo(s) conectado(s) y listo(s)\n`);
    return true;
}

// Verificar si una sesión está activa
async function isSessionActive(driver) {
    try {
        await driver.getWindowSize();
        return true;
    } catch (e) {
        return false;
    }
}

// Ejecutar automatización en un dispositivo específico
async function executeStoryAutomationOnDevice(udid, driver, userResponse) {
    const startTime = Date.now();
    try {
        // Verificar que la sesión esté activa antes de continuar
        const isActive = await isSessionActive(driver);
        if (!isActive) {
            throw new Error('Sesión cerrada, necesita reconexión');
        }
        // ABRIR STORY
        const storyElement = await waitForElementMaxSpeed(driver, '//*[@resource-id="com.instagram.android:id/row_profile_header_imageview"]', 100);
        if (storyElement) {
            await storyElement.click();
        }

        await driver.pause(100); // Reducido de 200 a 100
        
        // Hacer clic en la pregunta (búsqueda optimizada)
        try {
            const questionElements = await driver.$$('//*[contains(@text, "?")] | //*[contains(@content-desc, "?")]');
            if (questionElements.length > 0) {
                await questionElements[0].click();
                await driver.pause(50); // Reducido de 100 a 50
            }
        } catch (e) {
            // Continuar
        }
        
        // BUSCAR CAMPO DE TEXTO (optimizado)
        let inputField = null;
        let found = false;
        
        const editTexts = await driver.$$('//android.widget.EditText');
        const maxCheck = Math.min(editTexts.length, 5); // Limitar a primeros 5 elementos
        
        // Definir palabras clave según idioma una sola vez
        const isEnglish = udid === '48130DLAQ004NY';
        const keywords = isEnglish 
            ? ['type', 'something']
            : ['escribe', 'algo', 'responder'];
        
        for (let i = 0; i < maxCheck; i++) {
            try {
                const editText = editTexts[i];
                const location = await editText.getLocation().catch(() => null);
                if (!location || location.y >= 1500) continue;
                
                const [hint, text, contentDesc] = await Promise.all([
                    editText.getAttribute('hint').catch(() => null),
                    editText.getAttribute('text').catch(() => null),
                    editText.getAttribute('content-desc').catch(() => null)
                ]);
                
                // Verificar con palabras clave
                const hintLower = hint ? hint.toLowerCase() : '';
                const textLower = text ? text.toLowerCase() : '';
                const descLower = contentDesc ? contentDesc.toLowerCase() : '';
                
                const hasMatch = keywords.some(keyword => 
                    hintLower.includes(keyword) || textLower.includes(keyword) || descLower.includes(keyword)
                );
                
                if (hasMatch) {
                    inputField = editText;
                    found = true;
                    break;
                }
            } catch (e) {
                continue;
            }
        }
        
        if (!found && editTexts.length > 0) {
            // Buscar por ubicación (más rápido, sin verificar tamaño primero)
            for (let i = 0; i < Math.min(editTexts.length, 5); i++) {
                try {
                    const editText = editTexts[i];
                    const location = await editText.getLocation().catch(() => null);
                    if (location && location.y < 1500 && location.y > 200) {
                        inputField = editText;
                        found = true;
                        break;
                    }
                } catch (e) {
                    continue;
                }
            }
        }
        
        if (!found) {
            // Buscar por texto visible según idioma (timeout reducido)
            if (udid === '48130DLAQ004NY') {
                inputField = await waitForElementMaxSpeed(driver, '//*[contains(@text, "Type something")]', 100);
            } else {
                inputField = await waitForElementMaxSpeed(driver, '//*[contains(@text, "Escribe algo")]', 100);
            }
            if (inputField) {
                try {
                    const location = await inputField.getLocation();
                    if (location.y >= 1500) {
                        inputField = null;
                    } else {
                        found = true;
                    }
                } catch (e) {
                    inputField = null;
                }
            }
        }
        
        let useSetValue = false;
        
        if (found && inputField) {
            try {
                await inputField.click();
                await driver.pause(30); // Reducido de 50 a 30
            } catch (e) {
                try {
                    await inputField.setValue(userResponse);
                    useSetValue = true;
                } catch (e2) {
                    try {
                        await inputField.click();
                        await driver.pause(30); // Reducido de 50 a 30
                    } catch (e3) {
                        found = false;
                    }
                }
            }
        } else {
        await driver.performActions([{
            type: 'pointer',
            id: 'finger1',
            parameters: { pointerType: 'touch' },
            actions: [
                    { type: 'pointerMove', x: 500, y: 800, duration: 0 },
                { type: 'pointerDown', button: 0 },
                { type: 'pointerUp', button: 0 }
            ]
        }]);
            await driver.pause(30); // Reducido de 50 a 30
        }
        
        if (!useSetValue) {
        await driver.keys(userResponse);
        }
        
        // ENVIAR - Optimizado para máxima velocidad
        // Lógica especial para dispositivo 48130DLAQ004NY (en inglés)
        if (udid === '48130DLAQ004NY') {
            // Estrategias optimizadas con timeouts reducidos
            // Buscar en paralelo los selectores más comunes
            const [sendButton1, sendButton2, sendButton3] = await Promise.all([
                waitForElementMaxSpeed(driver, '//android.widget.TextView[@text="Send"]', 50),
                waitForElementMaxSpeed(driver, '//*[@resource-id="com.instagram.android:id/button_send"]', 50),
                waitForElementMaxSpeed(driver, '//*[contains(@text, "Send")]', 50)
            ]);
            
            let sendButton = sendButton1 || sendButton2 || sendButton3;
            
            if (!sendButton) {
                // Buscar por content-desc (timeout reducido)
                sendButton = await waitForElementMaxSpeed(driver, '//*[contains(@content-desc, "Send")]', 50);
            }
            if (!sendButton) {
                // También buscar "Enviar" por si acaso (timeout reducido)
                sendButton = await waitForElementMaxSpeed(driver, '//*[contains(@text, "Enviar")]', 50);
            }
            if (!sendButton) {
                // Buscar cualquier botón clickeable en la parte inferior (limitado a primeros 10)
                const clickableElements = await driver.$$('//*[@clickable="true"]');
                const maxCheck = Math.min(clickableElements.length, 10); // Limitar a 10 elementos
                for (let i = 0; i < maxCheck; i++) {
                    try {
                        const element = clickableElements[i];
                        const location = await element.getLocation();
                        // Filtrar primero por ubicación (más rápido)
                        if (location.y > 1500 && location.x > 800) {
                            const [text, contentDesc] = await Promise.all([
                                element.getAttribute('text').catch(() => null),
                                element.getAttribute('content-desc').catch(() => null)
                            ]);
                            if ((text && (text.toLowerCase().includes('send') || text.toLowerCase().includes('enviar'))) || 
                                (contentDesc && (contentDesc.toLowerCase().includes('send') || contentDesc.toLowerCase().includes('enviar')))) {
                                sendButton = element;
                                break;
                            }
                        }
                    } catch (e) {
                        continue;
                    }
                }
            }
            if (!sendButton) {
                // Buscar ImageView que pueda ser el botón de enviar (timeout reducido)
                sendButton = await waitForElementMaxSpeed(driver, '//android.widget.ImageView[@clickable="true"]', 50);
            }
            if (!sendButton) {
                // Último recurso: coordenadas del centro de la pantalla
                console.log(`  📍 Obteniendo dimensiones de pantalla y usando centro...`);
                try {
                    const windowSize = await driver.getWindowSize();
                    const centerX = Math.floor(windowSize.width / 2);
                    const centerY = Math.floor(windowSize.height / 2);
                    console.log(`  📐 Pantalla: ${windowSize.width}x${windowSize.height}, Centro: (${centerX}, ${centerY})`);
                    await driver.performActions([{
                        type: 'pointer',
                        id: 'finger1',
                        parameters: { pointerType: 'touch' },
                        actions: [
                            { type: 'pointerMove', x: centerX, y: centerY, duration: 0 },
                            { type: 'pointerDown', button: 0 },
                            { type: 'pointerUp', button: 0 }
                        ]
                    }]);
                } catch (e) {
                    // Si falla obtener dimensiones, usar coordenadas por defecto del centro
                    console.log(`  ⚠️  No se pudieron obtener dimensiones, usando centro por defecto (540, 960)`);
                    await driver.performActions([{
                        type: 'pointer',
                        id: 'finger1',
                        parameters: { pointerType: 'touch' },
                        actions: [
                            { type: 'pointerMove', x: 540, y: 960, duration: 0 }, // Centro típico para 1080x1920
                            { type: 'pointerDown', button: 0 },
                            { type: 'pointerUp', button: 0 }
                        ]
                    }]);
                }
            } else {
            await sendButton.click();
            }
        } else {
            // Lógica optimizada para los demás dispositivos (búsqueda en paralelo)
            const [sendButton, sendButtonById, sendButtonGeneric] = await Promise.all([
                waitForElementMaxSpeed(driver, '//android.widget.TextView[@text="Enviar"]', 20),
                waitForElementMaxSpeed(driver, '//*[@resource-id="com.instagram.android:id/button_send"]', 20),
                waitForElementMaxSpeed(driver, '//*[contains(@text, "Enviar")]', 20)
            ]);
            
        if (sendButton) {
            await sendButton.click();
            } else if (sendButtonById) {
                await sendButtonById.click();
            } else if (sendButtonGeneric) {
                    await sendButtonGeneric.click();
                } else {
                    await driver.keys(['Enter']);
            }
        }

        // Esperar mínimo tiempo después de enviar
        await driver.pause(200); // Reducido de 500 a 200
        
        // Volver al perfil para la siguiente respuesta (optimizado)
        try {
            // Presionar tecla atrás directamente (más rápido que buscar botón)
            await driver.pressKeyCode(4); // KEYCODE_BACK
            await driver.pause(150); // Reducido de 300 a 150
            
            // Verificar que estamos en el perfil (timeout reducido)
            const profileCheck = await waitForElementMaxSpeed(driver, '//*[@resource-id="com.instagram.android:id/row_profile_header_imageview"]', 500);
            if (!profileCheck) {
                // Navegar al perfil de nuevo (optimizado)
                const searchButton = await waitForElementMaxSpeed(driver, '//*[@resource-id="com.instagram.android:id/search_tab"]', 500);
                if (searchButton) {
                    await searchButton.click();
                    await driver.pause(150); // Reducido de 300 a 150
                }
                
                const searchInput = await waitForElementMaxSpeed(driver, '//android.widget.EditText', 300);
        if (searchInput) {
            await searchInput.click();
            await searchInput.setValue('javirrcoreano');
                    await driver.pause(300); // Reducido de 500 a 300
        }

        const profileResults = await driver.$$('//android.widget.TextView[@text="javirrcoreano"]');
        if (profileResults.length >= 2) {
            await profileResults[1].click();
        } else if (profileResults.length === 1) {
            await profileResults[0].click();
                }
                await driver.pause(300); // Reducido de 500 a 300
            }
        } catch (e) {
            // Si falla volver al perfil, continuar de todas formas
            // No mostrar mensaje para no ralentizar
        }

        const endTime = Date.now();
        const executionTime = endTime - startTime;
        return { success: true, time: executionTime, udid };
        
    } catch (error) {
        const endTime = Date.now();
        const executionTime = endTime - startTime;
        console.error(`  ❌ Error en ${udid}: ${error.message}`);
        
        // Verificar si la sesión sigue activa después del error
        try {
            const isActive = await isSessionActive(driver);
            if (!isActive) {
                console.error(`  ⚠️  Sesión cerrada en ${udid} después del error`);
            }
        } catch (e) {
            // Ignorar errores al verificar
        }
        
        return { success: false, time: executionTime, udid, error: error.message };
    }
}

// Marcar stories como vistos en un dispositivo
async function markStoriesAsViewedOnDevice(udid, driver) {
    const startTime = Date.now();
    try {
        // Verificar que la sesión esté activa
        const isActive = await isSessionActive(driver);
        if (!isActive) {
            throw new Error('Sesión cerrada, necesita reconexión');
        }

        console.log(`  📖 Revisando stories en ${udid}...`);
        
        // Asegurarse de estar en el perfil
        const profileCheck = await waitForElementMaxSpeed(driver, '//*[@resource-id="com.instagram.android:id/row_profile_header_imageview"]', 1000);
        if (!profileCheck) {
            // Navegar al perfil si no estamos ahí
            const searchButton = await waitForElementMaxSpeed(driver, '//*[@resource-id="com.instagram.android:id/search_tab"]', 1000);
            if (searchButton) {
                await searchButton.click();
                await driver.pause(200);
            }
            
            const searchInput = await waitForElementMaxSpeed(driver, '//android.widget.EditText', 500);
            if (searchInput) {
                await searchInput.click();
                await searchInput.setValue('javirrcoreano');
                await driver.pause(500);
            }
            
            const profileResults = await driver.$$('//android.widget.TextView[@text="javirrcoreano"]');
            if (profileResults.length >= 2) {
                await profileResults[1].click();
            } else if (profileResults.length === 1) {
                await profileResults[0].click();
            }
            await driver.pause(500);
        }

        // Abrir el story (hacer clic en el círculo del perfil)
        const storyElement = await waitForElementMaxSpeed(driver, '//*[@resource-id="com.instagram.android:id/row_profile_header_imageview"]', 1000);
        if (storyElement) {
            await storyElement.click();
            await driver.pause(800); // Reducido de 2000ms a 800ms - tiempo suficiente para abrir viewer
            
            // Verificar que el viewer se abrió usando múltiples selectores (timeouts reducidos)
            const viewer1 = await waitForElementMaxSpeed(driver, '//*[@resource-id="com.instagram.android:id/reel_viewer_container"]', 500);
            const viewer2 = await waitForElementMaxSpeed(driver, '//*[contains(@resource-id, "reel")]', 300);
            const viewer3 = await waitForElementMaxSpeed(driver, '//*[contains(@resource-id, "viewer")]', 300);
            
            if (!viewer1 && !viewer2 && !viewer3) {
                // Continuar de todas formas, no mostrar mensaje para no ralentizar
            }
        } else {
            console.log(`  ⚠️  No se encontró story en ${udid}`);
            return { success: true, time: Date.now() - startTime, udid };
        }

        // Navegar por los stories para marcarlos como vistos
        // Continuar hasta que la app cierre el story automáticamente
        let storiesViewed = 0;
        const maxStories = 100; // Límite alto para asegurar que se revisen todos
        let consecutiveNoViewer = 0;

        // Obtener dimensiones de pantalla una vez
        const windowSize = await driver.getWindowSize();
        // Borde derecho: usar 98% del ancho para asegurar que está en la zona de "siguiente story"
        const rightEdgeX = Math.floor(windowSize.width * 0.98); // 98% del ancho (borde derecho)
        // Centro vertical: mitad de la pantalla en el eje Y
        const centerY = Math.floor(windowSize.height / 2); // Centro vertical exacto
        
        console.log(`  📐 Pantalla: ${windowSize.width}x${windowSize.height}, Click en: (${rightEdgeX}, ${centerY})`);

        for (let i = 0; i < maxStories; i++) {
            try {
                // Verificar si estamos de vuelta en el perfil (más confiable - verificar primero, timeout reducido)
                const backToProfile = await waitForElementMaxSpeed(driver, '//*[@resource-id="com.instagram.android:id/row_profile_header_imageview"]', 100);
                if (backToProfile) {
                    break; // Salir sin mensaje para no ralentizar
                }
                
                // Verificar si estamos en el viewer usando múltiples selectores en paralelo (timeouts reducidos)
                const [storyViewer1, storyViewer2, storyViewer3] = await Promise.all([
                    waitForElementMaxSpeed(driver, '//*[@resource-id="com.instagram.android:id/reel_viewer_container"]', 100),
                    waitForElementMaxSpeed(driver, '//*[contains(@resource-id, "reel")]', 100),
                    waitForElementMaxSpeed(driver, '//*[contains(@resource-id, "viewer")]', 100)
                ]);
                const storyViewer = storyViewer1 || storyViewer2 || storyViewer3;
                
                if (!storyViewer) {
                    consecutiveNoViewer++;
                    if (consecutiveNoViewer >= 3 && storiesViewed === 0) {
                        break; // Abortar sin mensaje
                    }
                    if (consecutiveNoViewer >= 5) {
                        break; // Salir sin mensaje
                    }
                    await driver.pause(150); // Reducido de 300ms
                } else {
                    consecutiveNoViewer = 0;
                }
                
                // Esperar tiempo mínimo para que el story se marque como visto (optimizado)
                await driver.pause(800); // Reducido de 1500ms a 800ms - suficiente para marcar como visto
                
                // Usar tap en el borde derecho (más seguro, no cierra el viewer)
                try {
                    // Tap preciso en el borde derecho de la pantalla
        await driver.performActions([{
            type: 'pointer',
            id: 'finger1',
            parameters: { pointerType: 'touch' },
            actions: [
                            { type: 'pointerMove', x: rightEdgeX, y: centerY, duration: 0 },
                { type: 'pointerDown', button: 0 },
                            { type: 'pause', duration: 50 }, // Reducido de 80ms a 50ms
                { type: 'pointerUp', button: 0 }
            ]
        }]);
        
                    storiesViewed++;
                    
                    if (storiesViewed % 10 === 0) { // Mostrar cada 10 en vez de cada 5 para reducir logging
                        console.log(`  📊 ${udid}: ${storiesViewed} stories revisados...`);
                    }
                    
                    // Pausa mínima después del tap para que el story avance
                    await driver.pause(300); // Reducido de 600ms a 300ms
                    
                    // Verificar si volvimos al perfil (timeout reducido)
                    const backToProfile = await waitForElementMaxSpeed(driver, '//*[@resource-id="com.instagram.android:id/row_profile_header_imageview"]', 150);
                    
                    if (backToProfile) {
                        break; // Salir sin mensaje
                    }
                    
                    // No verificar el viewer después del tap para ahorrar tiempo
                    // Continuar directamente al siguiente story
                } catch (e) {
                    // Si hay error, verificar rápidamente si el viewer sigue activo
                    const stillInViewer = await waitForElementMaxSpeed(driver, '//*[@resource-id="com.instagram.android:id/reel_viewer_container"]', 100);
                    if (!stillInViewer) {
                        break;
                    }
                }
            } catch (e) {
                // Si hay error general, verificar rápidamente si el viewer sigue activo
                const stillInViewer = await waitForElementMaxSpeed(driver, '//*[@resource-id="com.instagram.android:id/reel_viewer_container"]', 100);
                if (!stillInViewer) {
                    break;
                }
            }
        }

        // Si todavía estamos en el viewer después del bucle, salir manualmente (timeouts reducidos)
        const stillInViewer = await waitForElementMaxSpeed(driver, '//*[@resource-id="com.instagram.android:id/reel_viewer_container"]', 200);
        if (stillInViewer) {
            await driver.pressKeyCode(4); // KEYCODE_BACK
            await driver.pause(300); // Reducido de 500ms a 300ms
        }
        
        // Verificar si ya estamos en el perfil (timeout reducido)
        const backToProfile = await waitForElementMaxSpeed(driver, '//*[@resource-id="com.instagram.android:id/row_profile_header_imageview"]', 300);
        if (!backToProfile) {
            // Solo navegar de nuevo si realmente no estamos en el perfil (caso raro)
            const searchButton = await waitForElementMaxSpeed(driver, '//*[@resource-id="com.instagram.android:id/search_tab"]', 500);
            if (searchButton) {
                await searchButton.click();
                await driver.pause(150); // Reducido de 200ms
            }
            
            const searchInput = await waitForElementMaxSpeed(driver, '//android.widget.EditText', 300);
            if (searchInput) {
                await searchInput.click();
                await searchInput.setValue('javirrcoreano');
                await driver.pause(300); // Reducido de 500ms
            }
            
            const profileResults = await driver.$$('//android.widget.TextView[@text="javirrcoreano"]');
            if (profileResults.length >= 2) {
                await profileResults[1].click();
            } else if (profileResults.length === 1) {
                await profileResults[0].click();
            }
            await driver.pause(300); // Reducido de 500ms
        }

        const endTime = Date.now();
        const executionTime = endTime - startTime;
        console.log(`  ✅ Stories revisados en ${udid} (${storiesViewed} stories, ${(executionTime/1000).toFixed(1)}s)`);
        return { success: true, time: executionTime, udid, storiesViewed };
        
    } catch (error) {
        const endTime = Date.now();
        const executionTime = endTime - startTime;
        console.error(`  ❌ Error revisando stories en ${udid}: ${error.message}`);
        
        // Intentar volver al perfil en caso de error
        try {
            await driver.pressKeyCode(4); // KEYCODE_BACK
            await driver.pause(300);
        } catch (e) {
            // Ignorar errores al volver
        }
        
        return { success: false, time: executionTime, udid, error: error.message };
    }
}

// Marcar stories como vistos en todos los dispositivos en paralelo
async function markStoriesAsViewedParallel() {
    const globalStartTime = Date.now();
    console.log(`\n📖 Revisando stories en ${deviceDrivers.size} dispositivo(s) en paralelo...\n`);
    
    // Ejecutar en todos los dispositivos en paralelo
    const promises = Array.from(deviceDrivers.entries()).map(([udid, driver]) => {
        return markStoriesAsViewedOnDevice(udid, driver);
    });
    
    const results = await Promise.all(promises);
    
    const globalEndTime = Date.now();
    const totalTime = globalEndTime - globalStartTime;
    
    console.log('\n📊 RESULTADOS DE REVISIÓN:');
    results.forEach(result => {
        if (result.success) {
            const storiesCount = result.storiesViewed || 0;
            console.log(`  ✅ ${result.udid}: ${storiesCount} stories revisados en ${result.time}ms (${(result.time/1000).toFixed(2)}s)`);
        } else {
            console.log(`  ❌ ${result.udid}: Error - ${result.error} (${result.time}ms)`);
        }
    });
    console.log(`\n🎉 Tiempo total de revisión (paralelo): ${totalTime}ms (${(totalTime/1000).toFixed(2)}s)\n`);
    
    return results.every(r => r.success);
}

// Ejecutar automatización en todos los dispositivos en paralelo
async function executeStoryAutomationParallel(userResponse) {
    const globalStartTime = Date.now();
    console.log(`⚡ Ejecutando automatización en ${deviceDrivers.size} dispositivo(s) en paralelo...`);
    console.log(`📝 Respuesta: "${userResponse}"\n`);
    
    // Ejecutar en todos los dispositivos en paralelo
    const promises = Array.from(deviceDrivers.entries()).map(([udid, driver]) => {
        return executeStoryAutomationOnDevice(udid, driver, userResponse);
    });
    
    const results = await Promise.all(promises);
    
    const globalEndTime = Date.now();
    const totalTime = globalEndTime - globalStartTime;
    
    console.log('\n📊 RESULTADOS:');
    results.forEach(result => {
        if (result.success) {
            console.log(`  ✅ ${result.udid}: Completado en ${result.time}ms (${(result.time/1000).toFixed(2)}s)`);
        } else {
            console.log(`  ❌ ${result.udid}: Error - ${result.error} (${result.time}ms)`);
        }
    });
    console.log(`\n🎉 Tiempo total (paralelo): ${totalTime}ms (${(totalTime/1000).toFixed(2)}s)\n`);
    
    return results.every(r => r.success);
}

async function main() {
    try {
        // Inicializar todas las conexiones
        const connected = await initializeAllConnections();
        if (!connected) {
            return;
        }

        console.log('🚀 SISTEMA LISTO - Ejecución en paralelo activada');
        console.log('💡 Perfiles ya cargados');
        console.log('⚡ La respuesta se enviará a todos los dispositivos simultáneamente\n');

        // Revisar y marcar stories como vistos antes de pedir la respuesta
        try {
            await markStoriesAsViewedParallel();
        } catch (error) {
            console.error('⚠️  Error al revisar stories:', error.message);
            console.log('🔄 Continuando de todas formas...\n');
        }

        // Loop principal
        while (true) {
            try {
                const userResponse = readlineSync.question('📝 Ingresa tu respuesta (o "salir" para terminar): ');
                
                if (userResponse.toLowerCase() === 'salir') {
                    console.log('👋 Cerrando sesiones...');
                    break;
                }
                
                if (userResponse.trim() === '') {
                    console.log('⚠️  Respuesta vacía, intenta de nuevo\n');
                    continue;
                }
                
                try {
                    const success = await executeStoryAutomationParallel(userResponse);
                
                if (success) {
                    console.log('✅ Listo para la siguiente respuesta...\n');
                } else {
                        console.log('⚠️  Algunos dispositivos tuvieron errores, revisa los resultados arriba\n');
                        console.log('💡 Las sesiones siguen activas, puedes intentar otra respuesta\n');
                    }
                } catch (execError) {
                    console.error('❌ Error ejecutando automatización:', execError.message);
                    console.log('🔄 Verificando sesiones...\n');
                    
                    // Verificar y reconectar sesiones si es necesario
                    try {
                        for (const [udid, driver] of deviceDrivers.entries()) {
                            const isActive = await isSessionActive(driver);
                            if (!isActive) {
                                console.log(`  🔄 Sesión perdida en ${udid}, intentando reconectar...`);
                                const port = devicePorts.get(udid);
                                const newDriver = await initializeDeviceConnection(udid, port);
                                if (newDriver) {
                                    deviceDrivers.set(udid, newDriver);
                                    // El keep-alive se inicia automáticamente en initializeDeviceConnection
                                    console.log(`  ✅ Reconectado ${udid}`);
                                } else {
                                    console.log(`  ❌ No se pudo reconectar ${udid}`);
                                }
                            }
                        }
                    } catch (reconnectError) {
                        console.error('  ⚠️  Error al verificar/reconectar sesiones:', reconnectError.message);
                    }
                    
                    console.log('💡 Puedes intentar otra respuesta\n');
                    // Continuar el loop sin cerrar sesiones
                }
                
            } catch (error) {
                console.error('❌ Error en el loop principal:', error.message);
                console.log('🔄 Reintentando...\n');
                // Continuar el loop sin cerrar sesiones
            }
        }
        
    } catch (error) {
        console.error('❌ Error en la función principal:', error.message);
    } finally {
        // Cerrar todas las sesiones solo si no estamos en proceso de apagado controlado
        if (!isShuttingDown) {
            console.log('🔌 Cerrando todas las sesiones...');
            await closeAllSessions();
        }
    }
}

main();
