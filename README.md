# Foscam Control Center - FI9821W V2

Sistema integral y moderno de control, transmisión en vivo de ultra baja latencia, audio bidireccional (Walkie-Talkie), grabación continua 24/7 con respaldo en Google Drive (5 TB) y soporte para Aplicación Móvil PWA para la cámara IP **Foscam FI9821W V2**.

---

## 🚀 Cómo Iniciar la Aplicación

Tienes dos accesos directos por lote listos para usar:

1. **`start.bat` (Uso Local en Casa):**
   * Inicia el servidor en `http://localhost:8000` y abre tu navegador automáticamente.
   * Cualquier dispositivo en tu misma red Wi-Fi puede acceder mediante `http://192.168.1.6:8000`.

2. **`start_con_tunel.bat` (Acceso Remoto desde Cualquier Lugar):**
   * Inicia el servidor local y abre un túnel seguro mediante Cloudflare.
   * La ventana del túnel te mostrará una dirección pública HTTPS (ej. `https://xxxx.trycloudflare.com`) accesible desde tu celular sin abrir puertos en el router.

---

## 📱 App Móvil Portátil (PWA)

La aplicación web funciona como una **Progressive Web App (PWA)** nativa:
* **En Android (Chrome / Brave / Edge):** Pulsa el botón **"App Móvil"** en la barra superior o en el menú `⋮` del navegador selecciona **"Instalar aplicación"**.
* **En iOS (iPhone / iPad en Safari):** Pulsa el botón Compartir y elige **"Agregar a pantalla de inicio"**.
* Abre en pantalla completa, con icono propio, sin barra del navegador y con rendimiento acelerado por hardware para video a 60fps en cualquier parte del mundo.

---

## 🎮 Funcionalidades Principales

* **Transmisión de Video Fluida y en Tiempo Real:** Decodificación RTSP directa por hardware vía FFmpeg con latencia mínima (<150 ms) y selector de calidad (HD 720p vs VGA).
* **Botón de Reconexión Rápida de Cámara:** Reinicia de inmediato el worker RTSP y el flujo WebSocket sin tener que refrescar el navegador.
* **Monitoreo de Nivel de Señal Wi-Fi:** Muestra el porcentaje en tiempo real, calidad del enlace (Excelente, Buena, Débil) y BSSID del router.
* **Control PTZ Motorizado Preciso:**
  * **Modo Paso a Paso (Recomendado):** Toques con microajuste de 120ms, 250ms o 500ms que frenan automáticamente para evitar sobregiros.
  * **Modo Continuo:** Movimiento sostenido tipo joystick.
  * **Controles HUD en Pantalla:** Flechas interactivas transparentes sobre el propio reproductor de video.
  * **Atajos de Teclado:** Flechas `⬆️⬇️⬅️➡️`, `Espacio` para frenar de emergencia y `C` para centrar.
  * **Patrullaje Automático y Presets:** Guardar y llamar posiciones guardadas con 1 clic.
* **Grabación Continua 24/7 & Nube Google Drive (5 TB):**
  * Segmentación automática en MP4 con FFmpeg stream copy (**0% de consumo de CPU** en el PC).
  * Audio AAC codificado para reproducción directa universal en móviles y navegadores.
  * Sincronización automática en segundo plano con Google Drive.
  * Barra de cuota en vivo (5 TB) y configuración directa de credenciales en la pestaña *"Grabación / 5TB"*.
  * Reproductor de video integrado para ver grabaciones inmediatamente en la app.
  * Opción de auto-borrado local para conservar espacio en disco tras la subida exitosa a la nube.
* **Audio Bidireccional (Walkie-Talkie & Sound Board):**
  * Protocolo propietario Foscam HD PCM 16-bit 8000 Hz decodificado e implementado mediante WebSocket.
  * Botón PTT interactivo con selector manos libres (1 clic).
  * Panel de efectos sonoros instantáneos (Timbre, Sirena policial, Alerta, Ladrido de perro disuasivo).
  * Transmisión de notas de voz y archivos de audio personalizados al parlante de la cámara.
* **Visión Nocturna & Control de Infrarrojos:** Modos automático y forzado manual.
* **Detección de Movimiento & Alarmas:** Control de sensibilidad y alertas en tiempo real.
* **Galería de Capturas:** Fotos tomadas en resolución nativa de la cámara guardadas en la carpeta `snapshots/`.
* **Configuración Wi-Fi & Escaneo:** Conexión y escaneo de redes cercanas.
* **Sincronización Horaria:** Sincronización del reloj interno de la cámara con el de tu PC.

---

## 🛠️ Estructura del Proyecto

```text
C:\app camara\
├── app.py                # Servidor backend principal FastAPI y WebSockets
├── talk_protocol.py      # Módulo de protocolo de audio talkback Foscam
├── recorder.py           # Motor de grabación continua segmentada MP4
├── gdrive_sync.py        # Sincronizador de nube Google Drive API v3
├── static/
│   ├── index.html        # Dashboard frontend interactivo (Tailwind + Lucide + PWA)
│   ├── manifest.json     # Manifiesto PWA para instalación en Android / iOS
│   ├── sw.js             # Service Worker para funcionamiento offline y caché
│   ├── icon-192.png      # Icono de app móvil 192x192
│   └── icon-512.png      # Icono de app móvil 512x512
├── recordings/           # Videos segmentados grabados localmente
├── snapshots/            # Fotos capturadas desde la cámara
├── credentials/          # Claves JSON de Google Drive (protegidas en .gitignore)
├── requirements.txt      # Dependencias de Python
└── README.md             # Documentación del sistema
```
