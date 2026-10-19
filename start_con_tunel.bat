@echo off
title Foscam Control Center + Tunel Cloudflare
echo ======================================================
echo    FOSCAM CONTROL CENTER + TUNEL PUBLICO REMOTO
echo ======================================================
echo.
echo 1. Iniciando servidor local en segundo plano...
start "Foscam FastAPI Server" python app.py
timeout /t 3 /nobreak >nul
echo.
echo 2. Iniciando tunel Cloudflare (ver ventana emergente para URL publica)...
start "Cloudflare Tunnel" cloudflared.exe tunnel --url http://127.0.0.1:8000
echo.
echo 3. Abriendo panel en tu navegador...
start http://localhost:8000
echo.
echo ======================================================
echo Listo! Para detener el sistema cierra las ventanas.
echo ======================================================
pause
