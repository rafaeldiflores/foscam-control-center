@echo off
title Foscam Control Center
echo ======================================================
echo           FOSCAM FI9821W V2 - CONTROL CENTER
echo ======================================================
echo.
echo Iniciando servidor en http://localhost:8000 ...
echo Abriendo navegador automaticamente...
echo.
start http://localhost:8000
python app.py
pause
