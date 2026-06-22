@echo off
title Mergen Core Engine
color 0A
cd /d "%~dp0"
echo Gerekli TUI kutuphaneleri kontrol ediliyor...
pip install -r requirements.txt >nul 2>&1
echo Kutuphaneler hazir. Motor baslatiliyor...
timeout /t 1 >nul
python mergen_tui.py
pause
