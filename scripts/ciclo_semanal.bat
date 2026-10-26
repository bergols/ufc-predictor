@echo off
REM Wrapper do ciclo semanal (fecha o card que passou, registra o proximo).
REM Mesmo motivo do auto_capture.bat: o Agendador aponta para UM arquivo, em
REM vez de uma linha de comando com aspas aninhadas -- que e onde isso quebra
REM em silencio. Ver scripts/ciclo_semanal.py para a logica.
cd /d "%~dp0.."
echo.>> "data\ciclo_semanal.log"
echo ===== %DATE% %TIME% =====>> "data\ciclo_semanal.log"
".venv\Scripts\python.exe" -m scripts.ciclo_semanal >> "data\ciclo_semanal.log" 2>&1
