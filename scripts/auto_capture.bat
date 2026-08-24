@echo off
REM Wrapper da captura automatica da linha de fechamento.
REM Existe para o Agendador de Tarefas apontar para UM arquivo, em vez de uma
REM linha de comando com aspas aninhadas -- que e onde isso quebra em silencio.
REM Registrado como tarefa horaria; ver scripts/auto_capture.py para a logica.
cd /d "%~dp0.."
echo.>> "data\auto_capture.log"
echo ===== %DATE% %TIME% =====>> "data\auto_capture.log"
".venv\Scripts\python.exe" -m scripts.auto_capture >> "data\auto_capture.log" 2>&1
