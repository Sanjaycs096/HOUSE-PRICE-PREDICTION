@echo off
SETLOCAL

set "BASE_PY="
where py >nul 2>&1 && set "BASE_PY=py -3.10"
if not defined BASE_PY if exist "%LocalAppData%\Programs\Python\Python310\python.exe" set "BASE_PY=%LocalAppData%\Programs\Python\Python310\python.exe"
if not defined BASE_PY set "BASE_PY=python"

set "VENV_OK=0"
if exist ".venv\Scripts\python.exe" (
    .venv\Scripts\python.exe -c "import sys" >nul 2>&1 && set "VENV_OK=1"
)

REM Recreate the virtual environment if it is missing or broken
if "%VENV_OK%"=="0" (
    if exist ".venv" rmdir /s /q ".venv"
    %BASE_PY% -m venv .venv
)

set "PYTHON=.venv\Scripts\python.exe"

REM Upgrade pip and install requirements
%PYTHON% -m pip install --upgrade pip
if exist requirements.txt (
    %PYTHON% -m pip install --upgrade --force-reinstall -r requirements.txt
)

echo Starting the Flask app...
if exist app.py (
    %PYTHON% app.py
) else (
    echo app.py not found. Please run this script from the project root.
)

pause
