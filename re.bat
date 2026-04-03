@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "PYTHON_EXE=python"
set "SKIP_LABEL=0"

if /I "%~1"=="--skip-label" set "SKIP_LABEL=1"
if not "%~2"=="" (
  echo Usage: re.bat [--skip-label]
  exit /b 1
)

set "PROJECT_ROOT=%~dp0"
cd /d "%PROJECT_ROOT%"

echo [1/4] Stop existing board_viewer.py process...
powershell -NoProfile -ExecutionPolicy Bypass -Command "Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'python' -and $_.CommandLine -match 'board_viewer\.py' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force; Write-Output ('  Stopped PID ' + $_.ProcessId) }"

if "%SKIP_LABEL%"=="1" goto skip_label

echo [2/4] Run label.py...
"%PYTHON_EXE%" "label.py"
if errorlevel 1 (
  echo label.py failed.
  exit /b 1
)
goto start_board

:skip_label
echo [2/4] Skip label.py (because --skip-label was set).

:start_board
echo [3/4] Start board_viewer.py...
start "" /B "%PYTHON_EXE%" "board_viewer.py"

timeout /t 1 /nobreak >nul

echo [4/4] Refresh browser page...
start "" "http://127.0.0.1:8000/"
echo Done.
exit /b 0
