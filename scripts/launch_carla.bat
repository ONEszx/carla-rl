@echo off
setlocal

set "CARLA_EXE=E:\carla\WindowsNoEditor\CarlaUE4.exe"
set "CARLA_PORT=2000"

if not exist "%CARLA_EXE%" (
  echo [ERROR] CarlaUE4.exe not found: %CARLA_EXE%
  exit /b 1
)

start "CARLA" "%CARLA_EXE%" -carla-port=%CARLA_PORT% -quality-level=Low -windowed
