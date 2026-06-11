@echo off
setlocal

REM 自动启动 CARLA 的测试脚本；如果自动启动时序不稳定，优先改用 scripts\collect_autopilot_manual.bat

set "PROJECT_ROOT=%~dp0.."
set "CONDA_BAT=D:\Anaconda\condabin\conda.bat"
set "CONDA_ENV=carlaenv"
set "CARLA_EXE=E:\carla\WindowsNoEditor\CarlaUE4.exe"

if not exist "%CONDA_BAT%" (
  echo [ERROR] conda.bat not found: %CONDA_BAT%
  exit /b 1
)

if not exist "%CARLA_EXE%" (
  echo [ERROR] CarlaUE4.exe not found: %CARLA_EXE%
  exit /b 1
)

call "%CONDA_BAT%" activate %CONDA_ENV%
if errorlevel 1 exit /b 1

python "%PROJECT_ROOT%\data_collection\collect_carla_dataset.py" ^
  --launch_carla ^
  --close_carla_on_exit ^
  --carla_exe "%CARLA_EXE%" ^
  --mode autopilot ^
  --num_episodes 1 ^
  --num_steps 100 ^
  --output_path "%PROJECT_ROOT%\data\easycarla_collect_test.hdf5"
