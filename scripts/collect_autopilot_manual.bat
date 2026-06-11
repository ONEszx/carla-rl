@echo off
setlocal

set "PROJECT_ROOT=%~dp0.."
set "CONDA_BAT=D:\Anaconda\condabin\conda.bat"
set "CONDA_ENV=carlaenv"

REM ===== 手动启动 CARLA 后使用这个脚本 =====
REM 推荐先打开 CarlaUE4.exe，等待主场景完全进入，再运行本脚本。

set "OUTPUT_PATH=%PROJECT_ROOT%\data\easycarla_collect_manual_autopilot.hdf5"
set "MODE=autopilot"
set "NUM_EPISODES=2"
set "NUM_STEPS=200"
set "SEED=42"

REM ===== 连接参数 =====
REM 默认直接复用你当前已启动 CARLA 的地图，不强制 load_world。
set "PORT=2000"
set "CLIENT_TIMEOUT=30"

REM ===== EasyCarla 环境参数 =====
set "NUMBER_OF_VEHICLES=20"
set "NUMBER_OF_WALKERS=0"
set "DT=0.1"
set "TRAFFIC=off"
set "DESIRED_SPEED=8"
set "MAX_TIME_EPISODE=300"
set "MAX_WAYPOINTS=12"
set "MAX_NEARBY_VEHICLES=5"
set "VIEW_MODE=top"
set "VISUALIZE_WAYPOINTS=0"
set "MAX_EGO_SPAWN_TIMES=200"

if not exist "%CONDA_BAT%" (
  echo [ERROR] conda.bat not found: %CONDA_BAT%
  exit /b 1
)

call "%CONDA_BAT%" activate %CONDA_ENV%
if errorlevel 1 exit /b 1

echo [INFO] Please make sure CARLA is already running on localhost:%PORT%
echo [INFO] Collecting dataset to %OUTPUT_PATH%

python "%PROJECT_ROOT%\data_collection\collect_carla_dataset.py" ^
  --mode %MODE% ^
  --num_episodes %NUM_EPISODES% ^
  --num_steps %NUM_STEPS% ^
  --seed %SEED% ^
  --output_path "%OUTPUT_PATH%" ^
  --port %PORT% ^
  --client_timeout %CLIENT_TIMEOUT% ^
  --use_current_world ^
  --number_of_vehicles %NUMBER_OF_VEHICLES% ^
  --number_of_walkers %NUMBER_OF_WALKERS% ^
  --dt %DT% ^
  --traffic %TRAFFIC% ^
  --desired_speed %DESIRED_SPEED% ^
  --max_time_episode %MAX_TIME_EPISODE% ^
  --max_waypoints %MAX_WAYPOINTS% ^
  --max_nearby_vehicles %MAX_NEARBY_VEHICLES% ^
  --view_mode %VIEW_MODE% ^
  --visualize_waypoints %VISUALIZE_WAYPOINTS% ^
  --max_ego_spawn_times %MAX_EGO_SPAWN_TIMES%
