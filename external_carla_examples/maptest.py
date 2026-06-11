import carla
import numpy as np
import os
import time
import shutil

# 1. 清理输出目录（保留你的逻辑）
output_dir = 'output'
if os.path.exists(output_dir):
    print(f"检测到旧的 {output_dir} 目录，正在删除...")
    shutil.rmtree(output_dir)
os.makedirs(output_dir)
print(f"已创建新的输出目录: {output_dir}")

# 2. 核心修复：连接CARLA的正确流程
try:
    # 第一步：先连接客户端（超时时间建议≥10秒）
    client = carla.Client("localhost", 2000)
    client.set_timeout(30.0)  # 延长超时时间，避免网络延迟导致连接失败
    print("✅ 客户端已连接，正在获取默认世界...")

    # 第二步：先获取默认世界，再加载指定Town（CARLA强制要求）
    world = client.get_world()  # 必须先get_world，再load_world！
    print(f"📌 开始加载地图 Town03...")
    world = client.load_world('Town03')  # 加载指定地图
    print(f"✅ 地图 Town03 加载完成！")

    # 第三步：设置天气（你的逻辑保留）
    weather = carla.WeatherParameters(
        cloudiness=100.0,
        precipitation=50.0,
        sun_altitude_angle=50.0)
    world.set_weather(weather)
    print("✅ 天气设置完成！")

    # 验证：打印当前地图名称，确认连接成功
    print(f"📊 当前运行地图：{world.get_map().name}")

    # 保持运行（避免程序退出，方便查看UE4窗口）
    print("🛑 按 Ctrl+C 退出程序...")
    while True:
        time.sleep(1)

except KeyboardInterrupt:
    print("\n✅ 程序正常退出")
except Exception as e:
    print(f"❌ 连接/运行失败：{str(e)}")
    # 常见错误排查提示
    print("🔍 排查建议：")
    print("   1. 确认 CarlaUE4.exe 已启动（路径：CARLA安装目录/WindowsNoEditor/CarlaUE4.exe）")
    print("   2. 确认端口 2000 未被占用（关闭其他CARLA程序/重启电脑）")
    print("   3. 若远程连接，将 'localhost' 改为CARLA服务器的IP地址")