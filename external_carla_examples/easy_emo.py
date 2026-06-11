import carla
import numpy as np
import os
import time
import shutil  # 用于删除目录

try:
    # 检查并清理输出目录
    output_dir = 'output'
    if os.path.exists(output_dir):
        print(f"检测到旧的 {output_dir} 目录，正在删除...")
        shutil.rmtree(output_dir)  # 删除目录及其内容
    os.makedirs(output_dir)  # 创建新目录
    print(f"已创建新的输出目录: {output_dir}")

    # 连接客户端
    client = carla.Client("localhost", 2000)
    client.set_timeout(20.0)
    world = client.load_world('Town01')

    # 设置天气
    weather = carla.WeatherParameters(
        cloudiness=10.0,
        precipitation=50.0,
        sun_altitude_angle=50.0)
    world.set_weather(weather)

    # 生成车辆
    model3_bp = world.get_blueprint_library().find('vehicle.tesla.model3')
    model3_bp.set_attribute('color', '255,255,255')
    spawn_points = world.get_map().get_spawn_points()
    model3 = world.spawn_actor(model3_bp, np.random.choice(spawn_points))

    # 启用自动驾驶
    model3.set_autopilot(True)
    print("车辆已启动自动驾驶")

    # 生成相机
    camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '1280')
    camera_bp.set_attribute('image_size_y', '720')
    camera_bp.set_attribute('sensor_tick', '0.05')  # 约20FPS

    # 相机位置（车辆后方）
    camera = world.spawn_actor(
        camera_bp,
        carla.Transform(carla.Location(x=-5.5, z=2.5), carla.Rotation(pitch=8.0)),
        attach_to=model3,
        attachment_type=carla.AttachmentType.SpringArm
    )

    # 图像保存回调
    saved_frames = []


    def save_image(image):
        image.save_to_disk('output/%06d.png' % image.frame)
        saved_frames.append(image.frame)
        if len(saved_frames) % 20 == 0:
            print(f"已保存 {len(saved_frames)} 张图片")


    camera.listen(save_image)

    # 获取 spectator（游戏窗口视角）
    spectator = world.get_spectator()

    print("开始采集图像，按Ctrl+C停止...")

    # 主循环：更新视角跟随车辆
    start_time = time.time()
    duration = 30  # 运行30秒

    while time.time() - start_time < duration:
        # 获取车辆位置和朝向
        vehicle_transform = model3.get_transform()

        # 设置 spectator 位置（车辆上方俯视）
        spectator.set_transform(carla.Transform(
            vehicle_transform.location + carla.Location(z=50),  # 上方50米
            carla.Rotation(pitch=-90)  # 俯视角度
        ))

        time.sleep(0.01)  # 短暂休眠，避免CPU占用过高

finally:
    # 清理资源
    if 'camera' in locals() and camera is not None:
        camera.stop()
        camera.destroy()
    if 'model3' in locals() and model3 is not None:
        model3.destroy()
    print("资源清理完成")
