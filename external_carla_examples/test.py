import carla
import os
import shutil
import random
import time
import threading
from queue import Queue, Empty

try:
    # 清理输出目录
    output_dir = 'output'
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # 连接到服务器
    client = carla.Client('localhost', 2000)
    client.set_timeout(30.0)
    world = client.get_world()

    # 设置天气
    weather = carla.WeatherParameters(
        cloudiness=10.0,
        precipitation=50.0,
        sun_altitude_angle=50.0)
    world.set_weather(weather)

    # 保存原始设置
    original_settings = world.get_settings()

    # 配置同步模式（核心：确保传感器每tick触发）
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.1  # 仿真步长0.1秒 → 10Hz采集频率
    settings.no_rendering_mode = False  # 确保渲染开启（摄像头需要）
    world.apply_settings(settings)

    # 配置Traffic Manager同步
    tm = client.get_trafficmanager(8000)
    tm.set_synchronous_mode(True)

    # 创建车辆
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
    vehicle_bp.set_attribute('color', '255,255,255')
    spawn_points = world.get_map().get_spawn_points()
    spawn_point = random.choice(spawn_points)
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    vehicle.set_autopilot(True)
    tm.ignore_lights_percentage(vehicle, 100)

    # 配置摄像头（核心修改：sensor_tick=0.0，跟随仿真tick触发）
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '1280')
    camera_bp.set_attribute('image_size_y', '720')
    camera_bp.set_attribute('sensor_tick', '0.0')  # 关键：0.0表示每仿真tick触发一次
    camera_bp.set_attribute('shutter_speed', '1000')  # 避免曝光导致的帧不完整

    # 生成摄像头（固定到车辆）
    camera_transform = carla.Transform(
        carla.Location(x=0.0, y=0, z=20.0),  # 更高
        carla.Rotation(pitch=-90.0)  # 适度俯视
    )
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle,
                               attachment_type=carla.AttachmentType.Rigid)

    # ========== 核心修改1：使用队列实现阻塞式帧接收 ==========
    image_queue = Queue(maxsize=1)  # 单元素队列，确保帧按序接收
    frame_count = 0  # 手动计数，确保每帧都被记录
    saved_frames = set()  # 用集合记录已保存的帧ID，避免重复/丢失


    def save_image(image):
        """图像回调函数：将图像放入队列，而非直接保存（避免IO阻塞）"""
        try:
            # 将图像对象放入队列（阻塞直到队列有空位）
            image_queue.put(image, timeout=1.0)
        except Exception as e:
            print(f"⚠️  帧{image.frame}入队失败：{e}")


    # 启动摄像头监听
    camera.listen(save_image)
    print("📷 摄像头已启动，开始采集300帧（10Hz）...")

    # ========== 核心修改2：阻塞式采集逻辑，确保每帧必保存 ==========
    target_frames = 300  # 目标采集帧数
    collected_frames = 0  # 已成功采集的帧数

    # ===================== 仅新增：提前获取spectator对象 =====================
    spectator = world.get_spectator()  # 只获取一次，避免重复创建
    # =========================================================================

    while collected_frames < target_frames:
        try:
            # 1. 推进仿真tick（核心：每tick对应1帧）
            current_sim_frame = world.tick()

            # ===================== 核心修复：自车正上方鸟瞰视角 =====================
            # 每次tick都获取车辆的最新位置和姿态，然后更新视角
            vehicle_transform = vehicle.get_transform()
            spectator.set_transform(carla.Transform(
                # 位置：车辆正上方15米（x/y=0表示正上方，z=15是高度）
                vehicle_transform.location + carla.Location(x=0.0, y=0.0, z=20.0),
                # 角度：俯视90度（pitch=-90），朝向和车辆一致（能看到车头方向）
                carla.Rotation(pitch=-90.0, yaw=vehicle_transform.rotation.yaw)
            ))
            # ===========================================================================

            # 2. 阻塞等待当前tick的图像（最多等1秒，避免卡死）
            try:
                image = image_queue.get(timeout=1.0)
            except Empty:
                print(f"❌ 仿真帧{current_sim_frame}：未收到图像，重试...")
                continue  # 未收到图像，重新推进tick

            # 3. 同步保存图像（确保写入磁盘）
            try:
                save_path = f'output/frame_{collected_frames:06d}.png'
                image.save_to_disk(save_path, carla.ColorConverter.Raw)
                saved_frames.add(collected_frames)
                collected_frames += 1

                # 4. 进度打印
                if collected_frames % 50 == 0:
                    print(f"✅ 已保存 {collected_frames}/{target_frames} 帧（当前仿真帧：{image.frame}）")
            except Exception as e:
                print(f"⚠️  保存第{collected_frames}帧失败：{e}")
                continue  # 保存失败，重新采集该帧

        except Exception as e:
            print(f"\n❌ 采集过程出错：{e}")
            break

    # 仿真结束后，验证所有帧是否保存成功
    print("\n⏳ 采集完成，验证帧完整性...")
    time.sleep(1)  # 等待最后一批IO操作完成

    # 安全停止摄像头
    if camera.is_listening:
        camera.stop()
    print(f"\n📊 采集结束！目标帧数：{target_frames}，实际保存：{collected_frames} 张图片")

except Exception as e:
    print(f"\n❌ 运行出错：{e}")
    raise  # 抛出异常，方便排查

finally:
    # 1. 销毁摄像头（确保已停止监听）
    if 'camera' in locals():
        try:
            if camera.is_alive:
                if camera.is_listening:
                    camera.stop()
                camera.destroy()
                print("📸 摄像头已销毁")
        except:
            pass

    # 2. 销毁车辆
    if 'vehicle' in locals():
        try:
            if vehicle.is_alive:
                vehicle.destroy()
                print("🚗 车辆已销毁")
        except:
            pass

    # 3. 恢复原始仿真设置（关键）
    if 'world' in locals() and 'original_settings' in locals():
        try:
            tm.set_synchronous_mode(False)
            world.apply_settings(original_settings)
            time.sleep(0.5)
            print("🔧 仿真设置已恢复")
        except:
            pass

    # 4. 最终统计与校验
    final_file_count = len(os.listdir(output_dir)) if os.path.exists(output_dir) else 0
    print(f"\n✅ 最终结果：")
    print(f"   - 输出目录文件数：{final_file_count}")
    print(f"   - 代码计数帧数：{collected_frames}")
    print(f"   - 采集频率：10 Hz（仿真步长0.1秒）")
    print(f"   - 帧完整性：{'✅ 完整' if final_file_count == 300 else '❌ 缺失'}")