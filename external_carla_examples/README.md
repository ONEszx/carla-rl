# External CARLA Reference Scripts

这几个脚本最初不在 `EasyCarla-RL` 项目中，而是位于仓库外的工作区根目录。为了避免遗失，同时保留它们对当前项目改造的参考价值，这里将它们单独归档到 `external_carla_examples/` 分支中。

## 文件说明

### `easy_emo.py`
- 作用：启动 CARLA 后，加载 `Town01`、生成一辆 `vehicle.tesla.model3`，开启 autopilot，并挂载一个第三人称相机持续采图。
- 特点：流程简单，适合验证 `client -> world -> spawn actor -> sensor.listen()` 这一整套最小闭环是否正常。
- 对当前项目的启发：说明“直接连本地 CARLA + 简单 autopilot 回放”是稳定可行的，因此后续策略回放脚本保留了尽量直接的执行路径。

### `maptest.py`
- 作用：连接到本地 CARLA，先 `get_world()` 再 `load_world('Town03')`，并打印加载状态。
- 特点：非常适合做地图切换与连接链路测试。
- 对当前项目的启发：这个脚本直接促成了 `EasyCarla-RL/easycarla/envs/carla_env.py` 中“先 `get_world()`，再按需 `load_world()`”的连接逻辑，而不是一上来就强制重载地图。

### `printMap.py`
- 作用：连接 CARLA 并打印 `get_available_maps()` 的结果。
- 特点：是最轻量的地图枚举与服务可达性检查脚本。
- 对当前项目的启发：适合作为排查“端口通不通 / CARLA 是否已完全启动”的快速诊断工具。

### `teach.py`
- 作用：在同步模式下创建 autopilot 车辆与相机，用阻塞队列严格按 tick 采集 300 帧图像。
- 特点：
  - 开启 `synchronous_mode`
  - 固定 `fixed_delta_seconds = 0.1`
  - `TrafficManager` 同步
  - 使用 `Queue` 保证逐帧接收和保存
- 对当前项目的启发：证明了“同步仿真 + 队列收帧”是稳定数据采集的一个可靠方案，对后续设计 CARLA 数据采集流程和时序控制很有参考价值。

### `test.py`
- 作用：整体流程与 `teach.py` 类似，但相机安装在车辆正上方，并且每个 tick 都会更新 `spectator`，形成自车顶视鸟瞰视角。
- 特点：更偏向可视化演示和鸟瞰采集，而不是普通尾随视角。
- 对当前项目的启发：帮助确认了顶视角展示方式以及“车辆位置变化 -> spectator 实时更新”的实现方式。

## 与当前项目的关系

这些脚本不是 `EasyCarla-RL` 的正式训练/推理入口，但它们提供了几个关键参考：

- 更稳定的 CARLA 连接模式
- 是否需要复用当前世界还是重载地图
- 同步模式下的可靠采集方式
- autopilot + 传感器的最小可运行闭环
- 顶视角与演示脚本的实现方式

基于这些参考，本次项目中已经落地的内容包括：

- `easycarla/envs/carla_env.py` 中更稳的连接逻辑
- `data_collection/collect_carla_dataset.py` 的手动启动 CARLA 采集流程
- `scripts/collect_autopilot_manual.bat` 的手动采集入口
- `example/run_dql_in_carla.py` 的策略回放入口

## 使用建议

- 如果你只是想测试 CARLA 服务是否启动正常，优先看 `printMap.py`。
- 如果你想验证地图加载流程，优先看 `maptest.py`。
- 如果你想做最简单的 autopilot + 相机采样验证，优先看 `easy_emo.py`。
- 如果你要参考同步采集与逐帧保存机制，优先看 `teach.py`。
- 如果你要参考鸟瞰展示与 spectator 更新，优先看 `test.py`。

## 归档原则

这些文件被单独放在 `archive/external-carla-scripts` 分支中，目的是：
- 保存外部参考材料
- 不干扰主项目 `main` 的结构
- 方便后续需要时再择优吸收进正式代码
