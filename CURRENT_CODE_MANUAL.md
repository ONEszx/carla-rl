# EasyCarla-RL 当前代码说明手册

> 这份文档只讲**当前这份本地代码到底有哪些部分、每个部分负责什么、实际应该从哪些入口启动**。
> 
> 它不是论文规划文档，也不是方法综述，重点是帮你后续少走弯路。

---

## 1. 项目当前在做什么

当前这套 `EasyCarla-RL` 代码可以分成四层理解：

1. **环境层**：把 CARLA 封装成一个 Gym 风格环境；
2. **数据层**：采集 CARLA 轨迹并保存成 HDF5；
3. **离线 RL / 表征层**：基于离线数据训练 encoder 和 Diffusion QL；
4. **在线验证 / 在线微调层**：把离线训练好的策略放回 CARLA 里跑，或者继续在线微调。

从你当前第一章的角度看，真正最重要的是下面这几条主线：

- **环境入口**：`carla-v0`
- **表征训练入口**：`example/representation/train_representation.py`
- **离线主训练入口**：`example/train_diffusion_ql_priority.py`
- **离线消融总入口**：`example/run_ablation.py`
- **CARLA 回放验证入口**：`example/run_dql_in_carla.py`
- **在线微调入口**：`example/train_online_finetune.py`

如果你后面只想记“哪些最核心”，就记这 6 个入口就够了。

---

## 2. 目录结构怎么理解

### 2.1 环境与包

- `easycarla/`
  - 项目的环境包；
  - 这里定义了 `carla-v0` 这个 Gym 环境；
  - 真正的环境逻辑主要在 `easycarla/envs/carla_env.py`。

### 2.2 示例与训练代码

- `example/`
  - 训练、回放、消融、表示学习相关脚本都在这里；
  - 你第一章现在最常用的脚本基本都在这个目录下面。

子目录作用如下：

- `example/agents/`
  - RL 智能体实现；
  - 目前核心是 `ql_diffusion.py`，也就是 Diffusion Q-Learning 主体。

- `example/representation/`
  - 状态表征学习相关模块；
  - 包括 encoder、对比学习数据构造、priority score 生成。

- `example/utils/`
  - 训练过程用到的一些工具函数。

### 2.3 数据采集与数据格式

- `data_collection/`
  - 从 CARLA 采集数据并保存成 HDF5；
  - 提供自动启动 CARLA、轨迹缓存、状态 flatten 等工具。

- `data/`
  - 离线数据集所在目录；
  - 当前你已经在用的核心数据是：
    - `data/easycarla_offline_dataset.hdf5`

### 2.4 文档与规划

- `plan/`
  - 研究路线和阶段规划；
  - 更偏项目/论文推进，不是直接运行代码的地方。

- `think/`、`rethink/`
  - 主要是分析文档、草稿和思考记录；
  - 对理解项目有帮助，但不是主入口。

### 2.5 脚本

- `scripts/`
  - Windows 下的一些辅助 `.bat` 脚本；
  - 例如启动 CARLA 的 `scripts/launch_carla.bat`。

---

## 3. 环境层：CARLA 是怎么接进来的

### 3.1 环境主类

核心文件：`easycarla/envs/carla_env.py`

这个文件负责：

- 连接 CARLA 服务器；
- 加载当前世界 / 切换 Town；
- 生成 ego 车、周围车辆、行人；
- 挂载碰撞传感器和 LiDAR；
- 输出 Gym 风格的 `reset()` / `step()`；
- 计算 reward、cost、done；
- 返回观测字典。

### 3.2 环境输入是什么

环境创建时依赖一个 `params` 字典，常见参数包括：

- `port`
- `town`
- `use_current_world`
- `number_of_vehicles`
- `number_of_walkers`
- `dt`
- `max_time_episode`
- `desired_speed`
- `view_mode`
- `traffic`
- `lidar_max_range`
- `max_nearby_vehicles`

这些参数在多个脚本里都有默认配置，比如：

- `example/run_dql_in_carla.py`
- `example/train_online_finetune.py`
- `data_collection/collect_carla_dataset.py`

### 3.3 环境输出是什么

环境返回的是一个**观测字典**，不是直接的 307 维向量。

主要字段有：

- `ego_state`
- `lane_info`
- `lidar`
- `nearby_vehicles`
- `waypoints`

动作是 3 维连续控制：

- `throttle`
- `steer`
- `brake`

### 3.4 为什么训练时总说 307 维状态

因为训练代码并不直接吃原始字典，而是先把观测 flatten 成一个定长向量。

这部分定义在：`data_collection/collector_utils.py`

当前 flatten 规则是：

- `ego_state` → 9 维
- `lane_info` → 2 维
- `lidar` → 240 维
- `nearby_vehicles` → 20 维
- `waypoints` → 36 维

最后拼成：

- **307 维状态向量**

这是当前第一章所有离线/在线训练默认使用的输入格式。

---

## 4. 数据层：离线数据和采集代码是干什么的

### 4.1 当前主数据集

核心文件：`data/easycarla_offline_dataset.hdf5`

这是当前第一章离线训练最主要的数据源。

常用字段包括：

- `observations`
- `actions`
- `next_observations`
- `rewards`
- `done`

如果是采集脚本导出的更完整数据，还可能带有：

- `costs`
- `episode_ids`
- `timesteps`
- `source_mode`

### 4.2 数据采集入口

核心文件：`data_collection/collect_carla_dataset.py`

这个脚本负责：

- 连接或拉起 CARLA；
- 按指定模式采样轨迹；
- 把轨迹整理成 HDF5；
- 存到 `data/` 或你指定的输出位置。

它支持的采集模式有：

- `random`
- `autopilot`
- `policy`
- `mixed`

### 4.3 数据采集辅助模块

核心文件：`data_collection/collector_utils.py`

它主要负责：

- 定义状态维度和动作维度；
- 把观测字典 flatten 成 307 维向量；
- 暂存 transition；
- 把缓存写成 HDF5。

### 4.4 CARLA 启动辅助

核心文件：`data_collection/carla_launcher.py`

它负责：

- 启动本地 `CarlaUE4.exe`
- 等待端口 ready
- 等待 CARLA API ready
- 在需要时关闭 CARLA 进程

如果你自己手动开 CARLA，这个模块不一定要用。

---

## 5. RL 主体层：Diffusion QL 是哪一块

### 5.1 主体文件

核心文件：`example/agents/ql_diffusion.py`

这是当前第一章最核心的 RL 主体。

它里面主要包括：

- `Critic`
- `Diffusion_QL`

### 5.2 `Diffusion_QL` 负责什么

`Diffusion_QL` 负责：

- 定义 actor（扩散策略）
- 定义 twin critic
- 定义 target critic / EMA actor
- 从 replay buffer 采样训练
- 计算 BC loss + Q-learning loss
- 保存和加载模型
- 在推理时根据状态采样动作

### 5.3 现在为什么它还能接 encoder

这部分已经被你当前代码改过了。

现在的 `Diffusion_QL` 已经支持：

- 不带 encoder，直接用 raw 307 维状态；
- 带 encoder，但冻结 encoder；
- 带 encoder，并允许 RL 阶段继续 finetune encoder。

也就是说，它现在已经能服务于三种第一章实验设置：

- `baseline`
- `encoder_only`
- `full`

### 5.4 它不是哪一层

要注意，`Diffusion_QL` 只是**RL 主干**，它本身不负责：

- 从 HDF5 读 priority score；
- 做对比学习；
- 决定当前是哪个消融设置；
- 管理 CARLA 在线采样。

这些事情分别在别的入口脚本里做。

---

## 6. 表征层：encoder 和 priority 是哪一部分

### 6.1 encoder 定义

核心文件：`example/representation/encoder.py`

它定义了：

- `StateEncoder`
- `ContrastiveLoss`

当前 encoder 是一个简单 MLP：

- 输入：307 维状态
- 输出：默认 64 维 latent

### 6.2 表征训练入口

核心文件：`example/representation/train_representation.py`

这个脚本负责：

- 读取离线 HDF5 数据；
- 构造对比学习样本；
- 训练 `StateEncoder`；
- 保存 encoder checkpoint；
- 导出 priority score。

它的产物主要是：

- `example/params_representation/encoder_*.pth`
- `example/params_representation/encoder_final.pth`
- `example/params_representation/priority_scores.npz`

### 6.3 priority 是什么

当前 priority 的定位是：

- 不是在线动态更新的 PER；
- 不是跟着 RL encoder 实时漂移重算；
- 而是**基于冻结参考表征空间预先计算好的样本分数**。

这和你第一章的方法设计是一致的：

- `E_rl`：给 RL 主干用，可微调；
- `E_prio`：给 priority 参考系用，保持冻结。

### 6.4 priority 采样模块

核心文件：`example/representation/priority_sampler.py`

它负责：

- 读取 `priority_scores.npz`
- 把分数转成采样概率
- 按概率输出 batch index
- 在需要时退化成 uniform sampling

这部分本身不训练模型，只负责“怎么抽样”。

---

## 7. 第一章当前最重要的 5 个入口

下面这几项是你现在应该最熟的。

### 7.1 入口一：表征训练

文件：`example/representation/train_representation.py`

作用：

- 训练状态 encoder；
- 导出后续 `encoder_only` / `full` 会用到的 checkpoint 和 priority 分数。

什么时候用：

- 当你还没有 `encoder_final.pth` 和 `priority_scores.npz` 时；
- 或者你想重新做一版表征训练时。

### 7.2 入口二：离线主训练入口

文件：`example/train_diffusion_ql_priority.py`

作用：

- 当前第一章唯一推荐的离线主训练入口；
- 同时支持四种离线模式：
  - `bc`
  - `baseline`
  - `encoder_only`
  - `full`
- 其中实验映射关系是：
  - `bc` = 论文主表里的 `BC`
  - `baseline` = 论文主表里的 `OfflineRL`
  - `baseline -> train_online_finetune.py` = 论文主表里的 `Offline+Online`
  - `full` = 论文主表里的 `Ours`
- `encoder_only` 保留为方法消融，不属于四类主表方法。

它负责：

- 读取 HDF5 数据；
- 根据 ablation / method_type 决定使用 `Diffusion_BC` 还是 `Diffusion_QL`；
- 根据 ablation 决定是否加载 encoder；
- 根据 ablation 决定是否启用 priority sampling；
- 创建 replay buffer；
- 保存 actor / critic / encoder checkpoint；
- 输出训练曲线和历史指标。

这是第一章最核心的训练入口。

### 7.3 入口三：离线消融总控

文件：`example/run_ablation.py`

作用：

- 顺序运行多组离线实验；
- 默认只跑原来的三组消融：`baseline / encoder_only / full`；
- 也可以显式加上 `bc`，作为论文主表里的额外 baseline；
- 自动调用 `train_diffusion_ql_priority.py`；
- 自动汇总结果；
- 自动画对比图。

适合什么场景：

- 你已经准备好了 encoder 和 priority 文件；
- 想一次性跑完 `baseline / encoder_only / full`。

### 7.4 入口四：CARLA 回放验证

文件：`example/run_dql_in_carla.py`

作用：

- 把离线训练好的 checkpoint 放回 CARLA 跑；
- 做最简单的策略推理验证；
- 看模型是否能被加载、是否能连上 CARLA、是否能完成 rollout。

它支持：

- 自动解析 `actor_*.pth`
- 自动解析 `encoder_rl_<id>.pth`
- 没有 encoder 时直接回退到 raw 307 维模式

你刚刚做的最小测试，走的就是这个入口。

### 7.5 入口五：在线微调

文件：`example/train_online_finetune.py`

作用：

- 读取离线数据；
- 加载离线训练好的 Diffusion QL checkpoint；
- 在 CARLA 中采集新 transition；
- 构建 offline + online mixed replay；
- 继续在线微调同一个 RL 主干。

它当前支持两种模式：

- `--online_mode baseline`
  - plain `Offline+Online`；
  - 不加入主动选择增强；
  - 作为论文四类对比里的 `Offline+Online`。
- `--online_mode active`
  - 在同一主干上加入轻量 uncertainty heads；
  - 加入 novelty-guided retention；
  - 加入 adaptive mixed replay；
  - 作为第一章在线增强版本。

它会输出：

- `example/params_online_finetune/` 下的 checkpoint；
- `example/params_online_finetune/online_training_history.npz`。

你刚刚已经把 `baseline` 和 `active` 两条最小链路都跑通了。

---

## 8. 第一章当前推荐怎么使用这些入口

### 8.1 最简使用顺序

如果你只想按第一章主线推进，推荐顺序是：

1. `train_representation.py`
2. `run_ablation.py --variants bc baseline encoder_only full --seeds 0 1 2`
3. `train_online_finetune.py --online_mode baseline`（接 `baseline`，对应 `Offline+Online`）
4. `train_online_finetune.py --online_mode active`（接 `full`，对应在线 `Ours`）
5. `run_dql_in_carla.py`

### 8.2 如果只是先验证代码通不通

推荐顺序是：

1. 离线跑一个最小 `baseline`
2. 用 `run_dql_in_carla.py` 跑回放
3. 用 `train_online_finetune.py --online_mode baseline` 跑最小在线微调
4. 用 `train_online_finetune.py --online_mode active` 跑最小在线微调

你现在已经把这 4 步都实际跑通了。

### 8.3 如果只是做离线实验

那主要只需要：

- `train_representation.py`
- `train_diffusion_ql_priority.py`
- `run_ablation.py`

### 8.4 如果只是做在线测试

那主要只需要：

- `run_dql_in_carla.py`
- `train_online_finetune.py`

但前提是你已经有一个离线训练好的 checkpoint。

---

## 9. 当前各入口之间的关系

### 9.1 `train_representation.py` 和 `train_diffusion_ql_priority.py` 的关系

前者输出：

- `encoder_final.pth`
- `priority_scores.npz`

后者在 `encoder_only` / `full` 下会读取这些文件。

### 9.2 `train_diffusion_ql_priority.py` 和 `run_dql_in_carla.py` 的关系

前者输出：

- `actor_*.pth`
- `critic_*.pth`
- 如果用了 encoder，还会输出：
  - `encoder_rl_<epoch>.pth`
  - `encoder_rl.pth`

后者负责把这些结果加载回 CARLA。

### 9.3 `train_diffusion_ql_priority.py` 和 `train_online_finetune.py` 的关系

离线训练脚本先提供一个初始策略；
在线微调脚本在此基础上继续训练。

也就是说：

- 离线是 online 的起点；
- online 不是独立从零开始。

### 9.4 `run_ablation.py` 和主训练脚本的关系

`run_ablation.py` 不是一个新算法。

它本质上只是：

- 帮你批量调用 `train_diffusion_ql_priority.py`
- 自动把多组离线实验跑完
- 自动做离线结果汇总
- 在多 seed 情况下生成均值±方差对比图

---

## 10. 当前哪些文件是“主干”，哪些文件是“辅助”

### 10.1 当前真正的主干文件

对第一章最重要的主干文件是：

- `easycarla/envs/carla_env.py`
- `data_collection/collector_utils.py`
- `example/agents/ql_diffusion.py`
- `example/representation/encoder.py`
- `example/representation/train_representation.py`
- `example/representation/priority_sampler.py`
- `example/train_diffusion_ql_priority.py`
- `example/run_dql_in_carla.py`
- `example/train_online_finetune.py`

### 10.2 当前更偏辅助或历史用途的文件

这些文件不是没用，但不是你现在第一章最核心的入口：

- `example/train_diffusion_ql.py`
  - 更偏旧版 baseline 入口；
  - 现在建议以 `train_diffusion_ql_priority.py --ablation baseline` 为准。

- `easycarla_demo.py`
  - 更偏环境演示；
  - 不是第一章主训练入口。

- `think/`、`rethink/` 下的大部分文档
  - 更偏思考记录、论文材料和技术草稿；
  - 不是运行入口。

---

## 11. 正式实验时怎么跑（只保留最实用命令）

下面只写你正式实验时最常敲的命令，并说明每条命令的输入和输出。

### 11.1 先进入环境

```bash
conda activate carlaenv
cd D:\pycharm\carla_code\test\EasyCarla-RL
```

输入：

- `carlaenv` 环境；
- 项目根目录。

输出：

- 后续命令都在正确环境里运行。

### 11.2 训练表征模型

```bash
python example/representation/train_representation.py --data_path data/easycarla_offline_dataset.hdf5 --device cuda
```

输入：

- `data/easycarla_offline_dataset.hdf5`

输出：

- `example/params_representation/encoder_final.pth`
- `example/params_representation/priority_scores.npz`

### 11.3 正式跑离线四类 / 消融

```bash
python example/run_ablation.py --variants bc baseline encoder_only full --seeds 0 1 2 --num_epochs 100 --steps_per_epoch 1000 --batch_size 128 --device cuda
```

输入：

- 离线数据集；
- `encoder_final.pth`；
- `priority_scores.npz`；
- 3 个随机种子：`0 1 2`。

输出：

- `example/params_bc/seed_0/`、`seed_1/`、`seed_2/`
- `example/params_baseline/seed_0/`、`seed_1/`、`seed_2/`
- `example/params_encoder_only/seed_0/`、`seed_1/`、`seed_2/`
- `example/params_full/seed_0/`、`seed_1/`、`seed_2/`
- 各自的 checkpoint、training history、均值±方差汇总图

说明：

- `bc` = 论文主表里的 `BC`
- `baseline` = 论文主表里的 `OfflineRL`
- `encoder_only` = 方法消融
- `full` = 离线 `Ours`

### 11.4 正式跑 plain Online baseline

以 `seed=0` 为例：

```bash
python example/train_online_finetune.py --online_mode baseline --seed 0 --ckpt_dir example/params_baseline/seed_0 --offline_data_path data/easycarla_offline_dataset.hdf5 --online_epochs 10 --episodes_per_epoch 5 --max_steps_per_episode 200 --updates_per_epoch 200 --batch_size 128 --device cuda
```

输入：

- 离线 baseline checkpoint：`example/params_baseline/seed_0`
- 离线数据集：`data/easycarla_offline_dataset.hdf5`
- 已经启动好的 CARLA

输出：

- `example/params_online_finetune/actor_*.pth`
- `example/params_online_finetune/critic_*.pth`
- `example/params_online_finetune/online_training_history.npz`

紧接着手动备份：

```bash
xcopy /E /I /Y example\params_online_finetune example\params_online_baseline\seed_0
```

备份后的输出：

- `example/params_online_baseline/seed_0/`

说明：

- 这条对应论文主表里的 `Offline+Online`
- 在线脚本默认总是写到 `example/params_online_finetune/`，所以每跑完一个 seed 都要立刻备份

### 11.5 正式跑 active online

以 `seed=0` 为例：

```bash
python example/train_online_finetune.py --online_mode active --seed 0 --ckpt_dir example/params_full/seed_0 --offline_data_path data/easycarla_offline_dataset.hdf5 --online_epochs 10 --episodes_per_epoch 5 --max_steps_per_episode 200 --updates_per_epoch 200 --batch_size 128 --device cuda
```

输入：

- 离线 full checkpoint：`example/params_full/seed_0`
- 离线数据集：`data/easycarla_offline_dataset.hdf5`
- 已经启动好的 CARLA

输出：

- `example/params_online_finetune/actor_*.pth`
- `example/params_online_finetune/critic_*.pth`
- `example/params_online_finetune/online_training_history.npz`

紧接着手动备份：

```bash
xcopy /E /I /Y example\params_online_finetune example\params_online_active\seed_0
```

备份后的输出：

- `example/params_online_active/seed_0/`

说明：

- 这条对应论文主表里的在线 `Ours`
- `seed_1`、`seed_2` 也按同样方式重复

### 11.6 用同一命令做 CARLA 统一评测

以 `active seed_0` 为例：

```bash
python example/run_dql_in_carla.py --ckpt_dir example/params_online_active/seed_0 --num_episodes 5 --max_steps 300 --device cuda
```

输入：

- 某个训练完成的 checkpoint 目录；
- 已经启动好的 CARLA。

输出：

- 终端中的 rollout summary：
  - `avg_reward`
  - `avg_cost`
  - `avg_steps`
  - `collision_rate`
  - `offroad_rate`

正式评测时，至少要测这几类目录：

- `example/params_bc/seed_x`
- `example/params_baseline/seed_x`
- `example/params_full/seed_x`
- `example/params_online_baseline/seed_x`
- `example/params_online_active/seed_x`

### 11.7 正式实验的最短顺序

如果你只想记最少步骤，就按这个顺序：

1. 跑 `train_representation.py`
2. 跑 `run_ablation.py --variants bc baseline encoder_only full --seeds 0 1 2`
3. 对每个 `seed_x` 跑 `train_online_finetune.py --online_mode baseline --ckpt_dir example/params_baseline/seed_x`
4. 立刻备份到 `example/params_online_baseline/seed_x`
5. 对每个 `seed_x` 跑 `train_online_finetune.py --online_mode active --ckpt_dir example/params_full/seed_x`
6. 立刻备份到 `example/params_online_active/seed_x`
7. 用 `run_dql_in_carla.py` 对所有最终目录做统一评测

### 11.8 你真正要记住的对应关系

主表四类：

- `BC` → `example/params_bc/seed_x`
- `OfflineRL` → `example/params_baseline/seed_x`
- `Offline+Online` → `example/params_online_baseline/seed_x`
- `Ours` → `example/params_online_active/seed_x`

消融：

- `encoder_only` → `example/params_encoder_only/seed_x`
- `full` → `example/params_full/seed_x`

---

## 12. 当前代码已经验证过什么

你这次已经实际验证过：

### 12.1 已通过的部分

- `run_dql_in_carla.py` 能连上 CARLA；
- 能加载 `params_baseline` 中的模型；
- 能完成至少一个最小 episode rollout；
- `train_online_finetune.py --online_mode baseline` 能连上 CARLA；
- `train_online_finetune.py --online_mode active` 能连上 CARLA；
- 能采集在线数据；
- 能完成最小 mixed replay 更新；
- 能保存在线微调结果到 `example/params_online_finetune`。

### 12.2 当前 warning 的含义

你测试时看到的 `gym` warning 主要是：

- `reset()` 返回的不是 `(obs, info)`；
- `terminated` 不是严格的 `bool`；
- 某些观测和 `observation_space` 的检查不完全一致。

这说明：

- 当前环境封装和新版 `gym` 的严格接口规范并不完全一致；
- 但**不影响当前最小训练/回放闭环已经跑通**。

换句话说，这些是“后续可清理的环境接口规范问题”，不是现在第一章主线的阻塞点。

---

## 13. 对你现在最重要的结论

如果你只想抓重点，可以直接记下面这几句：

1. **环境核心**在 `easycarla/envs/carla_env.py`
2. **307 维状态定义**在 `data_collection/collector_utils.py`
3. **RL 主干**在 `example/agents/ql_diffusion.py`
4. **表征训练入口**在 `example/representation/train_representation.py`
5. **离线主训练入口**在 `example/train_diffusion_ql_priority.py`
6. **离线消融批量入口**在 `example/run_ablation.py`
7. **CARLA 回放入口**在 `example/run_dql_in_carla.py`
8. **在线微调入口**在 `example/train_online_finetune.py`

对第一章来说，你后面大部分工作基本都会围绕这 8 个点转。

---

## 14. 一个最实用的记忆版流程

你以后如果忘了当前代码怎么走，直接按这个最短流程回忆：

- **先有 CARLA 环境** → `easycarla/envs/carla_env.py`
- **先有离线数据** → `data/easycarla_offline_dataset.hdf5`
- **先训练 encoder** → `example/representation/train_representation.py`
- **再做离线 RL** → `example/train_diffusion_ql_priority.py`
- **再去 CARLA 回放** → `example/run_dql_in_carla.py`
- **再做在线微调** → `example/train_online_finetune.py`

这就是当前第一章代码最清晰的一条线。
