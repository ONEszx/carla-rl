# EasyCarla-RL 项目详解

> 本文档对 EasyCarla-RL 项目进行全面解读，包括项目定位、架构设计、核心组件和使用方法。

---

## 1. 项目概述

### 1.1 项目定位

**EasyCarla-RL** 是一个轻量级、易于使用的 **CARLA 自动驾驶模拟器** 的 OpenAI Gym 环境接口，专门为**强化学习（RL）应用**设计。

**核心目标**：让研究人员和初学者能够高效地在模拟自动驾驶环境中训练和评估 RL 智能体，无需繁琐的工程配置。

### 1.2 主要特点

| 特点 | 描述 |
|------|------|
| **Gym 兼容** | 遵循标准 Gym 接口（`reset()`, `step()`） |
| **多模态感知** | 集成 LiDAR、车辆状态、周围车辆信息、路径点 |
| **安全感知** | 提供奖励信号和成本信号，支持安全约束学习 |
| **可配置参数** | 交通设置、车辆数量、传感器范围等均可自定义 |
| **离线数据集** | 提供超过 7,000 条轨迹、110 万步的离线数据 |

### 1.3 技术栈

```
CARLA Simulator 0.9.13+
    ↓
PyTorch 1.13.0 (GPU)
    ↓
Gym 0.26.2
    ↓
EasyCarla-RL
```

---

## 2. 项目结构

```
EasyCarla-RL/
├── easycarla/                 # 核心环境模块
│   ├── __init__.py           # Gym 注册 (carla-v0)
│   └── envs/
│       ├── __init__.py
│       └── carla_env.py       # 主环境类 (~840行)
│
├── example/                   # 高级示例
│   ├── agents/                # 智能体实现
│   │   ├── ql_diffusion.py   # Diffusion Q-Learning
│   │   ├── diffusion.py      # 扩散模型核心
│   │   ├── model.py          # MLP 网络
│   │   └── helpers.py        # 工具函数
│   ├── params_dql/           # 预训练模型权重
│   ├── utils/                # 工具函数
│   └── run_dql_in_carla.py  # 加载并运行预训练模型
│
├── data/                      # 数据集
│   └── easycarla_offline_dataset.hdf5  # 离线数据集 (~2.76GB)
│
├── easycarla_demo.py         # 快速入门演示
├── requirements.txt
├── setup.py
└── README.md
```

---

## 3. 核心组件详解

### 3.1 CarlaEnv 环境类

环境类 `CarlaEnv` 是整个项目的核心，遵循 Gym 接口标准：

#### 3.1.1 观测空间 (Observation Space)

| 组件 | 维度 | 描述 |
|------|------|------|
| `lidar` | 240 | 360° 水平激光扫描，每 1.5° 一个 bin |
| `ego_state` | 6→9 | 本车状态（位置、速度、加速度等） |
| `nearby_vehicles` | 5×6=30 | 最多 5 辆附近车辆的信息 |
| `waypoints` | 12×3=36 | 最多 12 个前方路径点 |
| `lane_info` | 2 | 车道宽度和横向偏移 |

**总维度**：307 维

```python
# 观测空间定义
self.observation_space = spaces.Dict({
    'lidar': spaces.Box(low=0.0, high=1.0, shape=(240,), dtype=np.float32),
    'ego_state': spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32),
    'nearby_vehicles': spaces.Box(low=-np.inf, high=np.inf, shape=(self.max_nearby_vehicles, 6), dtype=np.float32),
    'waypoints': spaces.Box(low=-np.inf, high=np.inf, shape=(self.max_waypoints, 4), dtype=np.float32),
    'lane_info': spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32),
})
```

#### 3.1.2 动作空间 (Action Space)

```python
# 动作空间定义
self.action_space = spaces.Box(
    low=np.array([0.0, -1.0, 0.0], dtype=np.float32),   # [throttle, steer, brake]
    high=np.array([1.0, 1.0, 1.0], dtype=np.float32)
)
```

#### 3.1.3 感知系统

**LiDAR 传感器**：
- 单通道 360° 水平扫描
- 最大范围：50 米（可配置）
- 分辨率：240 个 bin，每个 bin 1.5°

**ego_state 特征** (9 维)：
1. `ego_x, ego_y` - 全局位置
2. `ego_yaw` - 航向角
3. `speed` - 速度大小
4. `angular_velocity.z` - 角速度
5. `acceleration.x, acceleration.y` - 加速度
6. `front_vehicle_distance` - 前车距离
7. `relative_speed` - 与前车的相对速度

#### 3.1.4 奖励函数设计

```python
def _get_reward(self, obs, done):
    reward = 0.0
    
    # 1. 前进奖励（速度限制内）
    if speed <= desired_speed:
        reward += 1.0 * speed
    else:
        reward += -1.0 * (speed - desired_speed)
    
    # 2. 车道偏离惩罚
    reward += -1.0 * lateral_offset
    
    # 3. 平稳驾驶惩罚
    reward += -0.5 * abs(a_lat)
    
    # 4. 静止惩罚
    if front_distance > 10.0 and speed < 0.1:
        reward += -1.0
    
    # 5. 碰撞惩罚
    if self._is_collision:
        reward += -100.0
    
    # 6. 偏离道路惩罚
    if self._is_off_road:
        reward += -100.0
```

#### 3.1.5 成本函数（安全约束）

```python
def _get_cost(self, obs):
    cost = 0.0
    
    # 1. 碰撞成本
    if self._is_collision:
        cost += 20.0
    
    # 2. 偏离道路成本
    if self._is_off_road:
        cost += 20.0
    
    # 3. 超速成本
    if speed > desired_speed:
        cost += (speed - desired_speed) / desired_speed
    
    return cost
```

#### 3.1.6 终止条件

| 条件 | 描述 |
|------|------|
| 碰撞 | 检测到碰撞事件 |
| 超时 | 达到最大步数限制 |
| 驶出道路 | 不在可行驶车道 |
| 逆行 | 航向偏离车道方向 > 90° |
| 偏离车道 | 横向偏移超过车道宽度/2 + 1m |

### 3.2 离线数据集

**数据规模**：
- 7,000+ 条轨迹
- 110 万步 timesteps
- 2.76 GB (HDF5 格式)
- 采集自 Town03 地图

**数据比例**：专家策略:随机策略 = 8:2

**HDF5 数据结构**：
```
/
├── observations          # [N, 307] 当前观测
├── actions              # [N, 3]   [throttle, steer, brake]
├── rewards              # [N]      奖励值
├── costs                # [N]      成本值
├── done                 # [N]      是否结束
├── next_observations   # [N, 307] 下一观测
└── info
    ├── is_collision    # [N]      是否碰撞
    └── is_off_road     # [N]      是否驶出道路
```

### 3.3 预训练模型：Diffusion Q-Learning

`example/` 目录包含基于**扩散模型**的离线强化学习实现：

#### 3.3.1 算法原理

Diffusion Q-Learning 结合了：
1. **扩散模型 (Diffusion Model)**：用于建模动作分布
2. **Q-Learning**：用于估计动作价值

#### 3.3.2 网络架构

**MLP (时序嵌入网络)**：
```
Input: [action_dim + state_dim + time_dim]
  ↓
SinusoidalPosEmb(time)  # 时间步嵌入
  ↓
Linear(t_dim → t_dim*2) → Mish → Linear(t_dim*2 → t_dim)
  ↓
cat([x, t_emb, state])
  ↓
MLP(256, 256, 256)
  ↓
Linear(256 → action_dim)
```

**Critic (双 Q 网络)**：
```
state + action → Linear(256) → Mish → Linear(256) → Mish → Linear(256) → Mish → Linear(1)
```

#### 3.3.3 训练损失

```python
# 1. Critic 损失 (Q-Learning)
critic_loss = MSE(Q1, target_q) + MSE(Q2, target_q)

# 2. Actor 损失
bc_loss = actor.loss(action, state)  # 行为克隆损失
q_loss = -Q(state, new_action)       # Q 值最大化
actor_loss = bc_loss + eta * q_loss
```

---

## 4. 使用方法

### 4.1 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 安装包
pip install -e .

# 3. 启动 CARLA 服务端

# 4. 运行演示
python easycarla_demo.py
```

### 4.2 环境配置参数

```python
params = {
    'number_of_vehicles': 50,           # 周围车辆数量
    'number_of_walkers': 0,             # 行人数量
    'dt': 0.1,                          # 时间步长 (秒)
    'ego_vehicle_filter': 'vehicle.tesla.model3',  # 自动驾驶车辆
    'surrounding_vehicle_spawned_randomly': True, # 周围车辆随机生成
    'port': 2000,                        # CARLA 连接端口
    'town': 'Town03',                    # 地图
    'max_time_episode': 1000,            # 最大步数
    'max_waypoints': 12,                 # 路径点数量
    'visualize_waypoints': True,         # 可视化路径点
    'desired_speed': 8,                  # 期望速度 (m/s)
    'max_ego_spawn_times': 200,          # 最大尝试生成车辆次数
    'view_mode': 'top',                  # 'top'鸟瞰 或 'follow'跟随
    'traffic': 'off',                    # 'off'交通灯常绿
    'lidar_max_range': 50.0,             # LiDAR 最大范围
    'max_nearby_vehicles': 5,           # 最近车辆检测数量
}
```

### 4.3 基本交互流程

```python
import gym
import easycarla

# 创建环境
env = gym.make('carla-v0', params=params)

# 重置环境
obs = env.reset()

# 主循环
done = False
while not done:
    action = your_policy(obs)  # 你的策略
    next_obs, reward, cost, done, info = env.step(action)
    obs = next_obs

env.close()
```

### 4.4 加载离线数据

```python
import h5py
import torch

with h5py.File('easycarla_offline_dataset.hdf5', 'r') as f:
    observations = torch.tensor(f['observations'][:], dtype=torch.float32)
    actions = torch.tensor(f['actions'][:], dtype=torch.float32)
    rewards = torch.tensor(f['rewards'][:], dtype=torch.float32)
    next_observations = torch.tensor(f['next_observations'][:], dtype=torch.float32)
    dones = torch.tensor(f['done'][:], dtype=torch.float32)
```

---

## 5. 与论文方法的潜在结合

### 5.1 应用 ActiveRL 思想

论文 **"Active Reinforcement Learning Strategies for Offline Policy Improvement"** 提出的方法可以应用于本项目：

#### 5.1.1 表示学习

利用本项目的 307 维观测空间训练**表示模型集成**：
- 状态编码器 $E_s$：编码 ego_state, lane_info, waypoints
- 激光编码器：编码 lidar 信息
- 周围车辆编码器：编码 nearby_vehicles

#### 5.1.2 不确定性估计

```python
# 计算认知不确定性
def uncertainty(state, ensemble_models):
    embeddings = [model.encode(state) for model in ensemble_models]
    S = torch.cdist(torch.stack(embeddings), torch.stack(embeddings))
    return S.max().item()  # 最大分歧作为不确定性
```

#### 5.1.3 主动探索策略

1. **初始状态选择**：从离线数据覆盖不足的区域选择起始点
2. **轨迹收集**：从高不确定性区域开始探索
3. **早期终止**：当进入高置信区域时停止收集，避免浪费交互预算

### 5.2 潜在改进方向

| 方向 | 描述 |
|------|------|
| **数据效率提升** | 利用主动学习方法减少在线交互需求 |
| **安全约束强化** | 在不确定性高的区域（如交叉路口）增加安全约束 |
| **长尾场景覆盖** | 主动收集罕见交通场景数据（紧急刹车、异常车辆行为） |
| **多地图泛化** | 在 Town03 训练，主动探索 Town04/05 等新地图 |

---

## 6. 技术亮点

### 6.1 感知系统设计

- **自车坐标系**：所有感知数据转换到自车局部坐标系，确保旋转不变性
- **多模态融合**：LiDAR + 视觉语义（车道）+ 目标检测（车辆）+ 路径规划
- **高效计算**：360° 扫描离散化为 240 维向量，平衡精度与效率

### 6.2 安全机制

- **多层次检测**：碰撞、驶出道路、逆行、车道偏离
- **成本信号**：独立的成本函数用于安全约束学习
- **自适应终止**：根据行驶状态动态决定是否终止 episode

### 6.3 可扩展性

- **Gym 兼容**：易于与任何 RL 算法库集成（Stable-Baselines3、RLlib 等）
- **模块化设计**：传感器、奖励函数、终止条件均可独立配置
- **离线数据**：标准 HDF5 格式，支持任何离线 RL 算法的数据加载

---

## 7. 总结

EasyCarla-RL 是一个设计精良的自动驾驶模拟 RL 环境：

| 维度 | 评价 |
|------|------|
| **易用性** | ⭐⭐⭐⭐⭐ 标准 Gym 接口，配置简单 |
| **真实性** | ⭐⭐⭐⭐ CARLA 物理引擎，真实的交通场景 |
| **感知完整性** | ⭐⭐⭐⭐ LiDAR + 车辆状态 + 路径点 |
| **安全支持** | ⭐⭐⭐⭐ 独立的成本函数和终止条件 |
| **数据完整性** | ⭐⭐⭐⭐⭐ 110万步离线数据，可直接用于离线 RL |

该项目既适合初学者入门 RL 与自动驾驶仿真，也适合研究人员进行离线强化学习的算法研究。结合论文中的主动学习策略，可以进一步提升数据效率和安全性能。

---

**作者**：SilverWings (GitHub: https://github.com/silverwingsbot)

**参考资源**：
- [CARLA Simulator](https://github.com/carla-simulator/carla)
- [gym-carla](https://github.com/cjy1992/gym-carla)
- [Diffusion Q-Learning](https://github.com/Zhendong-Wang/Diffusion-Policies-for-Offline-RL)