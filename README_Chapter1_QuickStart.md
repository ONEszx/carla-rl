# Chapter 1 Quick Start

> 这份文档只保留 **Chapter 1 目前最核心的入口**，以及每个入口最小怎么用。

默认前提：

- 你已经手动启动了 CARLA；
- 你当前终端已经进入 `carlaenv`；
- 你当前目录已经是 `EasyCarla-RL/`。

---

## 0. 每次开始前

```bash
conda activate carlaenv
cd D:\pycharm\carla_code\test\EasyCarla-RL
```

---

## 1. 入口总览

### 1.1 表征训练入口

文件：`example/representation/train_representation.py`

用途：

- 训练 encoder；
- 导出 priority 分数；
- 给 `encoder_only` / `full` 使用。

最小用法：

```bash
python example/representation/train_representation.py --data_path data/easycarla_offline_dataset.hdf5 --device cuda
```

输出：

- `example/params_representation/encoder_final.pth`
- `example/params_representation/priority_scores.npz`

### 1.2 离线主训练入口

文件：`example/train_diffusion_ql_priority.py`

用途：

- 训练 Chapter 1 的离线方法；
- 支持 `bc / baseline / encoder_only / full` 四种模式。

最小用法：

```bash
python example/train_diffusion_ql_priority.py --ablation baseline --data_path data/easycarla_offline_dataset.hdf5 --device cuda
```

常用模式：

- `--ablation bc` → `BC`
- `--ablation baseline` → `OfflineRL`
- `--ablation encoder_only` → 方法消融
- `--ablation full` → 离线 `Ours`

输出：

- `example/params_<ablation>/`
- 里面包含 checkpoint、training history、训练曲线

### 1.3 离线批量实验入口

文件：`example/run_ablation.py`

用途：

- 批量跑离线多组实验；
- 支持多 seed；
- 自动汇总均值±方差图。

最小用法：

```bash
python example/run_ablation.py --variants bc baseline encoder_only full --seeds 0 1 2 --device cuda
```

输出：

- `example/params_bc/seed_x/`
- `example/params_baseline/seed_x/`
- `example/params_encoder_only/seed_x/`
- `example/params_full/seed_x/`

### 1.4 在线微调入口

文件：`example/train_online_finetune.py`

用途：

- 加载离线 checkpoint；
- 在 CARLA 中采集在线数据；
- 继续做 online finetune。

它有两种模式：

- `--online_mode baseline` → plain `Offline+Online`
- `--online_mode active` → 在线 `Ours`

最小用法：

```bash
python example/train_online_finetune.py --online_mode baseline --ckpt_dir example/params_baseline --model_id 1 --offline_data_path data/easycarla_offline_dataset.hdf5 --online_epochs 1 --episodes_per_epoch 1 --max_steps_per_episode 50 --updates_per_epoch 5 --batch_size 32 --device cpu
```

active 最小用法：

```bash
python example/train_online_finetune.py --online_mode active --ckpt_dir example/params_full --model_id 1 --offline_data_path data/easycarla_offline_dataset.hdf5 --online_epochs 1 --episodes_per_epoch 1 --max_steps_per_episode 50 --updates_per_epoch 5 --batch_size 32 --device cpu
```

输出：

- `example/params_online_finetune/`
- `example/params_online_finetune/online_training_history.npz`

注意：

- 这个脚本默认总是写到 `example/params_online_finetune/`；
- 正式实验时，每跑完一次都要手动备份到别的目录。

### 1.5 CARLA 回放评测入口

文件：`example/run_dql_in_carla.py`

用途：

- 把训练好的 checkpoint 放回 CARLA 做 rollout；
- 输出统一评测指标。

最小用法：

```bash
python example/run_dql_in_carla.py --ckpt_dir example/params_baseline --model_id 1 --num_episodes 1 --max_steps 100 --device cpu
```

评测输出：

- `avg_reward`
- `avg_cost`
- `avg_steps`
- `collision_rate`
- `offroad_rate`

---

## 2. 最常用的对应关系

主表四类：

- `BC` → `train_diffusion_ql_priority.py --ablation bc`
- `OfflineRL` → `train_diffusion_ql_priority.py --ablation baseline`
- `Offline+Online` → `train_online_finetune.py --online_mode baseline`
- `Ours` → `train_online_finetune.py --online_mode active`

消融：

- `encoder_only` → `train_diffusion_ql_priority.py --ablation encoder_only`
- `full` → `train_diffusion_ql_priority.py --ablation full`

---

## 3. 最短使用顺序

如果你只想记最短流程，就按这个顺序：

1. 跑 `train_representation.py`
2. 跑 `run_ablation.py`
3. 跑 `train_online_finetune.py --online_mode baseline`
4. 跑 `train_online_finetune.py --online_mode active`
5. 跑 `run_dql_in_carla.py`
