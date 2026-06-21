# Chapter 1 Training Commands

> 直接复制就能跑的核心训练命令清单。
>
> 默认前提：你已经手动启动 CARLA，终端已 `conda activate carlaenv`，当前目录是 `EasyCarla-RL/`。

---

## 0. 开始前

```bash
conda activate carlaenv
cd D:\pycharm\carla_code\test\EasyCarla-RL
```

---

## 1. 表征网络训练

### 1.1 推荐首版

```bash
python example/representation/train_representation.py --data data/easycarla_offline_dataset.hdf5 --device cuda --batch_size 256 --num_epochs 5 --max_batches_per_epoch 120 --log_every 10 --save_every 300
```

### 1.2 更充分版本

```bash
python example/representation/train_representation.py --data data/easycarla_offline_dataset.hdf5 --device cuda --batch_size 256 --num_epochs 10 --max_batches_per_epoch 200 --log_every 10 --save_every 500
```

### 1.3 输出

- `example/params_representation/encoder_final.pth`
- `example/params_representation/priority_scores.npz`
- `example/params_representation/training_log.txt`

---

## 2. 离线强化学习

### 2.1 BC

```bash
python example/train_diffusion_ql_priority.py --ablation bc --data_path data/easycarla_offline_dataset.hdf5 --seed 0 --output_dir example/params_bc/seed_0 --num_epochs 100 --steps_per_epoch 1000 --batch_size 128 --eval_freq 10 --device cuda
```

### 2.2 OfflineRL

```bash
python example/train_diffusion_ql_priority.py --ablation baseline --data_path data/easycarla_offline_dataset.hdf5 --seed 0 --output_dir example/params_baseline/seed_0 --num_epochs 100 --steps_per_epoch 1000 --batch_size 128 --eval_freq 10 --device cuda
```

### 2.3 Encoder only

```bash
python example/train_diffusion_ql_priority.py --ablation encoder_only --data_path data/easycarla_offline_dataset.hdf5 --seed 0 --output_dir example/params_encoder_only/seed_0 --encoder_ckpt example/params_representation/encoder_final.pth --priority_path example/params_representation/priority_scores.npz --num_epochs 100 --steps_per_epoch 1000 --batch_size 128 --eval_freq 10 --device cuda
```

### 2.4 Full

```bash
python example/train_diffusion_ql_priority.py --ablation full --data_path data/easycarla_offline_dataset.hdf5 --seed 0 --output_dir example/params_full/seed_0 --encoder_ckpt example/params_representation/encoder_final.pth --priority_path example/params_representation/priority_scores.npz --num_epochs 100 --steps_per_epoch 1000 --batch_size 128 --eval_freq 10 --device cuda
```

---

## 3. 在线强化学习

### 3.1 Plain baseline

```bash
python example/train_online_finetune.py --online_mode baseline --seed 0 --ckpt_dir example/params_baseline/seed_0 --offline_data_path data/easycarla_offline_dataset.hdf5 --online_epochs 10 --episodes_per_epoch 5 --max_steps_per_episode 200 --updates_per_epoch 200 --batch_size 128 --device cuda
```

### 3.2 Active online

```bash
python example/train_online_finetune.py --online_mode active --seed 0 --ckpt_dir example/params_full/seed_0 --offline_data_path data/easycarla_offline_dataset.hdf5 --online_epochs 10 --episodes_per_epoch 5 --max_steps_per_episode 200 --updates_per_epoch 200 --batch_size 128 --device cuda
```

### 3.3 输出

- `example/params_online_finetune/actor_*.pth`
- `example/params_online_finetune/critic_*.pth`
- `example/params_online_finetune/online_training_history.npz`

### 3.4 跑完后备份

```bash
xcopy /E /I /Y example\params_online_finetune example\params_online_baseline\seed_0
```

```bash
xcopy /E /I /Y example\params_online_finetune example\params_online_active\seed_0
```

---

## 4. CARLA 回放评测

### 4.1 离线或在线 checkpoint

```bash
python example/run_dql_in_carla.py --ckpt_dir example/params_online_active/seed_0 --num_episodes 5 --max_steps 300 --device cuda
```

### 4.2 输出

- `avg_reward`
- `avg_cost`
- `avg_steps`
- `collision_rate`
- `offroad_rate`

---

## 5. 一键式正式顺序

1. 跑 `representation`
2. 跑 `offline` 四类
3. 备份 `bc / baseline / encoder_only / full`
4. 跑 `online baseline`
5. 备份到 `params_online_baseline`
6. 跑 `online active`
7. 备份到 `params_online_active`
8. 跑 `run_dql_in_carla.py` 统一评测
