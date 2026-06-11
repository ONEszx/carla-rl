# EffectiveRL: 两阶段离线-在线强化学习

> 面向自动驾驶的强化学习算法改进

---

## 一、整体框架

```
┌─────────────────────────────────────────────────────────┐
│  阶段1：离线强化学习（本文创新点①）                        │
├─────────────────────────────────────────────────────────┤
│  输入：现有离线数据 (1.1M样本)                            │
│  目标：更有效地利用离线数据                                │
│  创新：Uncertainty-Weighted Contrastive Learning         │
│  输出：预训练策略 + 状态表征模型                          │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  阶段2：在线主动探索（本文创新点②）                        │
├─────────────────────────────────────────────────────────┤
│  输入：离线预训练模型                                     │
│  目标：主动探索稀疏区域，补充数据                         │
│  创新：ΔU-Driven Action Selection with Adaptive Weights   │
│  输出：最终策略                                          │
└─────────────────────────────────────────────────────────┘
```

---

## 二、阶段1：离线强化学习

### 2.1 问题定义

**核心问题**：如何更有效地利用有限的离线数据？

```
现状分析：
- 离线数据分布不均，高密度区域样本过多，稀疏区域样本稀少
- 均匀采样训练浪费资源在高密度区域
- 稀疏区域学习不足
```

### 2.2 创新点①：Uncertainty-Weighted Contrastive Learning

#### 核心思想

```
论文方法：
- 使用增强噪声对比损失训练表示模型
- 损失：L = log(σ(v·v+)) + log(1-σ(v·v-)) - λ||E(s,a) - E(s')||²

本文改进：
1. 正样本：邻居状态（时间相邻），而非下一状态s'
2. 负样本：In-batch negatives（充分利用batch内样本）
3. 损失归一化：各项损失归一化后再加权
4. 不确定性加权：动态调整对比学习和转移一致性的权重
```

#### 技术方案

##### 2.2.1 表示模型集成

```python
class EnsembleEncoder(nn.Module):
    """
    表示模型集成：包含K个独立的编码器

    输入：状态向量 (batch_size, 307)
    输出：K个嵌入 (K, batch_size, 64)
    """

    def __init__(self, n_ensemble=5, input_dim=307, latent_dim=64, hidden_dim=256):
        super().__init__()
        self.encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, latent_dim)
            )
            for _ in range(n_ensemble)
        ])

    def forward(self, x):
        return torch.stack([enc(x) for enc in self.encoders], dim=0)

    def encode_mean(self, x):
        return self.forward(x).mean(dim=0)
```

##### 2.2.2 不确定性估计（O(K)方差）

```python
def compute_ensemble_uncertainty(embeddings):
    """
    用集成分歧估计认知不确定性（O(K)复杂度）

    U(s) = (1/K) * sum_k ||E^k(s) - mean(E)||²
    """
    mean_emb = embeddings.mean(dim=0, keepdim=True)
    variance = ((embeddings - mean_emb) ** 2).mean(dim=-1)
    return variance.mean(dim=0)
```

##### 2.2.3 正负样本构造

**关键设计**：
- 正样本：时间邻居（s_{t-1} 或 s_{t+1}），而非下一状态s'
- 负样本：排除邻居的随机样本（确保不是近邻）
- 转移目标：下一状态s'（单独作为正则项）

```python
class ContrastiveSampleConstructor:
    """
    对比学习正负样本构造器

    核心设计：
    - 正样本：时间邻居状态
    - 负样本：排除邻居的随机采样
    - 转移目标：下一状态s'
    """

    def __init__(self, neighbor_window=5):
        self.neighbor_window = neighbor_window

    def construct(self, states, next_states):
        """
        返回锚点、正样本、负样本的索引
        """
        N = len(states)
        anchors, positives, negatives = [], [], []

        for i in range(N):
            # 正样本：时间邻居
            if i > 0:
                anchors.append(i)
                positives.append(i - 1)
            elif i < N - 1:
                anchors.append(i)
                positives.append(i + 1)

            # 负样本：排除邻居窗口
            neighbor_range = set(range(max(0, i - self.neighbor_window),
                                        min(N, i + self.neighbor_window + 1)))
            neg_pool = [j for j in range(N) if j not in neighbor_range and j != i]
            if len(neg_pool) > 0:
                negatives.append(np.random.choice(neg_pool))

        return anchors, positives, negatives
```

##### 2.2.4 归一化对比损失（In-batch Negatives）

**关键修复**：使用in-batch negatives，而非单个负样本

```python
class NormalizedContrastiveLoss(nn.Module):
    """
    归一化不确定性加权对比损失

    关键设计：
    1. In-batch negatives：充分利用batch内所有样本
    2. 损失归一化：避免量级不一致
    3. 不确定性动态权重
    """

    def forward(self, anchor, positive, negatives, transition_target, uncertainty):
        """
        Args:
            anchor: (B, D)
            positive: (B, D)
            negatives: (B, B-1, D) 同batch内其他样本
            transition_target: (B, D)
            uncertainty: (B,)
        """
        # InfoNCE with in-batch negatives
        pos_sim = F.cosine_similarity(anchor, positive, dim=-1) / self.temperature
        neg_sim = F.cosine_similarity(
            anchor.unsqueeze(1), negatives, dim=-1
        ) / self.temperature

        all_sim = torch.cat([pos_sim.unsqueeze(-1), neg_sim], dim=-1)
        info_nce_loss = F.cross_entropy(all_sim, torch.zeros(len(anchor), dtype=torch.long, device=anchor.device))

        # 转移一致性
        transition_loss = 1 - F.cosine_similarity(anchor, transition_target, dim=-1)

        # 归一化
        uncertainty_norm = uncertainty / (uncertainty.max() + 1e-8)
        transition_weight = 0.3 + 0.7 * uncertainty_norm
        contrastive_weight = 1.0 - 0.5 * uncertainty_norm

        contrastive_loss_norm = torch.sigmoid(info_nce_loss)
        transition_loss_norm = transition_loss / 2.0

        loss = (contrastive_weight.mean() * contrastive_loss_norm +
                self.lambda_transition * transition_weight.mean() * transition_loss_norm.mean())

        return loss
```

##### 2.2.5 平滑优先级采样

```python
def compute_smooth_priority_weights(uncertainty, alpha=0.5):
    """
    平滑优先级采样：p ∝ U^α

    避免采样极端倾斜
    """
    uncertainty = np.clip(uncertainty, 1e-8, None)
    weights = uncertainty ** alpha
    return weights / weights.sum()
```

##### 2.2.6 与Diffusion QL的集成

```python
class WeightedDiffusionQL(Diffusion_QL):
    """
    加权Diffusion Q-Learning

    1. 预训练表示模型（不确定性加权对比学习 + in-batch negatives）
    2. 平滑优先级采样训练Diffusion QL
    """

    def __init__(self, *args, use_priority=True, n_ensemble=5, priority_alpha=0.5):
        super().__init__(*args)
        self.use_priority = use_priority
        self.priority_alpha = priority_alpha
        if use_priority:
            self.encoder = EnsembleEncoder(n_ensemble=n_ensemble)

    def pretrain_encoder(self, states, next_states, epochs=50):
        """预训练表示模型"""
        # 使用ContrastiveDataset + NormalizedContrastiveLoss
        pass

    def compute_priority_weights(self, states):
        """计算平滑采样权重"""
        embeddings = self.encoder(states)
        uncertainty = compute_ensemble_uncertainty(embeddings)
        return compute_smooth_priority_weights(uncertainty.cpu().numpy(), self.priority_alpha)
```

#### 合理性分析

| 设计 | 理由 |
|------|------|
| 时间邻居作为正样本 | 符合对比学习范式，语义相似 |
| In-batch negatives | 充分利用batch信息，梯度信号强 |
| 损失归一化 | 避免量级不一致 |
| O(K)方差 | 复杂度低，稳定 |
| p∝U^α平滑采样 | 避免极端倾斜 |

---

## 三、阶段2：在线主动探索

### 3.1 问题定义

**核心问题**：离线数据无法完全覆盖状态空间，如何高效探索新区域？

```
本文改进：
1. 使用Q-Ensemble方差估计动作不确定性
2. 自适应探索权重
3. 负向惩罚（相对变化率）
```

### 3.2 创新点②：ΔU-Driven Action Selection

#### 核心思想

```
1. 探索动作：添加噪声，而非选择"最远"的动作
2. 利用动作：平衡Q-Ensemble方差和Q值
3. 探索奖励：正向奖励 + 相对变化率负惩罚
```

#### 技术方案

##### 3.2.1 Q-Ensemble（用于动作不确定性）

```python
class QEnsemble(nn.Module):
    """
    Q值集成：用于估计动作的不确定性

    不确定性来源：不同Q网络对(s,a)的预测分歧
    """

    def __init__(self, state_dim=307, action_dim=3, hidden_dim=256, n_ensemble=5):
        super().__init__()
        self.q_networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(state_dim + action_dim, hidden_dim),
                nn.Mish(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Mish(),
                nn.Linear(hidden_dim, 1)
            )
            for _ in range(n_ensemble)
        ])

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return torch.stack([net(x) for net in self.q_networks], dim=-1).squeeze(-1)

    def q_uncertainty(self, state, action):
        """Q值方差"""
        return self.forward(state, action).var(dim=-1)
```

##### 3.2.2 自适应探索器

```python
class AdaptiveUncertaintyExplorer:
    """
    自适应不确定性驱动探索器

    关键设计：
    1. 使用Q-Ensemble方差（与动作相关）
    2. Batch化计算
    3. 相对变化率负惩罚
    """

    def __init__(self, policy, encoder, q_ensemble, ...):
        self.policy = policy
        self.encoder = encoder
        self.q_ensemble = q_ensemble

    def get_adaptive_alpha(self, training_progress):
        """自适应探索权重"""
        if training_progress < 0.3:
            return 0.8
        elif training_progress < 0.7:
            return 0.8 - 0.6 * (training_progress - 0.3) / 0.4
        else:
            return 0.2

    def _exploitation_action(self, state, alpha):
        """
        利用动作：平衡Q-Ensemble方差和Q值

        关键：使用Q-Ensemble方差，而非状态不确定性
        """
        device = next(self.q_ensemble.parameters()).device
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

        # 生成候选动作
        with torch.no_grad():
            base_action = self.policy.sample(state_tensor).cpu().numpy().flatten()

        candidates = [base_action]
        for _ in range(self.num_candidates - 1):
            noise = np.random.randn(*base_action.shape) * 0.15
            candidates.append(np.clip(base_action + noise, -1, 1))

        # Batch化计算（避免CPU-GPU同步）
        candidates_tensor = torch.FloatTensor(np.array(candidates)).to(device)
        state_batch = state_tensor.repeat(len(candidates), 1)

        with torch.no_grad():
            q_values = self.q_ensemble.q_mean(state_batch, candidates_tensor)
            q_unc = self.q_ensemble.q_uncertainty(state_batch, candidates_tensor)

        scores = alpha * q_unc + (1 - alpha) * q_values
        return candidates[scores.argmax().item()]

    def compute_exploration_reward(self, state, next_state, env_reward, beta=0.3):
        """
        探索奖励：使用相对变化率

        r_total = r_env + β × ΔU - penalty(relative_ΔU < threshold)
        """
        # 计算不确定性
        current_unc = compute_ensemble_uncertainty(self.encoder(state)).item()
        next_unc = compute_ensemble_uncertainty(self.encoder(next_state)).item()
        reduction = current_unc - next_unc

        # 正向奖励
        exploration_bonus = beta * max(0, reduction)

        # 负向惩罚（相对变化率）
        penalty = 0
        if current_unc > 1e-5:
            relative_reduction = reduction / current_unc
            if relative_reduction < self.negative_penalty_threshold:
                penalty = self.negative_penalty_weight * abs(relative_reduction)

        return env_reward + exploration_bonus - penalty
```

#### 与论文方法的对比

| 方面 | 论文方法 | 本文方法 |
|------|---------|---------|
| 探索目标 | 选择U最大的动作 | 添加噪声探索 |
| 利用策略 | 纯策略动作 | Q-Ensemble方差 + Q值 |
| 下一状态 | 未说明 | 真实环境交互 |
| 探索权重 | 固定 | 自适应 |
| 负惩罚 | 无 | 相对变化率 |

---

## 四、完整算法框架

```
┌─────────────────────────────────────────────────────────┐
│  阶段1：离线训练                                         │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  1. 训练EnsembleEncoder                                 │
│     - 正样本：时间邻居                                   │
│     - 负样本：In-batch negatives                        │
│     - 损失：归一化后加权                                │
│                                                         │
│  2. 平滑优先级采样Diffusion QL                          │
│     - p ∝ U^α                                           │
│                                                         │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  阶段2：在线主动探索                                     │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  1. 自适应探索权重 α (0.8 → 0.2)                        │
│                                                         │
│  2. Q-Ensemble方差估计动作不确定性                       │
│     - score = α × Q_var + (1-α) × Q_value              │
│                                                         │
│  3. 探索奖励 + 相对变化率负惩罚                         │
│     - r = r_env + β × ΔU - penalty                      │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 五、两个创新点的总结

### 创新点①：Uncertainty-Weighted Contrastive Learning

| 项目 | 内容 |
|------|------|
| **问题** | 离线数据分布不均，稀疏区域样本学习不充分 |
| **方法** | 时间邻居正样本 + In-batch negatives + 损失归一化 |
| **创新** | 正确的对比学习范式 + 数学严谨的加权 |
| **优势** | 表示质量高，针对性学习稀疏区域 |

### 创新点②：ΔU-Driven Action Selection

| 项目 | 内容 |
|------|------|
| **问题** | 固定探索策略无法平衡探索与利用 |
| **方法** | Q-Ensemble方差 + 自适应权重 + 相对变化率负惩罚 |
| **创新** | 动作相关不确定性 + 高效Batch计算 |
| **优势** | 高效探索，规避无效危险区域 |

---

## 六、实验设计

### 6.1 离线阶段消融

| 实验 | 说明 |
|------|------|
| Baseline | 论文基础对比损失 |
| - Neighbor | 使用next_state作为正样本 |
| - InBatchNeg | 使用单个负样本 |
| - Normalize | 不使用损失归一化 |
| Ours (Full) | 完整方法 |

### 6.2 在线阶段消融

| 实验 | 说明 |
|------|------|
| 论文基线 | 固定探索率 |
| - Q-Var | 使用状态不确定性代替Q-Ensemble方差 |
| - Adaptive α | 使用固定α=0.5 |
| - Relative Penalty | 使用固定阈值 |
| Ours (Full) | 完整方法 |

### 6.3 超参数消融

| 参数 | 搜索范围 |
|------|----------|
| 邻居窗口 | [1, 3, 5, 10] |
| 温度系数 | [0.05, 0.1, 0.2, 0.5] |
| 平滑因子α | [0.4, 0.5, 0.6, 0.7] |
| 负惩罚阈值 | [-0.2, -0.1, -0.05] |

---

## 七、关键修复总结

| 问题 | 原方案（错误） | 修复后 |
|------|---------------|--------|
| 正样本 | next_state | 时间邻居 |
| 负样本 | 单个负样本 | In-batch negatives |
| 动作不确定性 | 状态不确定性（与动作无关） | Q-Ensemble方差 |
| 负惩罚 | 固定阈值 | 相对变化率 |
| 计算效率 | 循环同步 | Batch化计算 |

---

最后更新：2026-06-02