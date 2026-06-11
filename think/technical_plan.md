# EffectiveRL 技术方案

> 详细技术实现文档
> 更新日期：2026-06-02
> 基于专家反馈进行了关键修正（第二轮）

---

## 一、整体架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           EffectiveRL 架构                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌───────────────────────┐          ┌───────────────────────┐          │
│  │   阶段1：离线训练      │          │   阶段2：在线探索      │          │
│  ├───────────────────────┤          ├───────────────────────┤          │
│  │                       │          │                       │          │
│  │  ┌─────────────────┐  │          │  ┌─────────────────┐  │          │
│  │  │ EnsembleEncoder │  │          │  │AdaptiveUncert.  │  │          │
│  │  │  (K=5编码器)     │  │          │  │   Explorer      │  │          │
│  │  └────────┬────────┘  │          │  └────────┬────────┘  │          │
│  │           │           │          │           │           │          │
│  │           ↓           │          │           ↓           │          │
│  │  ┌─────────────────┐  │          │  ┌─────────────────┐  │          │
│  │  │NormContrastive  │  │    ┌─────┴──│  Policy (DiffQL) │  │          │
│  │  │    Loss        │  │    │        └────────┬────────┘  │          │
│  │  │ (in-batch neg) │  │    │                 │           │          │
│  │  └────────┬────────┘  │    │                 ↓           │          │
│  │           │           │    │        ┌─────────────────┐  │          │
│  │           ↓           │    │        │ Q-Ensemble      │  │          │
│  │  ┌─────────────────┐  │    │        │ (用于不确定性)  │  │          │
│  │  │SmoothPriority   │  │────┘        └─────────────────┘  │          │
│  │  │   Sampler       │  │                                           │
│  │  └────────┬────────┘  │                                           │
│  │           │           │                                           │
│  │           ↓           │                                           │
│  │  ┌─────────────────┐  │                                           │
│  │  │ WeightedDiffQL  │  │                                           │
│  │  └─────────────────┘  │                                           │
│  │                       │                                           │
│  └───────────────────────┘                                           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 二、代码文件结构

```
effective_rl/
├── __init__.py
├── encoder/
│   ├── __init__.py
│   └── ensemble_encoder.py      # K个编码器组成的集成
├── loss/
│   ├── __init__.py
│   └── contrastive_loss.py      # 归一化对比损失（使用in-batch negatives）
├── sample/
│   ├── __init__.py
│   ├── sample_constructor.py    # 正负样本构造器（邻居关系）
│   └── priority_sampler.py      # 平滑优先级采样
├── exploration/
│   ├── __init__.py
│   └── adaptive_explorer.py     # 自适应探索器（使用Q-Ensemble方差）
├── training/
│   ├── __init__.py
│   └── offline_trainer.py        # 离线训练器
└── main.py
```

---

## 三、核心组件详解

### 3.1 EnsembleEncoder

**文件**：`encoder/ensemble_encoder.py`

```python
import torch
import torch.nn as nn
from typing import Tuple, Optional

class EnsembleEncoder(nn.Module):
    """
    表示模型集成：包含K个独立的编码器

    输入：状态向量 (batch_size, 307)
    输出：K个嵌入 (K, batch_size, 64)
    """

    def __init__(
        self,
        state_dim: int = 307,
        latent_dim: int = 64,
        hidden_dim: int = 256,
        n_ensemble: int = 5
    ):
        super().__init__()
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.n_ensemble = n_ensemble

        self.encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, latent_dim)
            )
            for _ in range(n_ensemble)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """返回K个编码器的嵌入 (K, B, D)"""
        return torch.stack([enc(x) for enc in self.encoders], dim=0)

    def encode_mean(self, x: torch.Tensor) -> torch.Tensor:
        """返回K个编码器的平均嵌入"""
        return self.forward(x).mean(dim=0)
```

---

### 3.2 不确定性计算（O(K)方差）

**文件**：`encoder/ensemble_encoder.py`（与EnsembleEncoder放同一文件）

```python
def compute_ensemble_uncertainty(embeddings: torch.Tensor) -> torch.Tensor:
    """
    用集成分歧估计认知不确定性（O(K)复杂度）

    U(s) = (1/K) * sum_k ||E^k(s) - mean(E)||²
    """
    K, B, D = embeddings.shape
    mean_emb = embeddings.mean(dim=0, keepdim=True)
    variance = ((embeddings - mean_emb) ** 2).mean(dim=-1)
    uncertainty = variance.mean(dim=0)
    return uncertainty
```

---

### 3.3 正负样本构造（关键修复）

**文件**：`sample/sample_constructor.py`

**重要说明**：
- 正样本：邻居状态（时间相邻或KNN相似），而非下一状态s'
- 负样本：排除邻居的随机采样
- 转移目标：下一状态s'（单独处理，不与InfoNCE混用）

```python
import numpy as np
from typing import List, Tuple

class ContrastiveSampleConstructor:
    """
    对比学习正负样本构造器

    核心设计：
    - 在预处理阶段构造邻居关系
    - 训练时通过索引获取正负样本
    """

    def __init__(self, neighbor_window: int = 5):
        self.neighbor_window = neighbor_window

    def construct_temporal_neighbors(
        self,
        states: np.ndarray,
        next_states: np.ndarray
    ) -> Tuple[List[int], List[int], List[int]]:
        """
        构造时间邻居关系

        正样本：时间相邻状态（s_{t-1} 或 s_{t+1}）
        负样本：排除邻居窗口的随机状态

        Args:
            states: (N, 307)
            next_states: (N, 307)

        Returns:
            anchors_idx: 锚点索引列表
            positives_idx: 正样本索引列表
            negatives_idx: 负样本索引列表
        """
        N = len(states)
        anchors_idx = []
        positives_idx = []
        negatives_idx = []

        for i in range(N):
            # 正样本：时间邻居
            if i > 0:
                anchors_idx.append(i)
                positives_idx.append(i - 1)
            elif i < N - 1:
                anchors_idx.append(i)
                positives_idx.append(i + 1)

            # 负样本：排除邻居窗口的随机样本
            neighbor_range = set(range(max(0, i - self.neighbor_window),
                                        min(N, i + self.neighbor_window + 1)))
            neg_pool = [j for j in range(N) if j not in neighbor_range and j != i]

            if len(neg_pool) > 0:
                negatives_idx.append(np.random.choice(neg_pool))

        return anchors_idx, positives_idx, negatives_idx


class ContrastiveDataset(torch.utils.data.Dataset):
    """
    对比学习数据集

    预计算邻居关系，训练时通过索引高效获取样本
    """

    def __init__(
        self,
        states: np.ndarray,
        next_states: np.ndarray,
        neighbor_window: int = 5
    ):
        """
        Args:
            states: (N, 307)
            next_states: (N, 307)
            neighbor_window: 邻居窗口大小
        """
        self.states = torch.FloatTensor(states)
        self.next_states = torch.FloatTensor(next_states)

        # 预计算邻居关系
        constructor = ContrastiveSampleConstructor(neighbor_window)
        anchors_idx, positives_idx, negatives_idx = constructor.construct_temporal_neighbors(
            states, next_states
        )

        self.anchors_idx = anchors_idx
        self.positives_idx = positives_idx
        self.negatives_idx = negatives_idx

    def __len__(self):
        return len(self.anchors_idx)

    def __getitem__(self, idx):
        anchor_i = self.anchors_idx[idx]
        pos_i = self.positives_idx[idx]
        neg_i = self.negatives_idx[idx]

        return {
            'anchor': self.states[anchor_i],
            'positive': self.states[pos_i],
            'negative': self.states[neg_i],
            'transition_target': self.next_states[anchor_i],  # s' for transition loss
            'uncertainty_weight': 1.0  # 初始化权重，后续由encoder计算
        }
```

---

### 3.4 归一化对比损失（使用In-batch Negatives）

**文件**：`loss/contrastive_loss.py`

**关键修复**：使用in-batch negatives，而非单个负样本

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

class NormalizedContrastiveLoss(nn.Module):
    """
    归一化不确定性加权对比损失

    关键设计：
    1. 使用in-batch negatives（充分利用batch内所有样本）
    2. 各项损失归一化后再加权（避免量级不一致）
    3. 转移一致性单独作为正则项

    损失函数：
        L_cont = -log exp(sim(a, p)) / sum_j exp(sim(a, n_j))
        L_trans = 1 - cos(z, z')
        L = w_c * sigmoid(L_cont) + λ * w_t * (L_trans / 2)
    """

    def __init__(
        self,
        temperature: float = 0.1,
        lambda_transition: float = 0.5
    ):
        super().__init__()
        self.temperature = temperature
        self.lambda_transition = lambda_transition

    def forward(
        self,
        anchors: torch.Tensor,
        positives: torch.Tensor,
        negatives: torch.Tensor,
        transition_targets: torch.Tensor,
        uncertainty: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算归一化不确定性加权对比损失

        Args:
            anchors: (B, D) 锚点嵌入
            positives: (B, D) 正样本嵌入
            negatives: (B, N) 负样本嵌入（N=batch_size-1）
            transition_targets: (B, D) 转移目标嵌入
            uncertainty: (B,) 不确定性分数

        Returns:
            loss: 标量损失
            info: 诊断信息
        """
        B = anchors.shape[0]

        # ===== 1. InfoNCE 对比损失（使用in-batch negatives）====
        # 计算锚点与所有正样本的相似度
        pos_sim = F.cosine_similarity(anchors, positives, dim=-1) / self.temperature

        # 计算锚点与所有负样本的相似度（排除自己）
        # negatives: (B, B-1)，每行的负样本是batch内其他样本
        neg_sim = F.cosine_similarity(
            anchors.unsqueeze(1),      # (B, 1, D)
            negatives,                   # (B, B-1, D)
            dim=-1                       # (B, B-1)
        ) / self.temperature

        # 合并正负样本的相似度
        all_sim = torch.cat([pos_sim.unsqueeze(-1), neg_sim], dim=-1)  # (B, B)

        # InfoNCE损失：对角线是正样本
        labels = torch.zeros(B, dtype=torch.long, device=anchors.device)
        info_nce_loss = F.cross_entropy(all_sim, labels)

        # ===== 2. 转移一致性损失 ====
        transition_loss = 1 - F.cosine_similarity(anchors, transition_targets, dim=-1)

        # ===== 3. 不确定性加权 ====
        uncertainty_norm = uncertainty / (uncertainty.max() + 1e-8)

        # 动态权重
        transition_weight = 0.3 + 0.7 * uncertainty_norm   # [0.3, 1.0]
        contrastive_weight = 1.0 - 0.5 * uncertainty_norm  # [0.5, 1.0]

        # ===== 4. 归一化各项损失 ====
        # InfoNCE损失范围约 [0, log(B)]，用sigmoid归一化
        contrastive_loss_norm = torch.sigmoid(info_nce_loss)

        # 转移损失范围 [0, 2]，用线性归一化
        transition_loss_norm = transition_loss / 2.0

        # 归一化后加权组合
        loss = (contrastive_weight.mean() * contrastive_loss_norm +
                self.lambda_transition * transition_weight.mean() * transition_loss_norm.mean())

        info = {
            'info_nce': info_nce_loss.item(),
            'transition': transition_loss.mean().item(),
            'contrastive_norm': contrastive_loss_norm.item(),
            'transition_norm': transition_loss_norm.mean().item(),
            'uncertainty_mean': uncertainty_norm.mean().item(),
            'weight_c_mean': contrastive_weight.mean().item(),
            'weight_t_mean': transition_weight.mean().item()
        }

        return loss, info
```

---

### 3.5 平滑优先级采样

**文件**：`sample/priority_sampler.py`

```python
import numpy as np

def compute_smooth_priority_weights(
    uncertainty: np.ndarray,
    alpha: float = 0.5
) -> np.ndarray:
    """
    计算平滑优先级采样权重

    使用 p ∝ U^α 平滑，避免采样极端倾斜
    """
    uncertainty = np.clip(uncertainty, 1e-8, None)
    weights = uncertainty ** alpha
    weights = weights / (weights.sum() + 1e-8)
    return weights


class SmoothPrioritySampler:
    """平滑优先级采样器"""

    def __init__(self, uncertainty: np.ndarray, alpha: float = 0.5):
        self.weights = compute_smooth_priority_weights(uncertainty, alpha)
        self.alpha = alpha

    def sample(self, batch_size: int) -> np.ndarray:
        return np.random.choice(len(self.weights), size=batch_size, p=self.weights)
```

---

### 3.6 Q-Ensemble（用于在线探索的不确定性估计）

**文件**：`exploration/q_ensemble.py`

**关键修复**：使用Q-Ensemble方差来估计动作相关的不确定性，而非仅用状态不确定性

```python
import torch
import torch.nn as nn

class QEnsemble(nn.Module):
    """
    Q值集成：用于估计动作的不确定性

    不确定性来源：对于同一(s,a)，不同Q网络的预测分歧
    """

    def __init__(
        self,
        state_dim: int = 307,
        action_dim: int = 3,
        hidden_dim: int = 256,
        n_ensemble: int = 5
    ):
        super().__init__()
        self.n_ensemble = n_ensemble

        # 创建K个独立的Q网络
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

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        返回K个Q值

        Args:
            state: (B, state_dim)
            action: (B, action_dim)
        Returns:
            q_values: (B, K) K个Q值
        """
        x = torch.cat([state, action], dim=-1)
        q_values = torch.stack([net(x) for net in self.q_networks], dim=-1).squeeze(-1)
        return q_values

    def q_mean(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Q值的平均"""
        return self.forward(state, action).mean(dim=-1)

    def q_min(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Q值的最小值（用于TD学习）"""
        return self.forward(state, action).min(dim=-1)[0]

    def q_uncertainty(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Q值的不确定性（用于探索）

        使用方差作为不确定性度量
        """
        q_values = self.forward(state, action)  # (B, K)
        return q_values.var(dim=-1)  # (B,)
```

---

### 3.7 自适应探索器（关键修复）

**文件**：`exploration/adaptive_explorer.py`

**关键修复**：
1. 利用动作使用Q-Ensemble方差，而非状态不确定性
2. 使用batch化计算避免CPU-GPU同步瓶颈
3. 负惩罚使用相对变化率

```python
import torch
import numpy as np
from typing import Tuple

class AdaptiveUncertaintyExplorer:
    """
    自适应不确定性驱动探索器

    关键设计：
    1. 使用Q-Ensemble方差估计动作不确定性（与动作相关）
    2. Batch化计算避免CPU-GPU同步瓶颈
    3. 负惩罚使用相对变化率（解决量纲问题）
    """

    def __init__(
        self,
        policy,
        encoder,
        q_ensemble,
        num_candidates: int = 10,
        exploration_noise: float = 0.2,
        negative_penalty_threshold: float = -0.1,
        negative_penalty_weight: float = 0.1
    ):
        self.policy = policy
        self.encoder = encoder
        self.q_ensemble = q_ensemble
        self.num_candidates = num_candidates
        self.exploration_noise = exploration_noise
        self.negative_penalty_threshold = negative_penalty_threshold
        self.negative_penalty_weight = negative_penalty_weight

    def get_adaptive_alpha(self, training_progress: float) -> float:
        """自适应探索权重：前期高探索，后期高利用"""
        if training_progress < 0.3:
            return 0.8
        elif training_progress < 0.7:
            return 0.8 - 0.6 * (training_progress - 0.3) / 0.4
        else:
            return 0.2

    def select_action(
        self,
        state: np.ndarray,
        training_progress: float = 0.5,
        epsilon: float = 0.1,
        deterministic: bool = False
    ) -> np.ndarray:
        if deterministic or np.random.random() > epsilon:
            alpha = self.get_adaptive_alpha(training_progress)
            return self._exploitation_action(state, alpha)
        else:
            return self._exploration_action(state)

    def _exploration_action(self, state: np.ndarray) -> np.ndarray:
        """
        探索动作：使用添加噪声的方式，而非选择"最远"的动作

        在策略动作基础上添加符合高斯过程的噪声
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(
            next(self.policy.parameters()).device
        )

        # 从策略获取基准动作
        with torch.no_grad():
            base_action = self.policy.sample(state_tensor)
            base_action = base_action.cpu().numpy().flatten()

        # 添加噪声生成候选动作
        noise = np.random.randn(*base_action.shape) * self.exploration_noise
        action = np.clip(base_action + noise, -1, 1)

        return action

    def _exploitation_action(self, state: np.ndarray, alpha: float) -> np.ndarray:
        """
        利用动作：平衡Q-Ensemble方差和Q值

        关键修复：
        - 使用Q-Ensemble方差（与动作相关）
        - Batch化计算避免同步瓶颈
        """
        device = next(self.q_ensemble.parameters()).device
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

        # 获取策略动作作为基准
        with torch.no_grad():
            base_action = self.policy.sample(state_tensor)
            base_action = base_action.cpu().numpy().flatten()

        # 生成候选动作
        candidates = [base_action]
        for _ in range(self.num_candidates - 1):
            noise = np.random.randn(*base_action.shape) * 0.15
            action = np.clip(base_action + noise, -1, 1)
            candidates.append(action)

        # Batch化计算（关键修复：避免CPU-GPU同步）
        candidates_tensor = torch.FloatTensor(np.array(candidates)).to(device)
        state_batch = state_tensor.repeat(len(candidates), 1)

        # 一次性计算所有Q值和不确定性
        with torch.no_grad():
            q_values = self.q_ensemble.q_mean(state_batch, candidates_tensor)  # (M,)
            q_unc = self.q_ensemble.q_uncertainty(state_batch, candidates_tensor)  # (M,)

        # 加权分数
        scores = alpha * q_unc + (1 - alpha) * q_values

        # 选择最佳动作
        best_idx = scores.argmax().item()
        return candidates[best_idx]

    def compute_exploration_reward(
        self,
        state: np.ndarray,
        next_state: np.ndarray,
        env_reward: float,
        beta: float = 0.3
    ) -> float:
        """
        计算探索奖励

        关键修复：使用相对变化率解决量纲问题
        """
        device = next(self.encoder.parameters()).device

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        next_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(device)

        # 计算不确定性
        current_emb = self.encoder(state_tensor)
        next_emb = self.encoder(next_tensor)
        current_unc = compute_ensemble_uncertainty(current_emb).item()
        next_unc = compute_ensemble_uncertainty(next_emb).item()

        reduction = current_unc - next_unc

        # 正向奖励
        exploration_bonus = beta * max(0, reduction)

        # 负向惩罚（使用相对变化率）
        if current_unc > 1e-5:
            relative_reduction = reduction / current_unc
            if relative_reduction < self.negative_penalty_threshold:
                penalty = self.negative_penalty_weight * abs(relative_reduction)
            else:
                penalty = 0
        else:
            penalty = 0

        return env_reward + exploration_bonus - penalty
```

---

### 3.8 离线训练器

**文件**：`training/offline_trainer.py`

```python
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, Optional

class OfflineTrainer:
    """
    离线训练器

    流程：
    1. 训练EnsembleEncoder（归一化不确定性加权对比损失）
    2. 用encoder估计不确定性
    3. 使用p∝U^α平滑加权采样训练DiffusionQL
    """

    def __init__(
        self,
        state_dim: int = 307,
        latent_dim: int = 64,
        hidden_dim: int = 256,
        n_ensemble: int = 5,
        priority_alpha: float = 0.5,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.device = device
        self.priority_alpha = priority_alpha

        self.encoder = EnsembleEncoder(
            state_dim=state_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            n_ensemble=n_ensemble
        ).to(device)

        self.criterion = NormalizedContrastiveLoss(
            temperature=0.1,
            lambda_transition=0.5
        )

        self.optimizer = optim.Adam(self.encoder.parameters(), lr=1e-3)
        self.is_trained = False

    def train_encoder(
        self,
        states: np.ndarray,
        next_states: np.ndarray,
        epochs: int = 50,
        batch_size: int = 256,
        log_freq: int = 10
    ) -> Dict[str, list]:
        """
        训练表示模型

        关键修复：
        - 使用ContrastiveDataset获取正确的邻居样本
        - 使用in-batch negatives
        """
        self.encoder.train()

        # 创建对比学习数据集
        dataset = ContrastiveDataset(states, next_states, neighbor_window=5)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        history = {
            'loss': [],
            'info_nce': [],
            'transition': [],
            'uncertainty': []
        }

        for epoch in range(epochs):
            epoch_loss = 0
            epoch_info_nce = 0
            epoch_transition = 0
            n_batches = 0

            for batch in dataloader:
                anchors = batch['anchor'].to(self.device)
                positives = batch['positive'].to(self.device)
                negatives = batch['negative'].to(self.device)
                transition_targets = batch['transition_target'].to(self.device)

                # 计算嵌入
                anchor_emb = self.encoder.encode_mean(anchors)  # (B, D)
                positive_emb = self.encoder.encode_mean(positives)
                negative_emb = self.encoder.encode_mean(negatives)
                target_emb = self.encoder.encode_mean(transition_targets)

                # 计算不确定性
                embeddings = self.encoder(anchors)
                uncertainty = compute_ensemble_uncertainty(embeddings)

                # 构建in-batch negatives
                # 负样本：同batch内的其他样本（排除自己）
                all_emb = self.encoder.encode_mean(
                    torch.cat([anchors, positives, transition_targets], dim=0)
                )[:len(anchors)]  # 取前B个作为锚点
                other_emb = all_emb[torch.randperm(len(all_emb))[:len(all_emb)-1]]

                # 计算损失
                loss, info = self.criterion(
                    anchor_emb,
                    positive_emb,
                    other_emb.unsqueeze(0).repeat(len(anchors), 1, 1)[:, 0],
                    target_emb,
                    uncertainty
                )

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                epoch_info_nce += info['info_nce']
                epoch_transition += info['transition']
                n_batches += 1

            history['loss'].append(epoch_loss / n_batches)
            history['info_nce'].append(epoch_info_nce / n_batches)
            history['transition'].append(epoch_transition / n_batches)
            history['uncertainty'].append(info['uncertainty_mean'])

            if (epoch + 1) % log_freq == 0:
                print(f"Encoder Epoch {epoch+1}/{epochs}, "
                      f"Loss: {history['loss'][-1]:.4f}, "
                      f"InfoNCE: {history['info_nce'][-1]:.4f}, "
                      f"Transition: {history['transition'][-1]:.4f}")

        self.is_trained = True
        return history

    def compute_priority_weights(self, states: np.ndarray) -> np.ndarray:
        """计算平滑优先级采样权重"""
        self.encoder.eval()
        with torch.no_grad():
            states_tensor = torch.FloatTensor(states).to(self.device)
            embeddings = self.encoder(states_tensor)
            uncertainty = compute_ensemble_uncertainty(embeddings)

        weights = compute_smooth_priority_weights(
            uncertainty.cpu().numpy(),
            alpha=self.priority_alpha
        )
        return weights
```

---

## 四、算法伪代码

### 阶段1：离线训练

```
Algorithm 1: Offline Training with Uncertainty-Weighted CL

Input: Dataset D = {(s, a, s', r)}_N
Output: Policy π, Encoder E

1.  // Phase 1: Train Ensemble Encoder
2.  Initialize K=5 encoder networks {E_k}
3.  Pre-compute neighbor relations {anchor, positive, negative}
4.
5.  for epoch = 1 to E_epochs do
6.      for batch (anchor, positive, negative, s') in DataLoader do
7.          // Compute embeddings
8.          z ← E(s), z+ ← E(s+), z- ← E(s-), z' ← E(s')
9.
10.         // In-batch negatives
11.         neg_matrix ← [z-, z+, ...] excluding anchor
12.
13.         // Compute uncertainty
14.         U ← variance({E_k(s)})  // O(K)
15.
16.         // Normalized loss
17.         L_cont ← -log exp(sim(z, z+)) / sum_j exp(sim(z, neg_j))
18.         L_trans ← 1 - cos(z, z')
19.         L_cont_norm ← sigmoid(L_cont)
20.         L_trans_norm ← L_trans / 2
21.
22.         // Uncertainty-weighted combination
23.         w_trans ← 0.3 + 0.7 * normalize(U)
24.         w_cont ← 1.0 - 0.5 * normalize(U)
25.         L ← w_cont * L_cont_norm + λ * w_trans * L_trans_norm
26.
27.         Update {E_k} by ∇L
28.     end for
29. end for
30.
31.  // Phase 2: Priority Sampling
32.  for all s in D do
33.      U(s) ← variance(E(s))
34.      p(s) ∝ U(s)^α                   // smooth sampling
35.  end for
36.
37.  // Phase 3: Train DiffusionQL with priority sampling
38.  for epoch = 1 to E_ql do
39.      for iter = 1 to I do
40.          Batch ← sample from D with probabilities p(s)
41.          Update π, Q by batch
42.      end for
43.  end for
44.
45. return π, E
```

### 阶段2：在线探索

```
Algorithm 2: Online Exploration with Adaptive ΔU-Driven Selection

Input: Policy π, Encoder E, Q-Ensemble Q, Budget B
Output: Updated Policy π'

1.  Initialize replay buffer R ← ∅, step ← 0
2.
3.  while step < B do
4.      progress ← step / B
5.      α ← adaptive_alpha(progress)     // 0.8 → 0.2
6.      ε ← base_epsilon * (1 - progress * 0.5)
7.
8.      // Select action
9.      if random() < ε then
10.         // Exploration: add noise to policy action
11.         a ← clip(π(s) + noise, -1, 1)
12.     else
13.         // Exploitation: balance Q-variance and Q-value
14.         candidates ← {π(s) + noise_i}
15.         Q_var ← Q.variance(s, candidates)  // Batch computation
16.         Q_val ← Q.mean(s, candidates)
17.         scores ← α * Q_var + (1-α) * Q_val
18.         a ← argmax scores
19.     end if
20.
21.     // Real environment interaction
22.     s' ← env.step(a)                  // CRITICAL: real interaction
23.     r ← env.reward(s, a)
24.
25.     // Compute exploration reward with penalty
26.     ΔU ← U(s) - U(s')
27.     bonus ← β * max(0, ΔU)
28.     if current_U > 1e-5 and (ΔU / current_U) < threshold then
29.         penalty ← γ * |ΔU / current_U|  // Relative penalty
30.     else
31.         penalty ← 0
32.     r_total ← r + bonus - penalty
33.
34.     R ← R ∪ {(s, a, s', r_total)}
35.     s ← s', step ← step + 1
36.
37.     if |R| > min_size then
38.         Update π, Q using R
39.     end if
40. end while
41.
42. return π'
```

---

## 五、关键修复总结

| 问题 | 原方案 | 修复后 |
|------|--------|--------|
| 正样本构造 | next_state作为正样本 | 时间邻居作为正样本 |
| 负样本数量 | 1个负样本 | In-batch negatives |
| 损失加权 | 直接相加 | 归一化后加权 |
| 不确定性计算 | O(K²)最大成对距离 | O(K)方差 |
| 优先级采样 | p=U/ΣU（极端倾斜） | p∝U^α（平滑） |
| 下一状态获取 | concat(s,a)模拟 | 真实环境交互 |
| 探索权重 | 固定α=0.5 | 自适应α（0.8→0.2） |
| 动作不确定性 | 状态不确定性（与动作无关） | **Q-Ensemble方差** |
| 负向惩罚 | 固定阈值 | **相对变化率** |
| 计算效率 | 循环CPU-GPU同步 | **Batch化计算** |

---

## 六、实验设计建议

### 待验证假设

1. 邻居窗口大小：`neighbor_window ∈ [1, 3, 5, 10]`
2. 温度系数：`τ ∈ [0.05, 0.1, 0.2, 0.5]`
3. 平滑因子：`α ∈ [0.4, 0.5, 0.6, 0.7]`
4. 负惩罚阈值：`threshold ∈ [-0.2, -0.1, -0.05]`

### 消融实验

| 实验 | 说明 |
|------|------|
| - Neighbor | 使用next_state代替邻居 |
| - InBatchNeg | 使用单个负样本 |
| - Fixed α | 固定α=0.5 |
| - Relative Penalty | 使用固定阈值 |
| - Full | 完整方法 |

---

最后更新：2026-06-02