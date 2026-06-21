# 两阶段强化学习研究总纲与实施路线

> 目标：为当前 `EasyCarla-RL` 项目整理一份长期可沿用的执行总纲，明确研究主线、实现顺序、代码模块、实验路线与阶段性交付物。

---

## 一、总目标

当前项目最适合的研究路线已经比较明确：

- **阶段1：离线强化学习**
- **阶段2：在线强化学习**

整体目标不是推翻现有 `Diffusion QL` 框架，而是在现有工程基础上逐步补全两类能力：

1. 在离线阶段，通过状态表征驱动的数据利用方式提升初始策略质量
2. 在在线阶段，通过不确定性引导的主动采样与高效微调提升最终策略性能

最终形成一条从离线强化学习自然过渡到在线强化学习的完整研究路线。

---

## 二、研究主线

### 2.1 离线阶段

离线阶段的核心问题是：

**如何更高效地利用已有离线数据。**

核心做法是：

- 先用离线数据做对比表示学习，得到一个结构化状态空间
- 用该预训练 encoder 初始化两条离线支路：`E_rl` 与 `E_prio`
- `E_rl` 提供给 actor / critic 使用，并在离线 RL 中继续微调
- `E_prio` 始终冻结，作为稳定参考空间，用于样本重要性评估与优先级采样
- 在不重写 `Diffusion QL` 主体的前提下，用“更优表征 + 更优样本调度”提升数据利用效率

这里的方法定位需要明确：

- 本方案不是声称对比学习得到的表征一定等价于最优控制表征
- 本方案也不是声称潜空间中的稀疏样本一定就是高价值样本
- 更准确的理解是：先用对比学习提供一个稳定、结构化的初始化空间，再让 `E_rl` 朝任务相关方向适配，同时保留冻结的 `E_prio` 作为样本重加权的稳定参考系
- 因此，该方法更适合作为一种**兼顾可实现性、稳定性与通用性的离线增强框架**，而不是一个依赖强理论最优性的方案

这里明确不做：

- 强耦合联合训练
- 重写 `Diffusion QL` 主体
- 针对当前某一批数据的特化设计
- 随着 `E_rl` 漂移而高成本重算整库 priority

### 2.2 在线阶段

在线阶段的核心问题是：

**如何在已有离线策略基础上，用尽量少的 CARLA 交互继续提升策略。**

核心做法是：

- 复用离线阶段训练好的 `E_rl` 作为在线阶段的初始控制表征
- 在统一表征空间上增加多个轻量 Q heads
- 用多头分歧估计不确定性
- 用不确定性与新颖性判断哪些在线数据更值得采集和重点训练

---

## 三、为什么在线阶段继续复用 `Diffusion QL`

在线阶段我建议**继续复用 `Diffusion QL` 作为主干**，而不是中途切换到另一套在线 RL 算法。

原因有三点：

1. **研究主线更完整**  
   离线阶段和在线阶段都围绕同一个 backbone 展开，更容易形成一条清楚的 offline-to-online 叙事。

2. **工程衔接更自然**  
   可以直接继承离线训练好的 actor、critic 与表征模块，不需要把离线策略迁移到另一类算法。

3. **创新点更集中**  
   研究重点放在“数据怎么用、数据怎么采”，而不是“中途换一个 RL 主干”。

因此建议这样分工：

- **主方法**：继续使用 `Diffusion QL`
- **对比 baseline**：可以额外接入 `SAC`、`TD3` 等方法

也就是说，在线阶段不是推翻原方法，而是：

- 保留 `Diffusion QL` 主体
- 加入 mixed replay
- 加入 uncertainty-guided exploration
- 加入 novelty-guided sample selection

---

## 四、推荐的实现路线

### 4.1 先完成离线 MVP

#### 目标

- 跑通对比表示学习
- 跑通双编码器离线训练流程
- 跑通冻结参考空间下的离线优先级采样
- 训练出比原始 `Diffusion QL` 更强的离线初始策略

#### 要完成的事情

1. 增加 `encoder` 模块
2. 增加对比学习训练脚本
3. 增加表征空间样本打分逻辑
4. 增加优先级采样器或重加权逻辑
5. 将预训练 encoder 拆分为可微调的 `E_rl` 与冻结的 `E_prio`
6. 让 `Diffusion QL` 在 raw state 上按 batch 编码训练
7. 做离线 baseline 与消融实验，至少区分 `baseline / encoder_frozen / encoder_finetune / full`

#### 预期输出

- 一个训练好的对比学习初始化 encoder
- 一个离线 priority score 文件或缓存结果
- 一个带 RL 微调 encoder 的离线增强版 `Diffusion QL` 策略
- 一组能够拆分“表示学习收益 / encoder 微调收益 / priority 收益”的离线对比实验结果

### 4.2 再完成在线 MVP

#### 目标

- 构建在线主动采样与微调能力
- 让在线探索不是盲目的，而是有方向的

#### 要完成的事情

1. 复用离线阶段训练好的 `E_rl`
2. 增加多个轻量 Q heads
3. 用现有离线数据先把这些 heads 预热起来
4. 构建 online replay buffer
5. 构建 offline + online mixed replay 训练流程
6. 加入 uncertainty 引导探索
7. 加入 novelty 引导样本保留与优先训练
8. 完成少量在线微调实验

#### 预期输出

- 一个在线微调后的最终策略
- 一个可运行的主动采样机制
- 一组在线对比和消融结果

---

## 五、在线多头部分的最小可实现版本

为了保证实现难度可控，在线阶段建议先做一个最小可实现版本，而不是一开始就设计复杂 ensemble。

### 5.1 结构

核心结构为：

- 一个共享 encoder：`z = f(s)`
- 多个轻量 Q heads：`Q_1(z,a), Q_2(z,a), ..., Q_K(z,a)`

这里的 encoder 默认优先指离线阶段已经任务化微调过的 `E_rl`，因为在线阶段更关心控制适配能力，而不是继续沿用离线 priority 的冻结参考支路。

含义是：

- encoder 负责“怎么表示状态”
- 多个 Q heads 负责“怎么评估状态-动作值”

### 5.2 输入、输出与 loss

这一步直接使用现有离线数据中的 transition：

- `s`
- `a`
- `r`
- `s'`
- `done`

先通过 encoder 得到：

- `z = f(s)`
- `z' = f(s')`

然后每个 head 定义为：

- 输入：`(z, a)`
- 输出：标量 Q 值 `Q_k(z, a)`

这里的训练目标不是对比学习，而是**标准 TD 回归**。

第一版实现建议：

- 每个 head 对自己的 TD target 做 `MSE loss` 或 `Huber loss`
- 总损失取多个 head loss 的平均

### 5.3 为什么同一份离线数据可以训练多个 head

因为多个 head 不需要不同来源的数据，只需要对同一份 replay 数据形成不同判断。

建议通过以下方式制造差异：

- 不同随机初始化
- bootstrap 子集采样
- 每个 batch 使用不同样本 mask

这样做以后：

- 对熟悉区域，多个 head 往往更一致
- 对少见区域、分布外区域或尚未学稳的区域，多个 head 往往分歧更大

这些分歧就可以作为 uncertainty。

### 5.4 为什么这部分相对好实现

因为它不需要：

- 新的数据格式
- 新的环境接口
- 全新的在线 RL 主体

它本质上只是：

- 复用已有 replay 数据
- 复用已有 encoder 输出
- 在现有 critic 思路上扩成多个 head

因此这部分的主要新增复杂度只有两点：

1. 如何让不同 head 保持差异
2. 如何把多头分歧转成 uncertainty 指标

相比重做一整套在线算法，这条路线更直观，也更适合当前项目。

---

## 六、建议的代码模块拆分

### 6.1 离线阶段建议新增模块

- `representation/encoder.py`
  - 单表征网络定义
- `representation/contrastive_dataset.py`
  - 构造对比学习样本
- `representation/contrastive_trainer.py`
  - 表征预训练逻辑
- `representation/scoring.py`
  - 表征空间样本打分
- `representation/priority_sampler.py`
  - 优先级采样器
- `example/train_representation.py`
  - 表征训练入口
- `example/train_diffusion_ql_priority.py`
  - 接入 priority 的离线训练入口

### 6.2 在线阶段建议新增模块

- `online_rl/uncertainty_heads.py`
  - 多头价值网络
- `online_rl/online_buffer.py`
  - 在线 replay buffer
- `online_rl/active_selector.py`
  - 根据 uncertainty / novelty 选择动作或筛选样本
- `online_rl/mixed_trainer.py`
  - offline + online 混合训练逻辑
- `example/train_online_finetune.py`
  - 在线微调入口

### 6.3 实验与日志模块

- `example/eval_policy.py`
  - 统一评测脚本
- `example/run_ablation.py`
  - 消融实验入口
- `results/` 或 `logs/`
  - 保存训练指标和评测结果

---

## 七、分阶段实施计划

### 阶段A：基础梳理

#### 目标

确认所有现有训练与数据链路都能复用。

#### 内容

- 梳理离线训练入口
- 梳理状态维度与数据格式
- 梳理现有 CARLA 数据采集流程
- 梳理当前在线回放脚本

#### 完成标志

- 明确需要改动的文件
- 明确新增模块边界
- 明确离线与在线的输入输出格式

### 阶段B：离线表征

#### 目标

训练出稳定可用的单表征网络。

#### 内容

- 构建对比学习数据集
- 训练 encoder
- 导出每条状态的 embedding
- 验证表征空间是否可用于邻域判断

#### 完成标志

- 表征训练脚本能独立运行
- encoder checkpoint 可保存与加载
- 能成功导出 embedding 或 score

### 阶段C：离线增强训练

#### 目标

让表征真正服务于 `Diffusion QL` 的数据利用。

#### 内容

- 构建基于表征的 priority score
- 接入离线采样逻辑
- 训练 priority 版本 `Diffusion QL`
- 做与原始 `Diffusion QL` 的离线对比

#### 完成标志

- 优先级采样训练流程能稳定运行
- 与 baseline 相比有清楚的性能变化
- 能完成至少一组基础消融

### 阶段D：在线多头初始化

#### 目标

在离线数据上把在线阶段需要的多头结构预热出来。

#### 内容

- 复用离线 encoder
- 增加多个轻量 Q heads
- 用离线数据预训练这些 heads
- 加入 bootstrap / mask 机制让各 head 保持差异

#### 完成标志

- 多个 head 能正常输出
- 不同 head 对同一样本有可观察分歧
- uncertainty 计算逻辑可用

### 阶段E：在线主动采样

#### 目标

让策略在 CARLA 中进行有方向的探索和数据收集。

#### 内容

- 在线交互并持续采样
- 用 uncertainty 引导探索
- 用 novelty 判断新数据是否有补充价值
- 构建 online buffer 与 mixed training

#### 完成标志

- 在线微调脚本可运行
- 新数据能够回灌训练
- 最终策略能够较离线初始策略继续提升

### 阶段F：实验与论文支撑

#### 目标

完成对比实验和核心消融，为后续论文写作准备证据。

#### 内容

- 离线 baseline 对比
- 在线 baseline 对比
- 各子模块消融
- 训练曲线、成功率、奖励、成本等指标整理

#### 完成标志

- 有完整实验记录
- 有可重复执行的评测脚本
- 有能支撑论文主线的结果图表

---

## 八、预计工期

这里给的是代码实现层面的保守估计，不包含大规模反复跑 CARLA 实验的时间。

### 8.1 离线 MVP

预计：**1~2 天**

内容包括：

- encoder
- 对比学习训练入口
- 表征评分
- priority 采样
- 接入 `Diffusion QL`

### 8.2 在线 MVP

预计：**2~4 天**

内容包括：

- uncertainty heads
- online buffer
- mixed replay
- uncertainty / novelty 逻辑
- 在线微调入口

### 8.3 实验与评测脚本

预计：**1~2 天**

内容包括：

- baseline 对比
- 消融脚本
- 统一评测与日志保存

### 8.4 总体判断

如果目标是先做出第一版完整研究原型，那么：

**一周左右做出首版闭环是现实的。**

真正更耗时的通常不是写代码，而是：

- CARLA 环境运行
- 训练耗时
- 多轮实验复现

---

## 九、实验路线建议

### 9.1 主实验

优先完成：

- 原始 `Diffusion QL`
- 加入离线表征优先级后的 `Diffusion QL`
- 加入在线主动采样后的完整方法

### 9.2 离线阶段消融

建议至少做：

- 不加 encoder
- 有 encoder 但不用 priority
- 有 encoder + priority

### 9.3 在线阶段消融

建议至少做：

- 不加 uncertainty
- 只加 uncertainty
- 只加 novelty
- uncertainty + novelty 同时使用

### 9.4 外部方法对比

如果时间允许，可以加入：

- SB3 的 `SAC`
- SB3 的 `TD3`

但需要明确：

- SB3 更适合做在线 RL 对比
- 不适合直接作为当前主方法的实现库
- 离线主 baseline 仍优先使用现有 `Diffusion QL`

---

## 十、实现原则与推进顺序

### 10.1 实现原则

后续代码推进时，应始终坚持：

1. **先稳后扩**
2. **尽量复用现有代码**
3. **不做强耦合联合训练**
4. **通用性优先**
5. **实验范围先收敛**

### 10.2 实际推进顺序

如果从下一步就开始写代码，建议固定顺序为：

1. 先做离线表征训练脚本
2. 再做基于表征的 priority score 计算
3. 再把 priority 采样接入离线 `Diffusion QL`
4. 跑离线 baseline 和离线消融
5. 再做多头 uncertainty heads 与离线预热
6. 再做在线 buffer、主动采样、mixed training
7. 最后补全在线实验与对比

---

## 十一、最后总结

> 在现有 `EasyCarla-RL` 项目基础上，先通过单表征驱动的离线数据利用提升初始策略，再继续复用 `Diffusion QL` 主干，通过多头不确定性估计和主动在线采样提升微调效率，以“先离线、再在线、先稳后扩”的顺序逐步完成整个研究路线。

---

最后更新：2026-06-12
