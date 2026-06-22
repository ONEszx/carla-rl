# 国内外研究现状

## 1. 研究背景

自动驾驶智能决策是智能网联汽车领域的核心问题之一，其目标是在动态开放交通环境中，基于多源感知信息完成场景理解、行为选择与连续控制。围绕这一问题，国内外研究总体上经历了由传统多段式系统向端到端学习方法，再向融合大模型与多模态推理能力的智能决策框架演进的过程。传统多段式方案强调系统分层与工程可控性，端到端方法强调从数据中直接学习决策策略，而近年来出现的视觉—语言—动作模型则进一步强化了复杂场景中的语义理解与推理能力[1-2]。

从研究发展脉络看，三类方法分别对应自动驾驶决策系统的三个不同技术重心。传统模块化路线追求感知、预测、规划与控制的解耦优化，具有较高的可解释性和工程成熟度；强化学习驱动的端到端方法试图以统一目标函数实现整体优化，突出策略学习与闭环适应能力；VLA 及相关多模态大模型则更加关注复杂场景中的高层语义建模、决策解释与长时序推理[2],[7],[13]。因此，对上述三类研究路线进行系统梳理，有助于明确本课题在自动驾驶智能决策方向上的研究定位。

## 2. 传统多段式自动驾驶研究现状

传统自动驾驶系统通常采用“感知—预测—规划—控制”的分层架构，即先通过传感器与感知算法获取道路、车辆与障碍物信息，再进行轨迹预测、行为决策与运动规划，最后由控制模块输出底层控制指令。这一路线长期以来一直是工业界自动驾驶系统的主流实现方式，其优势在于模块职责清晰、系统可解释性较强、便于功能验证与安全评估[2]。

在国外研究与产业实践中，Waymo Driver 是典型代表。Waymo 团队围绕自动驾驶系统的感知融合、行为预测、运动规划与系统级安全评估构建了较完整的技术体系，并系统阐述了其安全方法学与 readiness determination 流程，反映出传统模块化路线在复杂真实交通环境中的工程可落地性[1-2]。除工业闭源系统外，开源自动驾驶软件栈 Autoware 也是该方向的重要代表。Autoware Foundation 推动形成了较为标准化的自动驾驶软件框架，而 University of Pennsylvania 团队基于 Autoware.Auto 参加 Japan Automotive AI Challenge 的工作，则进一步说明该平台在原型验证、系统集成和研究复现方面具有较强支撑能力[5-6]。

在国内，百度 Apollo 是最具代表性的模块化自动驾驶开放平台之一。Apollo 平台围绕定位、感知、预测、规划与控制建立了较系统的功能体系，在产业界和学术界均具有广泛影响[3]。其中，Haoyang Fan、Fan Zhu、Qi Kong 等依托百度 Apollo 提出的 EM Motion Planner，较为集中地体现了传统分层规划方法在工程实现中的典型思路，即通过分阶段求解与约束处理提高规划效率和可行性[4]。总体而言，Apollo 所代表的研究路线强调系统工程能力与模块协同优化，对国内自动驾驶平台化研究起到了重要推动作用。

总体来看，传统多段式方案在工程成熟度、可解释性和安全验证机制方面仍具有明显优势，但也存在较为突出的局限。一方面，各模块通常围绕局部目标独立设计与优化，系统整体难以实现真正意义上的全局最优；另一方面，感知误差、预测偏差与规划误差会沿链路逐层传递，导致系统在动态开放环境中的适应性受到限制。因此，传统多段式研究虽然为自动驾驶奠定了坚实的工程基础，但也促使学术界进一步探索更加统一的端到端决策方法[2],[4-6]。

## 3. 强化学习端到端自动驾驶研究现状

与传统分层系统不同，端到端自动驾驶旨在直接学习从环境状态到车辆控制动作的映射关系，以减少人工规则设计和模块间信息损失。在这一方向中，强化学习由于具备基于奖惩信号优化长期回报的能力，被广泛认为是实现自动驾驶智能决策的重要技术路径之一。尤其是在闭环控制问题中，强化学习相较纯监督学习更强调策略执行后的长期效果，因此在决策优化层面具有天然优势[7]。

早期强化学习自动驾驶研究多依赖在线交互式训练，但自动驾驶任务本身具有状态空间高维、试错成本高、安全约束强等特点，导致纯在线强化学习存在样本效率低、训练不稳定和现实部署困难等问题。为缓解上述矛盾，近年来研究重点逐渐转向离线强化学习、表征学习增强强化学习以及离线—在线结合的策略优化框架。Chen、Koltun 和 Krähenbühl 等来自 The University of Texas at Austin 与 Intel Labs 的研究者提出的《Learning to Drive from a World on Rails》，是端到端闭环驾驶研究中的代表性工作之一。该研究通过将数据驱动学习与可控环境约束结合，实现了从大规模驾驶数据中学习闭环驾驶能力，推动了端到端自动驾驶从开放环评估走向闭环验证[7]。

在离线强化学习方向，Zenan Li、Hang Zhao 等来自清华大学、上海启智研究院与 QCraft 的研究者提出了基于层次潜变量技能的离线强化学习方法，通过在策略学习中引入结构化潜在技能表示，提升了自动驾驶任务中的策略泛化能力与训练效率[8]。这类研究表明，单纯依赖行为克隆难以充分利用离线数据，而通过引入更强的状态表征与技能建模机制，有望在有限数据条件下获得更稳定的策略学习效果。

围绕闭环评测与仿真验证，Bench2Drive 的提出为端到端自动驾驶研究提供了更系统的多能力评估基准。该工作由上海交通大学相关团队提出，强调在更丰富、更具挑战性的交通任务中评估端到端自动驾驶系统的综合能力，为后续方法比较提供了统一标准[9]。在此基础上，Zhenjie Yang、Xiaosong Jia、Junchi Yan 等来自上海交通大学、复旦大学、Asynscale AI 和 AgiBot 的研究者提出 Raw2Drive，将世界模型与强化学习相结合，以提升 CARLA v2 场景下端到端驾驶的闭环性能[10]。随后，Yinfeng Gao、Dongbin Zhao 等来自北京科技大学、中国科学院自动化研究所和小米汽车的研究者提出 PerlAD，通过 pseudo-simulation 机制进一步增强离线数据在闭环强化学习中的利用效率，体现了当前研究正在尝试用更低的真实交互成本获得更高质量的策略提升[11]。此外，HKUST（Guangzhou）与 HKUST 的 Zewei Yang、Zengqi Peng、Jun Ma 等提出 SEG-Parking，将端到端离线强化学习应用于自动泊车任务，也说明离线强化学习方法正在逐步向更具体、更接近真实控制需求的自动驾驶子任务拓展[12]。

总体而言，强化学习端到端自动驾驶研究在推动自动驾驶决策由“规则设计”向“策略学习”转变方面具有重要意义，其主要优势在于能够通过统一目标函数进行整体优化，并具备一定的在线适应潜力。但现有研究仍面临若干关键问题：其一，离线数据分布与在线交互分布之间往往存在偏移，容易导致策略退化；其二，复杂场景下状态表征的稳定性、泛化性和可迁移性仍不足；其三，单一强化学习策略在复杂交通博弈与长时序决策中仍缺乏足够的深层推理能力。因此，如何将表征学习、离线强化学习与在线主动微调有效结合，仍是当前自动驾驶强化学习研究的核心问题之一[8-12]。

## 4. VLA驱动智能决策研究现状

随着视觉语言模型和大模型技术的发展，研究者开始尝试将语言语义建模、跨模态对齐与动作生成机制引入自动驾驶任务，进而形成视觉—语言—动作（Vision-Language-Action, VLA）驱动的智能决策研究方向。与传统感知或纯状态驱动方法相比，这一路线不再局限于“看见环境并做出动作”，而是进一步关注“理解场景—推理风险—生成决策”的全过程，因此在复杂场景理解、人机交互解释和高层决策表达等方面展现出较强潜力[13-17]。

在视觉语言推理方面，Chonghao Sima、Li Chen、Hongyang Li 等依托 OpenDriveLab、上海人工智能实验室、香港大学和德国图宾根大学提出 DriveLM，将驾驶问题建模为图结构视觉问答任务，使模型能够围绕交通参与者、道路结构和驾驶意图进行更具语义性的推理[13]。这一工作表明，自动驾驶决策问题可以从“端到端控制映射”进一步扩展为“面向驾驶任务的多模态认知推理”，为后续复杂决策建模提供了新的研究视角。

在大视觉语言模型与自动驾驶结合方面，Xiaoyu Tian、Hang Zhao 等来自清华大学与理想汽车的研究者提出 DriveVLM，系统探索了大视觉语言模型在自动驾驶场景理解、语义决策与任务统一建模中的潜力[14]。与之相近，Zhenhua Xu、Hengshuang Zhao 等来自香港大学、浙江大学、华为诺亚方舟实验室和悉尼大学的研究者提出 Driveclaude4，强调通过大语言模型增强端到端驾驶系统的可解释性与语义表达能力，从而提升决策过程的透明度[15]。这类研究虽然尚未完全进入高频实时闭环控制层，但在决策解释、驾驶意图表达和复杂场景理解方面已展现出重要价值。

在从“理解驾驶”走向“规划驾驶”的研究中，Chenbin Pan、Liu Ren 等来自 Syracuse University 与 Bosch Research North America 的研究者提出 VLP（Vision-Language Planning for Autonomous Driving），将视觉语言建模与自动驾驶规划问题相结合，推动多模态模型向轨迹规划和行为生成层延伸[16]。进一步地，Xingcheng Zhou、Alois Knoll 等来自 Technical University of Munich 和 Ludwig Maximilian University of Munich 的研究者提出 OpenDriveVLA，明确将自动驾驶建模为端到端视觉—语言—动作学习任务，尝试在统一模型中同时完成感知、语义理解与动作输出，代表了 VLA 路线向自动驾驶核心控制问题逼近的趋势[17]。

除上述直接面向 VLA 的工作外，Anthony Hu、Alex Kendall、Jamie Shotton 等 Wayve 研究者提出的 GAIA-1 虽然更偏向生成式世界模型，但其通过建模可生成的驾驶环境动态过程，为复杂场景中的长时序预测、行为模拟与决策支持提供了新的研究工具[18]。从某种意义上说，生成式世界模型与 VLA 路线在“提升驾驶系统对环境变化的理解和推理能力”这一目标上具有较强互补性。

总体来看，VLA 驱动的自动驾驶智能决策研究正在推动自动驾驶从“控制导向”进一步向“理解—推理—决策导向”发展。然而，该方向目前仍面临一些现实挑战。首先，多模态大模型虽然具备较强的语义理解与推理能力，但其推理延迟、计算开销和闭环稳定性尚难直接满足实时自动驾驶控制要求；其次，VLA 模型如何与连续动作控制和强化学习优化机制有效衔接，当前仍缺乏成熟范式；最后，大模型在车载部署、数据闭环更新与长期运行中的工程可用性仍需进一步验证。因此，如何将 VLA 用于复杂场景中的“慢思考”决策，并与高效率的强化学习快策略形成互补，已成为一个具有代表性的研究方向[14-18]。

## 5. 现有研究不足与本课题切入点

综合已有研究可以看出，传统多段式自动驾驶、强化学习端到端自动驾驶以及 VLA 驱动智能决策分别从工程可靠性、策略学习能力和复杂场景推理能力三个层面推动了自动驾驶技术的发展，但三者也分别存在明显局限。传统模块化方案虽然工程成熟度高，但难以避免多模块误差累积与全局协同不足；强化学习端到端方案虽能实现统一优化，但在样本效率、复杂场景泛化和在线稳定微调方面仍存在不足；VLA 与多模态大模型具备更强的语义理解与深层推理潜力，但目前尚难直接承担高实时性、高安全约束的闭环控制任务[4],[8],[14],[17]。

由此可见，现阶段尚缺乏一套能够同时兼顾高效学习、复杂推理与持续演化能力的统一自动驾驶智能决策框架。基于此，本课题拟以强化学习为主线，在第一阶段重点研究表征学习增强的离线—在线协同强化学习方法，以提升有限数据条件下的策略学习效率与闭环适应能力；在后续研究中进一步引入视觉—语言—动作模型，面向复杂交通场景构建具备深层推理能力的慢思考决策模块；最终结合快慢系统协同与反馈驱动优化机制，探索面向动态开放交通环境的端到端自动驾驶智能决策框架。上述研究思路既继承了强化学习在闭环决策中的优势，也吸收了多模态推理方法在复杂场景理解中的潜力，具有较明确的研究价值与现实意义。

## 参考文献

[1] Waymo. Waymo Driver[EB/OL]. https://waymo.com/waymo-driver/

[2] Webb N, Smith D, Ludwick C, Victor T, Hommes Q, Favaro F, Ivanov G, Daniel T. Waymo’s Safety Methodologies and Safety Readiness Determinations[EB/OL]. arXiv:2011.00054, 2020.

[3] Apollo Developer. Apollo Open Platform[EB/OL]. https://developer.apollo.auto/

[4] Fan H, Zhu F, Liu C, Zhang L, Zhuang L, Li D, Zhu W, Hu J, Li H, Kong Q. Baidu Apollo EM Motion Planner[EB/OL]. arXiv:1807.08048, 2018.

[5] Autoware Foundation. Autoware Overview[EB/OL]. https://autoware.org/autoware-overview/

[6] Zang Z, Tumu R, Betz J, Zheng H, Mangharam R. Winning the 3rd Japan Automotive AI Challenge -- Autonomous Racing with the Autoware.Auto Open Source Software Stack[EB/OL]. arXiv:2206.00770, 2022.

[7] Chen D, Koltun V, Krähenbühl P. Learning to Drive from a World on Rails[EB/OL]. arXiv:2105.00636, 2021.

[8] Li Z, Nie F, Sun Q, Da F, Zhao H. Boosting Offline Reinforcement Learning for Autonomous Driving with Hierarchical Latent Skills[EB/OL]. arXiv:2309.13614, 2023.

[9] Jia X, Yang Z, Li Q, Zhang Z, Yan J. Bench2Drive: Towards Multi-Ability Benchmarking of Closed-Loop End-to-End Autonomous Driving[EB/OL]. arXiv:2406.03877, 2024.

[10] Yang Z, Jia X, Li Q, Yang X, Yao M, Yan J. Raw2Drive: Reinforcement Learning with Aligned World Models for End-to-End Autonomous Driving (in CARLA v2)[EB/OL]. arXiv:2505.16394, 2025.

[11] Gao Y, Zhang Q, Liu D, Xia Z, Li G, Ma K, Chen G, Ye H, Chen L, Ding D W, Zhao D. PerlAD: Towards Enhanced Closed-loop End-to-end Autonomous Driving with Pseudo-simulation-based Reinforcement Learning[EB/OL]. arXiv:2603.14908, 2026.

[12] Yang Z, Peng Z, Ma J. SEG-Parking: Towards Safe, Efficient, and Generalizable Autonomous Parking via End-to-End Offline Reinforcement Learning[EB/OL]. arXiv:2509.13956, 2025.

[13] Sima C, Renz K, Chitta K, Chen L, Zhang H, Xie C, Beißwenger J, Luo P, Geiger A, Li H. DriveLM: Driving with Graph Visual Question Answering[EB/OL]. arXiv:2312.14150, 2023.

[14] Tian X, Gu J, Li B, Liu Y, Wang Y, Zhao Z, Zhan K, Jia P, Lang X, Zhao H. DriveVLM: The Convergence of Autonomous Driving and Large Vision-Language Models[EB/OL]. arXiv:2402.12289, 2024.

[15] Xu Z, Zhang Y, Xie E, Zhao Z, Guo Y, Wong K Y K, Li Z, Zhao H. Driveclaude4: Interpretable End-to-end Autonomous Driving via Large Language Model[EB/OL]. arXiv:2310.01412, 2023.

[16] Pan C, Yaman B, Nesti T, Mallik A, Allievi A G, Velipasalar S, Ren L. VLP: Vision-Language Planning for Autonomous Driving[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024.

[17] Zhou X, Han X, Yang F, Ma Y, Tresp V, Knoll A. OpenDriveVLA: Towards End-to-end Autonomous Driving with Large Vision Language Action Model[C]//Proceedings of the AAAI Conference on Artificial Intelligence. 2025.

[18] Hu A, Russell L, Yeo H, Murez Z, Fedoseev G, Kendall A, Shotton J, Corrado G. GAIA-1: A Generative World Model for Autonomous Driving[EB/OL]. arXiv:2309.17080, 2023.
