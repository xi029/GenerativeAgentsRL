# Generative Agents CN: Self-Evolving Multi-Agent Simulation

<div align="center">

![Architecture](https://img.shields.io/badge/Architecture-Neuro--Symbolic-blue)
![Algorithm](<https://img.shields.io/badge/RL-GRPO%20(On--Policy)-orange>)
![Framework](https://img.shields.io/badge/Framework-NumPy%20Native-green)
![LLM](https://img.shields.io/badge/LLM-DeepSeek%20%7C%20Qwen-purple)

**[简体中文]** | [English](./README_en.md)

_基于斯坦福Generative Agents论文项目的深度汉化与**认知强化学习 (Cognitive Reinforcement Learning)** 增强版_

</div>

---

## 概述

本项目旨在探索 **LLM 驱动的智能体社会** 向 **具备自我进化能力的多智能体系统 (MAS)** 的演进。

在原版 [Generative Agents](https://arxiv.org/abs/2304.03442) 的基础上，我们引入了 **Agentic RL** 闭环。通过构建轻量级的 **GRPO (Group Relative Policy Optimization)** 策略网络，并创新性地将其与 **In-Context Learning (上下文学习)** 相结合，实现了数值奖励信号向符号化认知提示的转化。这使得智能体不仅能“模拟”人类行为，更能通过环境反馈“学习”并优化协作策略。

## 核心特性 (Technical Highlights)

### 1. 🧠 去中心化 GRPO 策略优化 (Decentralized GRPO)

> _摒弃 Value Network，直接优化策略分布，实现极低开销的在线学习。_

- **矩阵级高效计算**：底层完全基于 **NumPy** 构建策略梯度计算图，移除了 PyTorch/TensorFlow 等重型依赖，将单步推理延迟降低至微秒级，极其适合大规模多智能体仿真。
- **组内相对优势估计 (Group-Relative Advantage)**：采用 GRPO 算法核心思想，通过计算 Agent 在动态组内的相对表现（而非绝对奖励）来归一化优势函数，有效解决了多智能体环境下的 **非平稳性 (Non-stationarity)** 问题。
- **离散化动作空间映射**：将连续的语义意图映射为 `Move`, `Chat`, `Task`, `Research` 等离散动作原语，构建了可微分的策略更新路径。

### 2. 🧬 神经-符号反馈闭环 (Neuro-Symbolic Feedback Loop)

> _打通数值奖励与自然语言推理之间的鸿沟。_

- **奖励信号语义化 (Reward-to-Prompt Injection)**：不同于传统 RL 仅更新权重，我们将高价值轨迹（High-Reward Trajectories）转化为自然语言描述，动态注入到 LLM 的 System Prompt 中。
- **认知强化 (Cognitive Reinforcement)**：智能体能够“感知”到哪些行为模式带来了正向反馈（如高效协作、精准信息共享），从而在后续的 **Chain-of-Thought (CoT)** 推理中自发偏向高回报行为。

### 3. 🕸️ 动态联盟与任务编排 (Dynamic Coalition Formation)

> _支持 Ad-hoc Teamwork 的弹性组织架构。_

- **属性驱动聚类**：基于 Agent 的语义属性（如 `group: artist_group`）自动构建 RL 计算图，无需人工硬编码拓扑结构。
- **异构策略并行**：支持多个异构小组（如“科学家组”追求知识产出，“竞选组”追求影响力）在同一沙箱中并行训练，互不干扰，模拟复杂的社会分工。

### 4. 🧠 任务驱动的记忆共鸣 (Task-Driven Memory Resonance)

> _基于当前目标的动态记忆重排序。_

- **多维检索评分**：在传统的 `Recency`, `Importance`, `Relevance` 之外，引入第四维度 **Task Resonance**。
- **注意力聚焦**：确保 Agent 在决策时能优先召回与当前主线任务高度相关的记忆片段，防止长程任务中的目标漂移。

### 5. ⚖️ LLM-as-a-Judge 评估框架

> _基于大模型的语义级多维量化评估。_

- **语义一致性校验**：利用 DeepSeek-R1 / Qwen2.5 等推理模型，对仿真日志进行深度语义分析。
- **多维度量化指标**：
  - **Task Alignment**: 行为序列与长期目标的对齐度。
  - **Interaction Efficiency**: 信息熵视角下的交互有效性。
  - **Persona Consistency**: 长期记忆与即时行为的人设一致性。

---

## 🛠️ 快速部署

### 1. 环境构建

```bash
# 克隆仓库
git clone https://github.com/your-username/GenerativeAgentsCN.git
cd GenerativeAgentsCN

# 构建轻量级虚拟环境
conda create -n agent python=3.9
conda activate agent

# 安装依赖 (无重型DL框架)
pip install -r requirements.txt
```

### 2. 定义智能体画像 (Agent Profile)

通过 JSON 配置动态注入 RL 属性，无需修改代码：

_文件: `generative_agents/frontend/static/assets/village/agents/阿比盖尔/agent.json`_

```json
{
  "name": "阿比盖尔",
  "innate": "数字艺术家...",
  "group": "artist_collective",  // [RL] 定义所属策略组
  "task": "最大化艺术展的社区影响力", // [RL] 定义优化目标
  ...
}
```

### 3. 启动进化仿真

```bash
cd generative_agents
# 启动仿真，步长设为10以观察策略收敛
python start.py --name evolution_v1 --step 10 --stride 10
```

### 4. 效果验证

使用内置的 `eval.py` 进行 A/B 测试：

```bash
# 对比 Baseline 与 RL 版本的表现
python eval.py \
  --before results/compressed/baseline/simulation.md \
  --after results/compressed/evolution_v1/simulation.md \
  --model deepseek-chat
```

---

## 🤝 引用与致谢

本项目基于以下工作构建：

- **Generative Agents**: [Park et al., 2023](https://arxiv.org/abs/2304.03442)
- **DeepSeek-R1**: [DeepSeek AI, 2024](https://api.deepseek.com)
- **Codebase Refactoring**: [wounderland](https://github.com/Archermmt/wounderland)
- **GenerativeAgentsCN**: [GenerativeAgentsCN](https://github.com/x-glacier/GenerativeAgentsCN)

特别感谢@x-glacier对Generative Agents项目中文重构工作的贡献。

## 📄 License

Apache-2.0 license
