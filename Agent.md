# 《Agent系统设计与实战开发》大纲

## 第一部分：基础认知

### 第1章 Agent 概述

* 什么是 Agent（与传统AI的区别）
* Agent 的核心特征：自主、感知反应、主动、交互、学习
* 应用场景
* 常见挑战与局限性

---

### 第2章 LLM 基础

* LLM简介
* Transformer 简述（简单了解一下Attention机制）
* 幻觉（Hallucination）与不确定性
* API调用与推荐

### 实战部分：认识一个最简单的Agent——TravelPlanAgent

#### 本章目标

跑通一个最基础的命令行 Agent：

#### 本章实现

* 基础命令行聊天流程
* LLM API 调用

#### 本章结束后

TravelPlanAgent v0.1：

```text
一个最基础的命令行聊天 Agent
```

---

### 第3章 Token 与上下文机制

* Token 是什么
* 上下文窗口（Context Window）
* Token 成本与性能权衡

### 实战部分：实现多轮上下文记忆

#### 本章实现

* messages 维护
* 上下文拼接
* 历史裁剪

#### 本章结束后

TravelPlanAgent v0.2：

```text
支持基于上下文的连续对话
```
---

## 第二部分：Agent 核心能力构建

### 第4章 Prompt Engineering

* Prompt 基础结构（System / User / Tool）
* Few-shot / Zero-shot


### 实战部分：通过 Prompt 让 Agent 更像“助手”

#### 本章实现
* System Prompt 优化（固定角色）
* 回复风格控制
* 输出格式控制

#### 本章结束后

TravelPlanAgent v0.3：

```text
具备固定角色、回复风格与输出约束
```

---

### 第5章 Agent 核心组件

* Perception（输入处理）
* Memory（短期/长期）
* Tools 使用（API/函数调用）
* Planning（任务分解 ReAct CoT）
* Reflection（自我修正纠错/改进）

### 实战部分：实现第一个会调用工具的 Agent

#### 本章实现

* 天气查询 Tool（输入参数）
* Tool Schema 设计
* Function Calling 调用
* Tool Result Observation 观察

#### 本章结束后
TravelPlanAgent v0.4：

```text
可以自主调用天气工具
```

### 第6章 Agent 工作原理

* Think（推理）：缺什么信息？该调用什么工具？
* Act（行动）：调用外部工具（搜索 / API / 数据库）。
* Observe（观察）：拿到工具返回的真实结果，喂给模型，进入下一轮思考。
* Think Again
* Act Again
* Observe Again
* 循环直到能给出最终答案

### 实战部分：手写 ReAct Agent Loop

#### 本章实现

```python 
while True:
    pass
```

完整循环：

```text
Think -> Act -> Observe
```

#### 增加能力

* 自动判断是否调用工具（Think）
* 多轮工具调用（Act）
* 工具结果反馈（Observe）
* Agent 循环退出条件

#### 本章结束后

TravelPlanAgent v0.5：

```text
实现ReAct Agent
```
---

## 第三部分：知识与检索

### 第7章 RAG（Retrieval-Augmented Generation）

* 为什么需要 RAG
* 向量数据库（Embedding）
* Chunking 策略
* 检索策略（Top-K / Hybrid）
* RAG 的常见问题（召回失败、噪声）

### 实战部分：给 Agent 增加旅游知识检索

仍然坚持单文件Agent

仅新增：

```text
assets/一个 md 文件，包含所需内容的知识库
```

---

#### 本章实现

* md 知识库
* 分chunk
* 做embedding 
* 计算相似度
* 检索结果拼接到 Prompt

#### 本章结束后

TravelPlanAgent v0.6：

```text
具备旅游知识检索能力
```
---

### 第8章 Memory 系统设计

* 短期记忆（Conversation Buffer）
* 长期记忆（Vector DB / Knowledge Base）
* Episodic vs Semantic Memory
* Memory 更新策略

### 实战部分：实现长期记忆与用户偏好记忆

#### 本章实现

* memory.md 记忆文件
* 用户偏好记忆
* 对话摘要
* Memory 压缩

#### 本章结束后
TravelPlanAgent v0.7：

```text
记住用户偏好与历史行为
```

---

## 第四部分：Agent 架构设计

### 第9章 Agent 架构模式

* 单 Agent 架构
* Self-Reflection
* Plan & Execute
* Multi-Agent（协作 / 竞争）
* Workflow/DAG

### 实战部分：让 Agent 学会规划旅行任务

#### 本章实现

* Task Decomposition 任务分解
* 行程规划
* 子任务拆分
* Plan 生成

例如：

```text
1. 查询天气
2. 查询景点
3. 规划路线
4. 安排行程
```

#### 增加 Reflection

* 自我检查
* 错误修正
* 行程合理性检查

#### 本章结束后

TravelPlanAgent v0.8：

```text id="2fjqzd"
具备旅行任务规划能力
```

---

### 第10章 多 Agent 系统

* 角色分工（Planner / Executor / Critic）
* 通信机制
* 协作协议
* 冲突解决

### 实战部分：单文件 Multi-Agent 协作

必须坚持单文件实现multi-agent


```python
class PlannerAgent:
    pass
class ToolAgent:
    pass
class CriticAgent:
    pass
```

### 本章实现

#### Planner Agent：负责任务拆解

#### Tool Agent：负责调用工具获取信息

#### Critic Agent：负责结果检查


### 本章结束后

TravelPlanAgent v0.9：

```text
一个完整的多 Agent 旅行助手
```
---

## 第五部分：知识拓展

### 第11章 Agent 框架与工具链

* LangChain/LangGraph
* Pydantic AI
* LlamaIndex
* 国内的框架
* MCP（和Tool Calling的区别）

## 实战部分：把手写 Agent 迁移到框架

用 Pydantic AI 规范手写 Agent，并将之前做的 Agent 迁移到框架中，保持功能一致，体验框架带来的便利；同时为后续接入 MCP 这类标准化工具/资源协议预留扩展空间或者我们直接接入一个可行的MCP。

---

### 第12章 Harness Engineering
* 什么是Harness Engineering
* 为什么需要Harness Engineering
* Harness Engineering 的四大支柱
* 先进团队的实战


