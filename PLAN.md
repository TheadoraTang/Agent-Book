# TravelPlanAgent v0.1-v0.7 单文件实现计划

## Summary
- 在仓库根目录新增 7 个独立 Python 文件：`TravelPlanAgent_v0.1.py` 到 `TravelPlanAgent_v0.7.py`。
- 每个文件都是可单独运行的命令行程序，并且按上一版本小步迭代，方便学生对比能力增长。
- 使用模型配置：
  - `MODEL_NAME = "ecnu-max"`
  - `BASE_URL = "https://chat.ecnu.edu.cn/open/api/v1"`
  - `API_KEY = "YOUR_API_KEY"`，等你提供后可替换。
- 只使用 Python 标准库，优先用 `urllib.request` 调用 OpenAI-compatible `/chat/completions` 接口，不引入 `openai`、`requests`、LangChain 等依赖。

## Key Changes
- `v0.1`：最小命令行聊天 Agent。
  - 接收用户输入，构造 `messages`，调用 LLM，打印回复。
  - 不保存历史，不设复杂角色。

- `v0.2`：加入旅行助手角色和输出约束。
  - 新增 `SYSTEM_PROMPT`。
  - 回复风格固定为友好、清晰、适合旅行规划。
  - 要求输出包含建议、理由和提醒。

- `v0.3`：加入 5 轮上下文记忆。
  - 维护 `conversation_history`。
  - 每轮追加 user/assistant 消息。
  - 超过 5 轮时保留最近上下文，避免 messages 无限增长。

- `v0.4`：加入天气工具。
  - 内置 `get_weather(city)` 模拟天气数据。
  - 让 Agent 根据用户问题判断是否需要天气。
  - 实现方式采用教学友好的规则判断：识别天气、气温、下雨、穿衣等关键词和常见城市名。
  - 工具结果会拼接进 prompt，再让 LLM 生成自然语言回答。

- `v0.5`：加入 Think-Act-Observe 循环。
  - 单文件内实现 `think()`、`act()`、`observe()`、`answer()` 四步。
  - Think 阶段让 LLM 输出 JSON 决策：是否调用工具、调用哪个工具、参数是什么。
  - Act 阶段执行内置工具。
  - Observe 阶段把工具结果交回 LLM。
  - 为稳定教学，若 JSON 解析失败，会回退到普通回答。

- `v0.6`：加入旅游知识检索。
  - 单文件内嵌小型旅游知识库，不额外创建 assets 文件。
  - 实现简单 chunk 切分、关键词重叠评分检索。
  - 新增 `search_travel_knowledge(query)` 工具。
  - Agent 可以结合天气工具和知识检索回答问题。

- `v0.7`：完整旅行规划 Agent。
  - 在 v0.6 基础上加入任务分解和自我检查。
  - 实现 `plan_tasks()`：把旅行规划拆成天气、景点、路线、行程、提醒等子任务。
  - 实现 `reflect_plan()`：检查行程是否缺少天气、预算、交通、节奏或注意事项。
  - 最终输出结构化旅行方案：需求理解、信息依据、每日安排、交通建议、预算提醒、风险提醒、可调整项。

## Interfaces
- 每个文件都包含相同的基础配置区：
  - `BASE_URL`
  - `API_KEY`
  - `MODEL_NAME`
  - `CHAT_COMPLETIONS_URL`
- 每个文件都提供：
  - `call_llm(messages, temperature=0.7)`
  - `main()`
- 从 `v0.4` 起新增工具函数；从 `v0.5` 起新增 Agent 循环函数。
- 所有文件都支持命令行运行：
  ```powershell
  python TravelPlanAgent_v0.1.py
  ```
- 退出命令统一支持：`exit`、`quit`、`退出`。

## Test Plan
- 对 7 个文件分别运行语法检查：
  ```powershell
  python -m py_compile TravelPlanAgent_v0.1.py TravelPlanAgent_v0.2.py TravelPlanAgent_v0.3.py TravelPlanAgent_v0.4.py TravelPlanAgent_v0.5.py TravelPlanAgent_v0.6.py TravelPlanAgent_v0.7.py
  ```
- 在未填真实 API key 时，运行后应给出清晰提示，不应抛出难懂异常。
- 填入 API key 后，逐个手动验证：
  - v0.1 能单轮聊天。
  - v0.2 回复符合旅行助手身份。
  - v0.3 能记住最近 5 轮对话。
  - v0.4 对“上海明天天气适合玩吗”能使用天气信息。
  - v0.5 能展示 Think/Act/Observe 过程并最终回答。
  - v0.6 能检索内嵌旅游知识。
  - v0.7 能生成完整旅行计划并做自我检查。

## Assumptions
- `https://chat.ecnu.edu.cn/open/api/v1` 兼容 OpenAI Chat Completions 风格，请求路径使用 `/chat/completions`。
- API key 暂时使用占位符变量，等你随后提供后再替换。
- 天气和旅游知识库采用教学内置模拟数据，不调用真实外部天气、地图或搜索 API。
- 所有代码放在仓库根目录，不新增代码目录，不新增依赖文件。
