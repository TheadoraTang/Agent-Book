"""
TravelPlanAgent 架构模板

这个文件不是某一个版本的完整实现，而是一份“长期骨架”：

建议学习路径：
- v0.1：补全 LLMClient.chat，让 Agent 能调用模型。
- v0.2：完善 system_prompt，让 Agent 有固定角色和输出约束。
- v0.3：启用 ConversationMemory，让 Agent 支持多轮上下文。
- v0.4：注册 WeatherTool，让 Agent 能调用天气工具。
- v0.5：实现 think / act / observe，让 Agent 能自主选择工具。
- v0.6：实现 TravelKnowledgeRetriever，让 Agent 能做 RAG 检索。
- v0.7：实现 TravelPlanner 和 SubAgent，让主 Agent 能拆解并调度子任务。
- v0.8：用 Pydantic AI 等成熟框架替换这里的手写结构。
"""

import json
from dataclasses import dataclass, field
from openai import OpenAI


@dataclass
class LLMConfig:
    """集中管理模型调用参数，对应课程中的 API 调用基础。"""

    base_url: str = AI_BASE_URL
    api_key: str = ""
    model_name: str = AI_MODEL_NAME
    temperature: float = 0.7
    timeout_seconds: int = 300


class LLMClient:
    """统一封装 LLM 请求，避免 Agent 主流程直接处理 HTTP 细节。"""

    def __init__(self, config: LLMConfig | None = None, openai_client: object | None = None) -> None:
        self.config = config or LLMConfig()
        self.openai_client = openai_client

    def chat(self, messages: list[dict[str, str]], temperature: float | None = None) -> str:
        """发送 messages 给模型，并返回 assistant 的文本内容。

        TODO(v0.1)：这是第一个需要学生真正补全和理解的模型调用入口。
        本模板使用 OpenAI Python SDK，而不是手写 urllib HTTP 请求。
        """
        if self.config.api_key in ["", "YOUR_API_KEY"]:
            return (
                "当前还没有配置 API_KEY。\n"
                "请先在 LLMConfig(api_key='...') 中填入真实密钥，"
                "再运行 TravelPlanAgent。"
            )

        try:
            openai_client = self._get_openai_client()
            completion = openai_client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                temperature=self.config.temperature if temperature is None else temperature,
            )
            return completion.choices[0].message.content or ""
        except ImportError:
            return "当前环境缺少 openai 包，请先运行：pip install openai"
        except Exception as error:
            return f"调用模型时出现错误：{error}"

    def _get_openai_client(self) -> object:
        if self.openai_client is not None:
            return self.openai_client

        self.openai_client = OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            timeout=self.config.timeout_seconds,
        )
        return self.openai_client


@dataclass
class ConversationMemory:
    """保存最近几轮对话，对应 TravelPlanAgent v0.3。"""

    max_rounds: int = 5
    messages: list[dict[str, str]] = field(default_factory=list)

    def add_user_message(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})
        self.trim()

    def add_assistant_message(self, content: str) -> None:
        self.messages.append({"role": "assistant", "content": content})
        self.trim()

    def build_messages(self, system_prompt: str, user_input: str) -> list[dict[str, str]]:
        """把系统提示、历史对话和当前输入组装成模型能理解的 messages。"""
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(self.messages)
        messages.append({"role": "user", "content": user_input})
        return messages

    def trim(self) -> None:
        max_messages = self.max_rounds * 2
        self.messages = self.messages[-max_messages:]


@dataclass
class ToolResult:
    """工具调用结果的统一格式，便于后续 Observe 阶段处理。"""

    tool_name: str
    success: bool
    content: str
    raw_data: object = None


class BaseTool:
    """所有工具的统一接口，对应 TravelPlanAgent v0.4-v0.5。"""

    name: str = "base_tool"
    description: str = "基础工具"

    def run(self, arguments: dict[str, object]) -> ToolResult:
        raise NotImplementedError


class WeatherTool(BaseTool):
    """天气工具模板。

    TODO(v0.4)：这里先用本地字典模拟天气，后续再替换为真实天气 API。
    """

    name = "weather"
    description = "查询目的地天气，适合回答气温、降雨、穿衣和出行风险。"

    city_weather_examples = {
        "上海": "上海今天多云，适合城市步行，但建议带一件薄外套。",
        "北京": "北京今天晴朗干燥，适合户外参观，注意补水和防晒。",
        "杭州": "杭州今天可能有阵雨，建议优先安排博物馆、茶馆和室内景点。",
        "成都": "成都今天阴天，适合慢节奏游玩和美食体验。",
        "广州": "广州今天较热，可能有短时阵雨，建议准备雨具并减少暴晒步行。",
    }

    def run(self, arguments: dict[str, object]) -> ToolResult:
        city = str(arguments.get("city", "")).strip()
        if not city:
            return ToolResult(self.name, False, "缺少 city 参数，无法查询天气。")

        weather = self.city_weather_examples.get(city)
        if not weather:
            supported = "、".join(self.city_weather_examples)
            return ToolResult(self.name, False, f"暂不支持 {city}。当前支持：{supported}。")

        return ToolResult(self.name, True, f"{weather}（示例天气数据，后续版本可替换为真实 API。）")


class TravelKnowledgeRetriever:
    """旅行知识检索器模板，对应 TravelPlanAgent v0.6 的 RAG 能力。

    这里先用关键词匹配表达结构。后续可以替换为：
    1. 文档切分；
    2. embedding；
    3. 向量相似度检索；
    4. 将检索结果放回 prompt。
    """

    def __init__(self) -> None:
        self.knowledge_chunks = [
            "杭州适合慢节奏旅行，西湖、灵隐寺、龙井村和博物馆都适合安排。",
            "雨天旅行可以优先安排博物馆、展览、茶馆、商场和餐饮体验。",
            "亲子或老人同行时，每天不要安排过多景点，要留出交通和休息时间。",
            "预算有限时，优先选择公共交通和集中区域游玩，减少跨城跨区移动。",
        ]

    def retrieve(self, query: str, top_k: int = 3) -> list[str]:
        """返回与 query 相关的知识片段。

        TODO(v0.6)：把当前关键词匹配替换为 embedding 检索。
        """
        scored_chunks = []
        for chunk in self.knowledge_chunks:
            score = sum(1 for char in set(query) if char in chunk)
            scored_chunks.append((score, chunk))
        scored_chunks.sort(reverse=True, key=lambda item: item[0])
        return [chunk for score, chunk in scored_chunks[:top_k] if score > 0]


@dataclass
class SubAgentResult:
    """子 Agent 的统一输出格式，方便主 Agent 汇总观察结果。"""

    agent_name: str
    role: str
    task: dict[str, object]
    success: bool
    observation: str


class BaseSubAgent:
    """所有子 Agent 的统一接口，对应 TravelPlanAgent v0.7。

    Tool 是底层动作，例如查天气；SubAgent 是任务执行者，例如负责天气判断。
    """

    name: str = "base_sub_agent"
    role: str = "基础子 Agent"

    def run(self, task: dict[str, object], context: dict[str, object]) -> SubAgentResult:
        raise NotImplementedError


class WeatherSubAgent(BaseSubAgent):
    """负责天气相关子任务，可以调用 WeatherTool。"""

    name = "weather_agent"
    role = "负责天气查询和天气出行建议"

    def __init__(self, weather_tool: WeatherTool) -> None:
        self.weather_tool = weather_tool

    def run(self, task: dict[str, object], context: dict[str, object]) -> SubAgentResult:
        result = self.weather_tool.run(task.get("arguments", {}))
        return SubAgentResult(
            agent_name=self.name,
            role=self.role,
            task=task,
            success=result.success,
            observation=result.content,
        )


class KnowledgeSubAgent(BaseSubAgent):
    """负责旅行知识检索和依据整理，可以调用 TravelKnowledgeRetriever。"""

    name = "knowledge_agent"
    role = "负责从旅行知识库中检索和整理依据"

    def __init__(self, retriever: TravelKnowledgeRetriever) -> None:
        self.retriever = retriever

    def run(self, task: dict[str, object], context: dict[str, object]) -> SubAgentResult:
        query = task.get("query") or context.get("user_input", "")
        chunks = self.retriever.retrieve(query)
        observation = "\n".join(chunks) if chunks else "没有检索到相关旅行知识。"
        return SubAgentResult(
            agent_name=self.name,
            role=self.role,
            task=task,
            success=bool(chunks),
            observation=observation,
        )


class ItinerarySubAgent(BaseSubAgent):
    """负责把已有观察结果整理成行程草案。

    TODO(v0.7)：这里可以升级为调用 LLM，生成更完整的结构化行程草案。
    """

    name = "itinerary_agent"
    role = "负责综合天气、知识库和用户需求生成行程草案"

    def run(self, task: dict[str, object], context: dict[str, object]) -> SubAgentResult:
        previous_observations = context.get("observations", [])
        if not previous_observations:
            observation = "暂时没有可综合的信息，需要先查询天气或检索旅行知识。"
            success = False
        else:
            observation = (
                "行程草案应综合以下依据：\n"
                f"{json.dumps(previous_observations, ensure_ascii=False, indent=2)}"
            )
            success = True
        return SubAgentResult(
            agent_name=self.name,
            role=self.role,
            task=task,
            success=success,
            observation=observation,
        )


class DirectAnswerSubAgent(BaseSubAgent):
    """处理不需要工具和 RAG 的普通对话任务。"""

    name = "direct_answer_agent"
    role = "负责无需外部工具的直接回答"

    def run(self, task: dict[str, object], context: dict[str, object]) -> SubAgentResult:
        return SubAgentResult(
            agent_name=self.name,
            role=self.role,
            task=task,
            success=True,
            observation="不需要额外工具，可以直接根据对话上下文回答。",
        )


class TravelPlanner:
    """旅行任务规划器，对应 TravelPlanAgent v0.7。"""

    def build_sub_tasks(self, user_input: str) -> list[dict[str, object]]:
        """把用户需求拆成若干子 Agent 任务。

        TODO(v0.7)：可以改为让 LLM 输出结构化 JSON 任务列表。
        """
        tasks: list[dict[str, object]] = []
        city = self.extract_city(user_input)

        if object(keyword in user_input for keyword in ["天气", "下雨", "气温", "穿什么"]):
            tasks.append(
                {
                    "agent": "weather_agent",
                    "task": "查询目的地天气，并给出出行提醒。",
                    "arguments": {"city": city},
                }
            )

        if object(keyword in user_input for keyword in ["攻略", "路线", "行程", "怎么玩", "亲子", "老人", "预算"]):
            tasks.append(
                {
                    "agent": "knowledge_agent",
                    "task": "检索旅行知识库，找到适合回答用户需求的依据。",
                    "query": user_input,
                }
            )

        if object(keyword in user_input for keyword in ["攻略", "路线", "行程", "计划", "安排", "几日游"]):
            tasks.append(
                {
                    "agent": "itinerary_agent",
                    "task": "综合前面的观察结果，整理行程草案。",
                    "query": user_input,
                }
            )

        if not tasks:
            tasks.append(
                {
                    "agent": "direct_answer_agent",
                    "task": "直接回答用户输入。",
                    "query": user_input,
                }
            )

        return tasks

    def build_plan_steps(self, user_input: str) -> list[dict[str, object]]:
        """兼容旧命名：早期版本可以把子任务理解为计划步骤。"""
        return self.build_sub_tasks(user_input)

    def extract_city(self, text: str) -> str:
        for city in ["上海", "北京", "杭州", "成都", "广州"]:
            if city in text:
                return city
        return ""


class TravelPlanAgent:
    """主 Agent：组织 Think-Act-Observe-Final Answer。"""

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        memory: ConversationMemory | None = None,
        planner: TravelPlanner | None = None,
        retriever: TravelKnowledgeRetriever | None = None,
        tools: list[BaseTool] | None = None,
    ) -> None:
        self.llm_client = llm_client or LLMClient()
        self.memory = memory or ConversationMemory()
        self.planner = planner or TravelPlanner()
        self.retriever = retriever or TravelKnowledgeRetriever()
        self.tools = {tool.name: tool for tool in (tools or [WeatherTool()])}
        self.sub_agents = self.build_sub_agents()

        # TODO(v0.2)：逐步完善角色、风格、边界和输出约束。
        self.system_prompt = (
            "你是 TravelPlanAgent，一个友好、可靠、适合初学者理解的旅行规划助手。"
            "你会根据用户需求给出清晰、具体、不过度承诺的旅行建议。"
            "如果涉及实时信息，要说明信息来源或提醒用户出行前再次核验。"
        )

    def build_sub_agents(self) -> dict[str, BaseSubAgent]:
        """注册子 Agent。

        TODO(v0.7)：后续可以把这里扩展成配置化注册，或替换为框架提供的 Agent 编排。
        """
        weather_tool = self.tools.get("weather")
        if not isinstance(weather_tool, WeatherTool):
            weather_tool = WeatherTool()

        agents: list[BaseSubAgent] = [
            WeatherSubAgent(weather_tool),
            KnowledgeSubAgent(self.retriever),
            ItinerarySubAgent(),
            DirectAnswerSubAgent(),
        ]
        return {agent.name: agent for agent in agents}

    def run(self, user_input: str) -> str:
        """对外主入口：输入用户问题，返回 Agent 回答。"""
        tasks = self.think(user_input)
        observations = self.dispatch(tasks, user_input)
        answer = self.final_answer(user_input, observations)

        # TODO(v0.3)：正式讲上下文时，再强调为什么要保存 user 和 assistant 两类消息。
        self.memory.add_user_message(user_input)
        self.memory.add_assistant_message(answer)
        return answer

    def think(self, user_input: str) -> list[dict[str, object]]:
        """Think：把用户输入拆成需要哪些子 Agent 完成的任务。"""
        # TODO(v0.7)：这里可以升级为让 LLM 输出结构化 JSON 子任务列表。
        return self.planner.build_sub_tasks(user_input)

    def dispatch(self, tasks: list[dict[str, object]], user_input: str) -> list[dict[str, object]]:
        """Dispatch：主 Agent 调度子 Agent，并收集观察结果。"""
        observations: list[dict[str, object]] = []
        context: dict[str, object] = {"user_input": user_input, "observations": observations}

        for task in tasks:
            agent_name = task.get("agent", "")
            sub_agent = self.sub_agents.get(agent_name)
            if not sub_agent:
                observations.append(
                    {
                        "agent_name": agent_name or "unknown_agent",
                        "role": "未注册子 Agent",
                        "task": task,
                        "success": False,
                        "observation": f"没有找到对应的子 Agent：{agent_name}",
                    }
                )
                continue

            result = sub_agent.run(task, context)
            observations.append(
                {
                    "agent_name": result.agent_name,
                    "role": result.role,
                    "task": result.task,
                    "success": result.success,
                    "observation": result.observation,
                }
            )
            context["observations"] = observations

        return observations

    def act_and_observe(self, steps: list[dict[str, object]]) -> list[dict[str, object]]:
        """Act + Observe：早期版本的工具执行入口。

        v0.4-v0.6 可以先使用这个方法理解工具调用；
        v0.7 之后推荐使用 dispatch 调度子 Agent。
        """
        observations: list[dict[str, object]] = []

        for step in steps:
            step_type = step.get("type")

            if step_type == "tool":
                tool_name = step.get("tool", "")
                tool = self.tools.get(tool_name)
                if not tool:
                    observations.append(
                        {
                            "type": "tool",
                            "success": False,
                            "content": f"没有找到工具：{tool_name}",
                        }
                    )
                    continue

                result = tool.run(step.get("arguments", {}))
                observations.append(
                    {
                        "type": "tool",
                        "tool_name": result.tool_name,
                        "success": result.success,
                        "content": result.content,
                    }
                )

            elif step_type == "retrieve":
                chunks = self.retriever.retrieve(step.get("query", ""))
                observations.append(
                    {
                        "type": "retrieve",
                        "success": bool(chunks),
                        "content": "\n".join(chunks) if chunks else "没有检索到相关旅行知识。",
                    }
                )

            else:
                observations.append(
                    {
                        "type": "chat",
                        "success": True,
                        "content": "不需要额外工具，可以直接根据对话上下文回答。",
                    }
                )

        return observations

    def final_answer(self, user_input: str, observations: list[dict[str, object]]) -> str:
        """Final Answer：把用户问题、历史上下文和子 Agent 观察结果交给 LLM。"""
        observation_text = json.dumps(observations, ensure_ascii=False, indent=2)
        prompt = f"""
            用户问题：
            {user_input}
            
            本轮子 Agent 已获得的观察结果：
            {observation_text}
            
            请用 TravelPlanAgent 的身份回答用户。
            要求：
            1. 先直接回应用户最关心的问题。
            2. 如果使用了天气或知识库信息，要自然说明依据。
            3. 如果信息不足，要明确告诉用户还需要补充什么。
            4. 不要暴露内部的主 Agent、子 Agent、调度、Think / Act / Observe 结构，除非用户主动询问。
            """
        messages = self.memory.build_messages(self.system_prompt, prompt)
        return self.llm_client.chat(messages)


def main() -> None:
    print("TravelPlanAgent 架构模板")
    print("输入 exit、quit 或 退出 可以结束对话。")
    print("提示：如果没有配置 API_KEY，程序会返回教学提示，不会直接崩溃。")

    agent = TravelPlanAgent()

    while True:
        try:
            user_input = input("\n你：").strip()
        except EOFError:
            print("\nTravelPlanAgent：输入已结束，下次旅行再见！")
            break
        if user_input.lower() in ["exit", "quit"] or user_input == "退出":
            print("TravelPlanAgent：下次旅行再见！")
            break

        answer = agent.run(user_input)
        print(f"TravelPlanAgent：{answer}")


if __name__ == "__main__":
    main()
