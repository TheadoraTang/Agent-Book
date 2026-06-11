import json
import logging
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field

import numpy as np


logger = logging.getLogger("travel_agent")
logger.setLevel(logging.INFO)
logger.handlers.clear()
handler = logging.StreamHandler(sys.stderr)
handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
logger.addHandler(handler)
logger.propagate = False


SYSTEM_PROMPT = """
你是 TravelPlanAgent，一个具备工具调用和 RAG 能力的旅行规划助手。
你需要根据用户问题自主决定调用哪些 Tools。

你可以使用四个 Tools：
- get_local_travel_tips：读取代码内置的城市出行提醒。
- get_weather_from_api：调用天气 API 获取实时天气。
- web_search_travel_guide：调用搜索 API 获取网页攻略、景点、路线信息。
- rag_search_travel_knowledge：使用 bge-m3 embedding 模型检索课程内置旅游知识库。

当用户询问目的地怎么玩、路线安排、景点建议、雨天安排、亲子/老人/预算等旅行规划问题时，
你应优先调用 rag_search_travel_knowledge 获取本地知识库片段。

最终回答必须说明：
1. 本轮使用了哪些 Tools。
2. 每个 Tool 做了什么事。
3. RAG 检索到了哪些相关知识，以及这些知识如何影响你的建议。
4. 如果 RAG 或外部工具结果不足，要明确提醒用户补充信息或出行前核验。
"""

LOCAL_TRAVEL_TIPS = {
    "上海": "本地小贴士：上海城市交通方便，地铁适合串联人民广场、南京东路、外滩、陆家嘴。外滩夜景人流较多，建议错峰前往。",
    "北京": "本地小贴士：北京景点预约要求较多，故宫、国家博物馆等建议提前预约。城市尺度大，跨区游玩要给交通留足时间。",
    "杭州": "本地小贴士：杭州西湖适合步行和骑行，但节假日湖滨、断桥一带人流密集。灵隐寺适合安排在上午。",
    "成都": "本地小贴士：成都适合慢节奏旅行。熊猫基地建议上午早点去，宽窄巷子、锦里更适合体验氛围而不是安排过满。",
    "广州": "本地小贴士：广州早茶适合上午体验，老城区适合步行探索。夏季闷热且阵雨多，行程最好留室内备选。",
}

CITY_COORDINATES = {
    "上海": {"latitude": 31.2304, "longitude": 121.4737},
    "北京": {"latitude": 39.9042, "longitude": 116.4074},
    "杭州": {"latitude": 30.2741, "longitude": 120.1551},
    "成都": {"latitude": 30.5728, "longitude": 104.0668},
    "广州": {"latitude": 23.1291, "longitude": 113.2644},
}

WEATHER_CODE_MAP = {
    0: "晴",
    1: "大致晴朗",
    2: "局部多云",
    3: "阴",
    45: "雾",
    48: "雾凇",
    51: "小毛毛雨",
    53: "中等毛毛雨",
    55: "强毛毛雨",
    61: "小雨",
    63: "中雨",
    65: "大雨",
    71: "小雪",
    73: "中雪",
    75: "大雪",
    80: "小阵雨",
    81: "中等阵雨",
    82: "强阵雨",
    95: "雷暴",
}

TRAVEL_KNOWLEDGE = """
# 上海城市旅行
上海适合第一次到访的经典路线是人民广场、南京东路、外滩、陆家嘴。白天可以看城市建筑和博物馆，晚上适合去外滩或陆家嘴看夜景。喜欢城市文化的游客可以安排武康路、思南路、衡山路、上海博物馆和上海当代艺术博物馆。亲子旅行可以考虑上海科技馆、上海自然博物馆和迪士尼。上海市内公共交通发达，地铁比打车更稳定。

# 北京历史文化旅行
北京适合历史文化主题旅行。经典路线包括天安门、故宫、景山公园、北海公园、什刹海和南锣鼓巷。故宫、国家博物馆等热门景点通常需要提前预约。长城距离市区较远，建议单独安排一天。北京城市尺度大，跨区移动耗时较长，行程不宜过密。

# 杭州西湖与茶文化
杭州适合慢节奏旅行。西湖可以安排断桥、白堤、苏堤、雷峰塔、曲院风荷和湖滨步行区。灵隐寺适合上午前往。喜欢茶文化可以去龙井村、中国茶叶博物馆和满觉陇。雨天可以安排南宋德寿宫遗址博物馆、浙江省博物馆、河坊街和室内茶馆。

# 成都慢旅行与美食
成都适合慢旅行和美食体验。常见路线包括大熊猫繁育研究基地、人民公园、宽窄巷子、武侯祠、锦里和太古里。熊猫基地建议上午早点去。美食可以安排火锅、串串、担担面、钟水饺和蛋烘糕。成都行程要留出喝茶、散步和休息的时间。

# 广州美食与城市文化
广州适合美食和城市文化旅行。常见路线包括陈家祠、沙面、永庆坊、北京路、珠江夜游和广东省博物馆。早茶适合安排在上午。夏季广州闷热且可能阵雨，行程应准备室内备选。广州地铁便利，但老城区步行体验也很好。

# 通用旅行规划原则
旅行规划时，每天不要安排过多景点。城市初访建议每天 2 到 4 个主要点位，并留出交通、排队、用餐和休息时间。亲子、老人同行时要降低步行强度。雨天优先安排博物馆、展览、商场、茶馆和餐饮体验。预算有限时，应优先选择公共交通和集中区域游玩。
"""


@dataclass
class LLMConfig:
    base_url: str = "https://chat.ecnu.edu.cn/open/api/v1"
    api_key: str = ""
    model_name: str = "ecnu-plus"
    temperature: float = 0.7
    timeout_seconds: int = 300

class LLMClient:
    def __init__(self, config: LLMConfig | None = None, openai_client: object | None = None) -> None:
        self.config = config or LLMConfig()
        self.openai_client = openai_client

    def chat_response(
        self,
        messages: list[dict[str, object]],
        temperature: float | None = None,
        tools: list[dict[str, object]] | None = None,
        tool_choice: str | None = None,
    ) -> dict[str, object]:
        if self.config.api_key in ["", "YOUR_API_KEY"]:
            return {"role": "assistant", "content": "请先把 LLMConfig 里的 api_key 替换成真实密钥。"}

        try:
            completion = self._get_openai_client().chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                temperature=self.config.temperature if temperature is None else temperature,
                tools=tools,
                tool_choice=tool_choice,
            )
            return self.message_to_dict(completion.choices[0].message)
        except ImportError:
            return {"role": "assistant", "content": "当前环境缺少 openai 包，请先运行：pip install openai"}
        except Exception as error:
            return {"role": "assistant", "content": f"调用模型时出现错误：{error}"}

    def chat(self, messages: list[dict[str, object]], temperature: float | None = None) -> str:
        return str(self.chat_response(messages, temperature=temperature).get("content", ""))

    def _get_openai_client(self) -> object:
        if self.openai_client is not None:
            return self.openai_client

        from openai import OpenAI

        self.openai_client = OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            timeout=self.config.timeout_seconds,
        )
        return self.openai_client

    def message_to_dict(self, message: object) -> dict[str, object]:
        if hasattr(message, "model_dump"):
            return message.model_dump(exclude_none=True)
        if isinstance(message, dict):
            return message
        return {
            "role": getattr(message, "role", "assistant"),
            "content": getattr(message, "content", ""),
            "tool_calls": getattr(message, "tool_calls", None) or [],
        }

@dataclass
class ConversationMemory:
    max_rounds: int = 5
    messages: list[dict[str, object]] = field(default_factory=list)

    def add_user_message(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})
        self.trim()

    def add_assistant_message(self, content: str) -> None:
        self.messages.append({"role": "assistant", "content": content})
        self.trim()

    def build_messages(self, system_prompt: str, user_input: str) -> list[dict[str, object]]:
        return [{"role": "system", "content": system_prompt}] + self.messages + [{"role": "user", "content": user_input}]

    def trim(self) -> None:
        self.messages = self.messages[-self.max_rounds * 2 :]

@dataclass
class ToolResult:
    tool_name: str
    function_name: str
    content: str
    success: bool = True
    raw_data: object = None


class BaseTool:
    name = "base_tool"
    function_name = "run"
    description = "基础工具"
    parameters: dict[str, object] = {"type": "object", "properties": {}}

    def to_openai_tool(self) -> dict[str, object]:
        return {"type": "function", "function": {"name": self.function_name, "description": self.description, "parameters": self.parameters}}

    def run(self, arguments: dict[str, object]) -> ToolResult:
        raise NotImplementedError


class LocalTravelTipsTool(BaseTool):
    name = "local_function"
    function_name = "get_local_travel_tips"
    description = "读取代码内置的城市出行提醒，包括交通、预约、人流、节奏和避坑建议。"
    parameters = {
        "type": "object",
        "properties": {"city": {"type": "string", "description": "城市名，例如：北京、上海、杭州。"}},
        "required": ["city"],
    }

    def run(self, arguments: dict[str, object]) -> ToolResult:
        city = str(arguments.get("city", "")).strip()
        content = LOCAL_TRAVEL_TIPS.get(city, f"没有识别到支持城市。当前支持：{', '.join(LOCAL_TRAVEL_TIPS)}。")
        return ToolResult(self.name, self.function_name, content, bool(city))


class WeatherApiTool(BaseTool):
    name = "weather_api"
    function_name = "get_weather_from_api"
    description = "调用天气 API 获取城市当前天气、气温、湿度、降水和风速。"
    parameters = LocalTravelTipsTool.parameters

    def run(self, arguments: dict[str, object]) -> ToolResult:
        city = str(arguments.get("city", "")).strip()
        coordinates = CITY_COORDINATES.get(city)
        if not coordinates:
            content = f"没有识别到支持城市，无法查询天气 API。当前支持：{', '.join(CITY_COORDINATES)}。"
            return ToolResult(self.name, self.function_name, content, False)

        query = urllib.parse.urlencode(
            {
                "latitude": coordinates["latitude"],
                "longitude": coordinates["longitude"],
                "current": "temperature_2m,relative_humidity_2m,precipitation,weather_code,wind_speed_10m",
                "timezone": "Asia/Shanghai",
            }
        )
        try:
            with urllib.request.urlopen(f"https://api.open-meteo.com/v1/forecast?{query}", timeout=20) as response:
                data = json.loads(response.read().decode("utf-8"))
        except Exception as error:
            return ToolResult(self.name, self.function_name, f"天气 API 调用失败：{error}", False)

        current = data.get("current", {})
        weather_text = WEATHER_CODE_MAP.get(current.get("weather_code"), f"未知天气代码 {current.get('weather_code')}")
        content = (
            f"{city} 当前天气：{weather_text}，"
            f"气温 {current.get('temperature_2m')} 摄氏度，"
            f"相对湿度 {current.get('relative_humidity_2m')}%，"
            f"降水量 {current.get('precipitation')} mm，"
            f"风速 {current.get('wind_speed_10m')} km/h。"
            "数据来自 Open-Meteo 天气 API。"
        )
        return ToolResult(self.name, self.function_name, content, True, current)


class WebSearchTool(BaseTool):
    name = "web_search"
    function_name = "web_search_travel_guide"
    description = "调用搜索 API 搜索目的地旅游攻略、景点、路线和注意事项。"
    search_api_url = "https://searchfree.site/api/search"
    parameters = {
        "type": "object",
        "properties": {"query_text": {"type": "string", "description": "搜索关键词，例如：北京三日游、上海亲子旅行攻略。"}},
        "required": ["query_text"],
    }

    def run(self, arguments: dict[str, object]) -> ToolResult:
        query_text = str(arguments.get("query_text", "")).strip()
        query = f"{query_text} 旅游攻略 景点 路线"
        request = urllib.request.Request(
            self.search_api_url,
            data=json.dumps({"query": query, "max_results": 3}).encode("utf-8"),
            headers={"Content-Type": "application/json", "User-Agent": "TravelPlanAgent/0.7"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="ignore")
            return ToolResult(self.name, self.function_name, f"web_search API 请求失败，状态码：{error.code}\n{detail}", False)
        except Exception as error:
            return ToolResult(self.name, self.function_name, f"web_search 调用失败：{error}", False)

        results = data.get("results", [])[:3]
        lines = ["web_search 搜索 API：searchfree.site", f"搜索词：{query}"]
        if data.get("answer"):
            lines.append(f"AI 摘要：{data['answer'][:300]}")
        lines.append("搜索结果：")
        for index, result in enumerate(results, start=1):
            lines.append(f"{index}. 标题：{result.get('title', '无标题')}")
            if result.get("content"):
                lines.append(f"   摘要：{result['content'][:160]}")
            if result.get("url"):
                lines.append(f"   链接：{result['url']}")
        if not results:
            lines.append("没有返回搜索结果。")
        return ToolResult(self.name, self.function_name, "\n".join(lines), bool(results), results)


class RagKnowledgeTool(BaseTool):
    name = "rag_search"
    function_name = "rag_search_travel_knowledge"
    description = "使用 bge-m3 embedding 模型检索课程内置旅游知识库，返回最相关的知识片段。"
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "用于 RAG 检索的问题，例如：杭州雨天怎么玩。"},
            "top_k": {"type": "integer", "description": "返回的知识片段数量，默认 3。"},
        },
        "required": ["query"],
    }

    def __init__(self, knowledge_text: str = TRAVEL_KNOWLEDGE, model_name: str = "BAAI/bge-m3") -> None:
        self.knowledge_text = knowledge_text
        self.model_name = model_name
        self.embedding_model = None
        self.chunks: list[str] | None = None
        self.embeddings = None

    def run(self, arguments: dict[str, object]) -> ToolResult:
        query = str(arguments.get("query", "")).strip()
        try:
            top_k = max(1, min(int(arguments.get("top_k", 3)), 5))
        except (TypeError, ValueError):
            top_k = 3

        try:
            chunks, chunk_embeddings = self.build_index()
            query_embedding = self.embed_texts([query])[0]
            scores = chunk_embeddings @ query_embedding
            ranked_indices = np.argsort(scores)[::-1][:top_k]
        except Exception as error:
            return ToolResult(self.name, self.function_name, f"RAG 检索失败：{error}", False)

        lines = ["RAG 检索工具：rag_search_travel_knowledge", f"检索问题：{query}", "最相关知识片段："]
        raw_results = []
        for rank, index in enumerate(ranked_indices, start=1):
            score = float(scores[index])
            lines.append(f"{rank}. 相似度：{score:.4f}")
            lines.append(chunks[index])
            raw_results.append({"score": score, "content": chunks[index]})
        return ToolResult(self.name, self.function_name, "\n".join(lines), True, raw_results)

    def build_index(self):
        if self.chunks is not None and self.embeddings is not None:
            return self.chunks, self.embeddings
        self.chunks = [chunk.strip() for chunk in self.knowledge_text.strip().split("\n\n") if chunk.strip()]
        logger.info("RAG: 正在为 %s 个知识片段生成 embedding...", len(self.chunks))
        self.embeddings = self.embed_texts(self.chunks)
        return self.chunks, self.embeddings

    def embed_texts(self, texts: list[str]):
        model = self.load_embedding_model()
        embeddings = model.encode(texts, batch_size=8, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(embeddings, dtype=np.float32)

    def load_embedding_model(self):
        if self.embedding_model is not None:
            return self.embedding_model
        logger.info("RAG: 正在加载 bge-m3 embedding 模型...")
        from sentence_transformers import SentenceTransformer

        self.embedding_model = SentenceTransformer(self.model_name)
        return self.embedding_model


def parse_tool_arguments(arguments_text: str) -> dict[str, object]:
    try:
        return json.loads(arguments_text or "{}")
    except json.JSONDecodeError:
        return {}


class TravelPlanAgent:
    def __init__(
        self,
        llm_client: LLMClient | None = None,
        memory: ConversationMemory | None = None,
        tools: list[BaseTool] | None = None,
        system_prompt: str = SYSTEM_PROMPT,
        max_tool_rounds: int = 3,
    ) -> None:
        self.llm_client = llm_client or LLMClient()
        self.memory = memory or ConversationMemory()
        self.tools = tools or [LocalTravelTipsTool(), WeatherApiTool(), WebSearchTool(), RagKnowledgeTool()]
        self.tool_map = {tool.function_name: tool for tool in self.tools}
        self.system_prompt = system_prompt
        self.max_tool_rounds = max_tool_rounds

    def run(self, user_input: str) -> str:
        messages = self.memory.build_messages(self.system_prompt, user_input)

        logger.info("Think: 向模型注册 Tools，让模型决定是否调用 RAG、天气、搜索或本地函数...")
        used_tools = []
        for round_index in range(1, self.max_tool_rounds + 1):
            response = self.llm_client.chat_response(
                messages,
                temperature=0.2,
                tools=[tool.to_openai_tool() for tool in self.tools],
                tool_choice="auto",
            )
            tool_calls = response.get("tool_calls", [])
            if not tool_calls:
                logger.info("Think: 模型没有继续调用工具。")
                if used_tools:
                    break
                final_answer = str(response.get("content", "模型没有返回可用回复。"))
                self.update_memory(user_input, final_answer)
                return final_answer

            logger.info("Think: 第 %s 轮请求调用 %s 个 Tool。", round_index, len(tool_calls))
            messages.append(response)
            for tool_call in tool_calls:
                result = self.execute_tool_call(tool_call)
                used_tools.append(result.function_name)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.get("id", ""),
                        "name": result.function_name,
                        "content": result.content,
                    }
                )

        logger.info("Finalizing: 将工具 Observe 结果交回模型，生成最终回答...")
        messages.append(
            {
                "role": "user",
                "content": (
                    "请停止调用工具，只根据上面的工具观察结果生成自然语言回答。"
                    "必须说明本轮使用了哪些工具，并重点说明 RAG 检索到的知识如何影响你的旅行建议。"
                ),
            }
        )
        final_response = self.llm_client.chat_response(messages, temperature=0.7)
        final_answer = str(final_response.get("content", "模型没有返回可用回复。"))
        self.update_memory(user_input, final_answer)
        return final_answer

    def execute_tool_call(self, tool_call: dict[str, object]) -> ToolResult:
        function_info = tool_call.get("function", {})
        function_name = function_info.get("name", "") if isinstance(function_info, dict) else ""
        arguments_text = function_info.get("arguments", "{}") if isinstance(function_info, dict) else "{}"
        arguments = parse_tool_arguments(str(arguments_text))

        tool = self.tool_map.get(function_name)
        if not tool:
            result = ToolResult("unknown_tool", function_name, f"未知工具：{function_name}", False)
        else:
            argument_text = ", ".join(f"{key}={value!r}" for key, value in arguments.items())
            logger.info("Tool Call: %s(%s)", function_name, argument_text)
            result = tool.run(arguments)

        logger.info("Observe: %s", result.content)
        return result

    def update_memory(self, user_input: str, final_answer: str) -> None:
        self.memory.add_user_message(user_input)
        self.memory.add_assistant_message(final_answer)


def run_agent(user_input: str, conversation_history: list[dict[str, object]]) -> str:
    memory = ConversationMemory(messages=list(conversation_history))
    agent = TravelPlanAgent(memory=memory)
    return agent.run(user_input)


def trim_history(conversation_history: list[dict[str, object]]) -> list[dict[str, object]]:
    return conversation_history[-ConversationMemory().max_rounds * 2 :]


def main() -> None:
    print("TravelPlanAgent v0.7：在 v0.6 基础上增加基于 bge-m3 embedding 的 RAG")
    print("可用 Tools：local_function、weather_api、web_search、rag_search_travel_knowledge。")
    print("输入 exit、quit 或 退出 可以结束对话。")

    agent = TravelPlanAgent()

    while True:
        user_input = input("\n你：").strip()
        if user_input.lower() in ["exit", "quit"] or user_input == "退出":
            print("TravelPlanAgent：下次旅行再见！")
            break

        final_answer = agent.run(user_input)
        print(f"TravelPlanAgent：{final_answer}")


if __name__ == "__main__":
    main()
