from __future__ import annotations

import json
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from openai import OpenAI


SYSTEM_PROMPT = """
你是 TravelPlanAgent，一个友好、可靠、适合初学者理解的旅行规划助手。
你可以利用对话上下文和工具观察结果回答问题。
如果提供了工具结果，请基于工具结果给出旅行建议。
你的回答需要满足：
1. 优先围绕旅行目的地、时间、预算、交通、天气、景点和注意事项展开。
2. 语气清晰、亲切，不夸大，不编造无法确定的信息。
3. 必须说明本轮使用了哪些 Tools，以及每个 Tool 做了什么事。
4. 输出尽量包含三个部分：工具使用情况、建议、提醒。
"""

LOCAL_TRAVEL_TIPS = {
    "上海": "本地小贴士：上海城市交通方便，地铁适合串联人民广场、南京东路、外滩、陆家嘴。外滩夜景人流较多，建议错峰前往。",
    "北京": "本地小贴士：北京景点预约要求较多，故宫、国家博物馆等建议提前预约。城市尺度大，跨区游玩要给交通留足时间。",
    "杭州": "本地小贴士：杭州西湖适合步行和骑行，但节假日湖滨、断桥一带人流密集。灵隐寺适合安排在上午。",
    "成都": "本地小贴士：成都适合慢节奏旅行。熊猫基地建议上午早点去，宽窄巷子、锦里更适合体验氛围而不是安排过满。",
    "广州": "本地小贴士：广州早茶适合上午体验，老城区适合步行探索。夏季闷热且阵雨多，行程最好留室内备选。",
}

WEATHER_KEYWORDS = ["天气", "气温", "下雨", "晴", "多云", "穿什么", "带伞", "冷不冷", "热不热", "实时"]
GUIDE_KEYWORDS = ["攻略", "景点", "路线", "行程", "怎么玩", "推荐", "打卡", "美食", "住宿"]
LOCAL_TIP_KEYWORDS = ["提醒", "注意", "小贴士", "避坑", "本地", "交通", "预算", "适合"]

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


@dataclass
class LLMConfig:
    """集中管理模型调用参数，对应课程中的 API 调用基础。"""

    base_url: str = "https://chat.ecnu.edu.cn/open/api/v1"
    api_key: str = "sk-4b905783f8ab4fed9f7c1879aaf2ae58"
    model_name: str = "ecnu-max"
    temperature: float = 0.7
    timeout_seconds: int = 300


class LLMClient:
    """统一封装 LLM 请求，避免 Agent 主流程直接处理 HTTP 细节。"""

    def __init__(self, config: LLMConfig | None = None, openai_client: object | None = None) -> None:
        self.config = config or LLMConfig()
        self.openai_client = openai_client

    def chat(self, messages: list[dict[str, str]], temperature: float | None = None) -> str:
        if self.config.api_key in ["", "YOUR_API_KEY"]:
            return "请先把 LLMConfig 里的 api_key 替换成真实密钥。"

        try:
            openai_client = self._get_openai_client()
            completion = openai_client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                temperature=self.config.temperature if temperature is None else temperature,
            )
            return completion.choices[0].message.content or ""
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
    """保存最近几轮对话，对应 TravelPlanAgent v0.4-v0.5。"""

    max_rounds: int = 5
    messages: list[dict[str, str]] = field(default_factory=list)

    def add_user_message(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})
        self.trim()

    def add_assistant_message(self, content: str) -> None:
        self.messages.append({"role": "assistant", "content": content})
        self.trim()

    def build_messages(self, system_prompt: str) -> list[dict[str, str]]:
        return [{"role": "system", "content": system_prompt}] + self.messages

    def trim(self) -> None:
        max_messages = self.max_rounds * 2
        self.messages = self.messages[-max_messages:]


@dataclass
class ToolResult:
    """工具调用结果的统一格式，便于 Agent 组装观察结果。"""

    tool_name: str
    tool_function: str
    tool_input: str
    tool_action: str
    content: str


class BaseTool:
    """所有工具的统一接口，对应 TravelPlanAgent v0.5。"""

    name: str = "base_tool"
    function_name: str = "run"
    description: str = "基础工具"

    def run(self, city: str) -> ToolResult:
        raise NotImplementedError


class LocalTravelTipsTool(BaseTool):
    """读取代码内置的城市出行提醒和本地小贴士。"""

    name = "local_function"
    function_name = "get_local_travel_tips"
    description = "读取代码内置的城市出行提醒和本地小贴士。"

    def run(self, city: str) -> ToolResult:
        content = LOCAL_TRAVEL_TIPS.get(city, f"暂时没有 {city} 的本地小贴士，请提醒用户补充目的地信息。")
        return ToolResult(self.name, self.function_name, city, self.description, content)


class WeatherApiTool(BaseTool):
    """调用 Open-Meteo 天气 API 获取目的地当前天气。"""

    name = "weather_api"
    function_name = "get_weather_from_api"
    description = "调用天气 API 获取目的地当前天气。"

    def run(self, city: str) -> ToolResult:
        coordinates = CITY_COORDINATES.get(city)
        if not coordinates:
            content = f"天气 API 暂时没有内置 {city} 的经纬度，请先补充城市坐标。"
            return ToolResult(self.name, self.function_name, city, self.description, content)

        query = urllib.parse.urlencode(
            {
                "latitude": coordinates["latitude"],
                "longitude": coordinates["longitude"],
                "current": "temperature_2m,relative_humidity_2m,precipitation,weather_code,wind_speed_10m",
                "timezone": "Asia/Shanghai",
            }
        )
        url = f"https://api.open-meteo.com/v1/forecast?{query}"

        try:
            with urllib.request.urlopen(url, timeout=20) as response:
                data = json.loads(response.read().decode("utf-8"))
        except Exception as error:
            content = f"天气 API 调用失败：{error}"
            return ToolResult(self.name, self.function_name, city, self.description, content)

        current = data.get("current", {})
        weather_code = current.get("weather_code")
        weather_text = WEATHER_CODE_MAP.get(weather_code, f"未知天气代码 {weather_code}")
        content = (
            f"{city} 当前天气：{weather_text}，"
            f"气温 {current.get('temperature_2m')} 摄氏度，"
            f"相对湿度 {current.get('relative_humidity_2m')}%，"
            f"降水量 {current.get('precipitation')} mm，"
            f"风速 {current.get('wind_speed_10m')} km/h。"
            "数据来自 Open-Meteo 天气 API。"
        )
        return ToolResult(self.name, self.function_name, city, self.description, content)


class WebSearchTool(BaseTool):
    """通过 DuckDuckGo 搜索获取目的地攻略、景点和路线信息。"""

    name = "web_search"
    function_name = "web_search_travel_guide"
    description = "通过搜索获取目的地攻略、景点和路线信息。"

    def run(self, city: str) -> ToolResult:
        query = f"{city} 旅行 攻略 景点 路线"
        params = urllib.parse.urlencode(
            {
                "q": query,
                "format": "json",
                "no_html": 1,
                "skip_disambig": 1,
            }
        )
        url = f"https://api.duckduckgo.com/?{params}"

        try:
            with urllib.request.urlopen(url, timeout=20) as response:
                data = json.loads(response.read().decode("utf-8"))
        except Exception as error:
            content = f"web_search 调用失败：{error}"
            return ToolResult(self.name, self.function_name, city, self.description, content)

        snippets = []
        if data.get("AbstractText"):
            snippets.append(data["AbstractText"])
        for topic in data.get("RelatedTopics", []):
            if isinstance(topic, dict) and topic.get("Text"):
                snippets.append(topic["Text"])
            if len(snippets) >= 3:
                break

        if not snippets:
            content = f"web_search 没有找到稳定摘要。搜索词：{query}。请提醒用户可以补充更具体的攻略需求。"
        else:
            content = f"web_search 搜索词：{query}\n搜索摘要：\n" + "\n".join(f"- {snippet}" for snippet in snippets)
        return ToolResult(self.name, self.function_name, city, self.description, content)


class TravelPlanAgent:
    """
    TravelPlanAgent v0.5：三种不同职责的 Tool Calling。
    weather_api 查天气，web_search 找攻略，local_function 读取本地小贴士。
    """

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        memory: ConversationMemory | None = None,
        tools: list[BaseTool] | None = None,
        system_prompt: str = SYSTEM_PROMPT,
    ) -> None:
        self.llm_client = llm_client or LLMClient()
        self.memory = memory or ConversationMemory()
        self.tools = {
            tool.name: tool
            for tool in (tools or [WeatherApiTool(), WebSearchTool(), LocalTravelTipsTool()])
        }
        self.system_prompt = system_prompt

    def run(self, user_input: str) -> str:
        user_message = self.build_user_message(user_input)
        self.memory.add_user_message(user_message)
        messages = self.memory.build_messages(self.system_prompt)
        answer = self.llm_client.chat(messages)
        self.memory.add_assistant_message(answer)
        return answer

    def build_user_message(self, user_input: str) -> str:
        city = self.extract_city(user_input)
        if not city:
            return user_input

        tools_used = []
        for tool_name in self.select_tools(user_input):
            result = self.tools[tool_name].run(city)
            tool_record = {
                "tool_name": result.tool_name,
                "tool_function": result.tool_function,
                "tool_input": result.tool_input,
                "tool_action": result.tool_action,
                "tool_result": result.content,
            }
            tools_used.append(tool_record)
            print(f"[工具调用] {result.tool_function}(city='{result.tool_input}')")
            print(f"[工具作用] {result.tool_action}")
            print(f"[工具结果] {result.content}")

        return (
            f"{user_input}\n\n"
            "本轮 Agent 已经调用以下 Tools，请在回答开头说明使用了哪些 Tools，以及每个 Tool 做了什么事：\n"
            f"{json.dumps(tools_used, ensure_ascii=False, indent=2)}"
        )

    def select_tools(self, user_input: str) -> list[str]:
        tools_used = []
        if any(keyword in user_input for keyword in WEATHER_KEYWORDS):
            tools_used.append("weather_api")
        if any(keyword in user_input for keyword in GUIDE_KEYWORDS) or any(
            keyword in user_input.lower() for keyword in ["web", "search"]
        ):
            tools_used.append("web_search")
        if any(keyword in user_input for keyword in LOCAL_TIP_KEYWORDS) or not tools_used:
            tools_used.append("local_function")
        return tools_used

    def extract_city(self, text: str) -> str:
        for city in LOCAL_TRAVEL_TIPS:
            if city in text:
                return city
        return ""


def main() -> None:
    print("TravelPlanAgent v0.5：三种不同职责的 Tool Calling")
    print("weather_api：查天气；web_search：找攻略；local_function：读取本地出行小贴士。")
    print("输入 exit、quit 或 退出 可以结束对话。")

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
