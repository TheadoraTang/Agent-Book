from __future__ import annotations

import json
import logging
import sys
import urllib.parse
import urllib.request
from dataclasses import dataclass, field


logger = logging.getLogger("travel_agent")
logger.setLevel(logging.INFO)
logger.handlers.clear()
handler = logging.StreamHandler(sys.stderr)
handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
logger.addHandler(handler)
logger.propagate = False


SYSTEM_PROMPT = """
你是 TravelPlanAgent，一个友好、可靠、适合初学者理解的旅行规划助手。
本版本只演示一种 Tool：weather_api。
每一轮对话都已经调用 weather_api，并把工具结果提供给你。
你的回答必须说明：
1. 本轮使用了哪个 Tool。
2. 这个 Tool 做了什么事。
3. 这个 Tool 返回了什么信息。
然后再给出旅行建议和提醒。
"""

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
    """
    集中管理模型调用参数
    """

    base_url: str = "https://chat.ecnu.edu.cn/open/api/v1"
    api_key: str = ""
    model_name: str = "ecnu-max"
    temperature: float = 0.7
    timeout_seconds: int = 300


class LLMClient:
    """
    统一封装 LLM 请求
    """

    def __init__(self, config: LLMConfig | None = None, openai_client: object | None = None) -> None:
        self.config = config or LLMConfig()
        self.openai_client = openai_client

    def chat(self, messages: list[dict[str, str]], temperature: float | None = None) -> str:
        if self.config.api_key in ["", "YOUR_API_KEY"]:
            raise RuntimeError("请先把 LLMConfig 里的 api_key 替换成真实密钥。")

        openai_client = self._get_openai_client()
        completion = openai_client.chat.completions.create(
            model=self.config.model_name,
            messages=messages,
            temperature=self.config.temperature if temperature is None else temperature,
        )
        return completion.choices[0].message.content or ""

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

@dataclass
class ConversationMemory:
    """
    保存最近几轮对话。
    """

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
        self.messages = self.messages[-self.max_rounds * 2 :]

@dataclass
class ToolResult:
    """
    工具调用结果的统一格式。
    """

    tool_name: str
    tool_function: str
    tool_input: str
    tool_action: str
    content: str


class BaseTool:
    """
    所有工具的统一接口。
    """

    name: str = "base_tool"
    function_name: str = "run"
    description: str = "基础工具"

    def run(self, tool_input: str) -> ToolResult:
        raise NotImplementedError


class WeatherApiTool(BaseTool):
    """
    调用 Open-Meteo 天气 API 获取目的地当前天气。
    """

    name = "weather_api"
    function_name = "get_weather_from_api"
    description = "调用天气 API 获取目的地当前天气。"

    def run(self, city: str) -> ToolResult:
        coordinates = CITY_COORDINATES.get(city)
        if not coordinates:
            content = f"没有识别到支持城市，无法查询天气 API。当前支持：{', '.join(CITY_COORDINATES)}。"
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
            raise RuntimeError("天气 API 调用失败。") from error

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


class TravelPlanAgent:
    """
    TravelPlanAgent v0.5 tool2：每轮只调用 weather_api。
    """

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        memory: ConversationMemory | None = None,
        tool: BaseTool | None = None,
        system_prompt: str = SYSTEM_PROMPT,
    ) -> None:
        self.llm_client = llm_client or LLMClient()
        self.memory = memory or ConversationMemory()
        self.tool = tool or WeatherApiTool()
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
        result = self.tool.run(city)
        tool_record = self.build_tool_record(result)

        logger.info("Tool Call: %s(city='%s')", result.tool_function, result.tool_input)
        logger.info("Tool Action: %s", result.tool_action)
        logger.info("Observe: %s", result.content)

        return (
            f"{user_input}\n\n"
            "本轮必须调用的 Tool 已经执行。请在回答开头说明 Tool 使用情况：\n"
            f"{json.dumps(tool_record, ensure_ascii=False, indent=2)}"
        )

    def build_tool_record(self, result: ToolResult) -> dict[str, str]:
        return {
            "tool_name": result.tool_name,
            "tool_function": result.tool_function,
            "tool_input": result.tool_input,
            "tool_action": result.tool_action,
            "tool_result": result.content,
        }

    def extract_city(self, text: str) -> str:
        for city in CITY_COORDINATES:
            if city in text:
                return city
        return "未识别城市"


def main() -> None:
    print("TravelPlanAgent v0.5 tool2：只调用 weather_api")
    print("这个版本每轮都会调用 get_weather_from_api。")
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
