from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field

SYSTEM_PROMPT = """
你是 TravelPlanAgent，一个友好、可靠、适合初学者理解的旅行规划助手。
本版本只演示一种 Tool：local_function。
每一轮对话都已经调用 local_function，并把工具结果提供给你。
你的回答必须说明：
1. 本轮使用了哪个 Tool。
2. 这个 Tool 做了什么事。
3. 这个 Tool 返回了什么信息。
然后再给出旅行建议和提醒。
"""

LOCAL_TRAVEL_TIPS = {
    "上海": "本地小贴士：上海城市交通方便，地铁适合串联人民广场、南京东路、外滩、陆家嘴。外滩夜景人流较多，建议错峰前往。",
    "北京": "本地小贴士：北京景点预约要求较多，故宫、国家博物馆等建议提前预约。城市尺度大，跨区游玩要给交通留足时间。",
    "杭州": "本地小贴士：杭州西湖适合步行和骑行，但节假日湖滨、断桥一带人流密集。灵隐寺适合安排在上午。",
    "成都": "本地小贴士：成都适合慢节奏旅行。熊猫基地建议上午早点去，宽窄巷子、锦里更适合体验氛围而不是安排过满。",
    "广州": "本地小贴士：广州早茶适合上午体验，老城区适合步行探索。夏季闷热且阵雨多，行程最好留室内备选。",
}

logger = logging.getLogger("travel_agent")
logger.setLevel(logging.INFO)
logger.handlers.clear()
handler = logging.StreamHandler(sys.stderr)
handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
logger.addHandler(handler)
logger.propagate = False


@dataclass
class LLMConfig:
    base_url: str = "https://chat.ecnu.edu.cn/open/api/v1"
    api_key: str = "sk-4b905783f8ab4fed9f7c1879aaf2ae58"
    model_name: str = "ecnu-max"
    temperature: float = 0.7
    timeout_seconds: int = 300


class LLMClient:
    def __init__(self, config: LLMConfig | None = None, openai_client: object | None = None) -> None:
        self.config = config or LLMConfig()
        self.openai_client = openai_client

    def chat(self, messages: list[dict[str, str]], temperature: float | None = None) -> str:
        if self.config.api_key in ["", "YOUR_API_KEY"]:
            return "请先把 LLMConfig 里的 api_key 替换成真实密钥。"

        try:
            completion = self._get_openai_client().chat.completions.create(
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
        if self.openai_client is None:
            from openai import OpenAI

            self.openai_client = OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                timeout=self.config.timeout_seconds,
            )
        return self.openai_client


@dataclass
class ConversationMemory:
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
    tool_name: str
    tool_function: str
    tool_input: str
    tool_action: str
    content: str


class LocalTravelTipsTool:
    name = "local_function"
    function_name = "get_local_travel_tips"
    description = "读取代码内置的城市出行提醒和本地小贴士。"

    def run(self, city: str) -> ToolResult:
        content = LOCAL_TRAVEL_TIPS.get(
            city,
            f"没有识别到支持城市。当前支持：{', '.join(LOCAL_TRAVEL_TIPS)}。",
        )
        return ToolResult(self.name, self.function_name, city, self.description, content)


class TravelPlanAgent:
    def __init__(
        self,
        llm_client: LLMClient | None = None,
        memory: ConversationMemory | None = None,
        tool: LocalTravelTipsTool | None = None,
        system_prompt: str = SYSTEM_PROMPT,
    ) -> None:
        self.llm_client = llm_client or LLMClient()
        self.memory = memory or ConversationMemory()
        self.tool = tool or LocalTravelTipsTool()
        self.system_prompt = system_prompt

    def run(self, user_input: str) -> str:
        user_message = self.build_user_message(user_input)
        self.memory.add_user_message(user_message)
        answer = self.llm_client.chat(self.memory.build_messages(self.system_prompt))
        self.memory.add_assistant_message(answer)
        return answer

    def build_user_message(self, user_input: str) -> str:
        city = self.extract_city(user_input)
        logger.info("Tool Call: %s(city='%s')", self.tool.function_name, city)

        result = self.tool.run(city)
        logger.info("Observe: %s", result.content)

        tool_record = {
            "tool_name": result.tool_name,
            "tool_function": result.tool_function,
            "tool_input": result.tool_input,
            "tool_action": result.tool_action,
            "tool_result": result.content,
        }
        return (
            f"{user_input}\n\n"
            "本轮必须调用的 Tool 已经执行。请在回答开头说明 Tool 使用情况：\n"
            f"{json.dumps(tool_record, ensure_ascii=False, indent=2)}"
        )

    def extract_city(self, text: str) -> str:
        for city in LOCAL_TRAVEL_TIPS:
            if city in text:
                return city
        return "未识别城市"


def main() -> None:
    print("TravelPlanAgent v0.5 tool1：只调用 local_function")
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

        print(f"TravelPlanAgent：{agent.run(user_input)}")


if __name__ == "__main__":
    main()
