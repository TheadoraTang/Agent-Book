from __future__ import annotations

from dataclasses import dataclass, field
from openai import OpenAI


SYSTEM_PROMPT = """
你是 TravelPlanAgent，一个友好、可靠、适合初学者理解的旅行规划助手。
你可以利用最近几轮对话中的目的地、日期、预算、同行人和偏好来继续回答。
你的回答需要满足：
1. 优先围绕旅行目的地、时间、预算、交通、天气、景点和注意事项展开。
2. 语气清晰、亲切，不夸大，不编造无法确定的信息。
3. 输出尽量包含三个部分：建议、理由、提醒。
"""


@dataclass
class LLMConfig:
    """集中管理模型调用参数，对应课程中的 API 调用基础。"""

    base_url: str = "https://chat.ecnu.edu.cn/open/api/v1"
    api_key: str = ""
    model_name: str = "ecnu-plus"
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
    """
    保存最近几轮对话的上下文，帮助模型理解连续对话。
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
        max_messages = self.max_rounds * 2
        self.messages = self.messages[-max_messages:]


class TravelPlanAgent:
    """
    TravelPlanAgent v0.4：支持最近 5 轮上下文记忆。
    重点观察 messages 如何随着 user 和 assistant 对话逐步增长。
    """

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        memory: ConversationMemory | None = None,
        system_prompt: str = SYSTEM_PROMPT,
    ) -> None:
        self.llm_client = llm_client or LLMClient()
        self.memory = memory or ConversationMemory()
        self.system_prompt = system_prompt

    def run(self, user_input: str) -> str:
        self.memory.add_user_message(user_input)
        messages = self.memory.build_messages(self.system_prompt)
        answer = self.llm_client.chat(messages)
        self.memory.add_assistant_message(answer)
        return answer


def main() -> None:
    print("TravelPlanAgent v0.4：支持最近 5 轮上下文记忆")
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
