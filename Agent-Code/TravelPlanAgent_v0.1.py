import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Optional


@dataclass
class LLMConfig:
    """
    集中保存模型调用需要的配置。：base_url、api_key、model_name。
    """
    base_url: str = ""
    api_key: str = ""
    model_name: str = ""
    temperature: float = 0.7
    timeout_seconds: int = 60

    @property
    def chat_completions_url(self) -> str:
        return f"{self.base_url}/chat/completions"


class LLMClient:
    """
    用于封装 LLM API 请求。
    """
    def __init__(self, config: Optional[LLMConfig] = None) -> None:
        self.config = config or LLMConfig()

    def chat(self, messages: list[dict[str, str]], temperature: Optional[float] = None) -> str:
        if self.config.api_key in ["", "YOUR_API_KEY"]:
            return "请先把 LLMConfig 里的 api_key 替换成真实密钥。"

        payload = {
            "model": self.config.model_name,
            "messages": messages,
            "temperature": self.config.temperature if temperature is None else temperature,
        }
        request = urllib.request.Request(
            self.config.chat_completions_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.config.api_key}",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
                result = json.loads(response.read().decode("utf-8"))
                return result["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="ignore")
            return f"API 请求失败，状态码：{error.code}\n{detail}"
        except Exception as error:
            return f"调用模型时出现错误：{error}"


class TravelPlanAgent:
    """
    TravelPlanAgent v0.1：最小单轮对话 Agent。
    1. 接收输入；
    2. 把输入组装成 messages；
    3. 调用 LLMClient 得到模型回复。
    """

    def __init__(self, llm_client: Optional[LLMClient] = None) -> None:
        self.llm_client = llm_client or LLMClient()

    def run(self, user_input: str) -> str:
        messages = self.build_messages(user_input)
        return self.llm_client.chat(messages)

    def build_messages(self, user_input: str) -> list[dict[str, str]]:
        return [{"role": "user", "content": user_input}]


def main() -> None:
    print("TravelPlanAgent v0.1：学会调用 LLM 进行单轮对话")

    try:
        user_input = input("\n你：").strip()
    except EOFError:
        print("\nTravelPlanAgent：没有收到输入。")
        return

    agent = TravelPlanAgent()
    answer = agent.run(user_input)
    print(f"TravelPlanAgent：{answer}")


if __name__ == "__main__":
    main()
