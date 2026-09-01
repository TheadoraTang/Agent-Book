from openai import OpenAI


class LLMConfig:
    """
    集中保存模型调用需要的配置：base_url、api_key、model_name。
    """
    base_url: str = "http://106.15.43.196:3001/v1"
    api_key: str = "sk-mpXFJjd8BJm4ReqMmHScLqQZ9oCxjp29p3Ku2XYrEXGvFvGe"
    model_name: str = "deepseek-v4-pro"
    temperature: float = 0.7
    timeout_seconds: int = 300


class LLMClient:
    """
    用于封装 LLM API 请求。
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

        self.openai_client = OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            timeout=self.config.timeout_seconds,
        )
        return self.openai_client


class TravelPlanAgent:
    """
    TravelPlanAgent v0.1：最小单轮对话 Agent。
    1. 接收输入；
    2. 把输入组装成 messages；
    3. 调用 LLMClient 得到模型回复。
    """

    def __init__(self, llm_client: LLMClient | None = None) -> None:
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
