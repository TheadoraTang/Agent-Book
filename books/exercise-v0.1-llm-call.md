# TravelPlanAgent v0.1 练习：补全 LLM 调用

这一题只练一件事：把已经整理好的 `messages` 发给模型，并取回模型回复。

`base_url`、`api_key`、`model_name` 这些基础配置已经放在 `LLMConfig` 里了，这道题不需要你修改它们。你只需要补全 `LLMClient.chat()` 里被挖掉的调用代码。

```sbs-exercise
title: 补全 TravelPlanAgent v0.1 的 LLM 调用
kind: completion
language: python
judgeMode: runSuccess
description: |
  补全 LLMClient.chat() 中的 LLM 调用环节。

  已经保留的内容：
  - LLMConfig 里的 base_url、api_key、model_name 等基础配置；
  - TravelPlanAgent 把用户输入整理成 messages 的流程；
  - OpenAI 客户端的创建方法 _get_openai_client()。

  你只需要在 BEGIN_SOLUTION 和 END_SOLUTION 之间完成三步：
  1. 取得 OpenAI 客户端；
  2. 调用 chat.completions.create(...)；
  3. 返回模型回复的文本内容。
code: |
  from openai import OpenAI


  class LLMConfig:
      """
      集中保存模型调用需要的配置：base_url、api_key、model_name。
      """
      base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
      api_key: str = "sk-5e9208884c9a4d0b848b81aa69507c5b"
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
              return "请先把 LLMConfig 里的 api_key 替换成真实密钥。"

          try:
              # BEGIN_SOLUTION
              pass
              # END_SOLUTION
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

```
