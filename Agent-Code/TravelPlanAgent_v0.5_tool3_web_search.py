import json
import logging
import socket
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from openai import OpenAI


logger = logging.getLogger("travel_agent")
logger.setLevel(logging.INFO)
logger.handlers.clear()
handler = logging.StreamHandler(sys.stderr)
handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
logger.addHandler(handler)
logger.propagate = False


SEARCH_API_URL = "https://searchfree.site/api/search"

SYSTEM_PROMPT = """
你是 TravelPlanAgent，一个友好、可靠、适合初学者理解的旅行规划助手。
本版本只演示一种 Tool：web_search。
每一轮对话都已经调用 web_search，并把工具结果提供给你。
你的回答必须说明：
1. 本轮使用了哪个 Tool。
2. 这个 Tool 做了什么事。
3. 这个 Tool 返回了什么信息。
然后再给出旅行建议和提醒。
"""

SUPPORTED_CITIES = ["上海", "北京", "杭州", "成都", "广州"]


@dataclass
class LLMConfig:
    """集中管理模型调用参数，对应课程中的 API 调用基础。"""

    base_url: str = "https://chat.ecnu.edu.cn/open/api/v1"
    api_key: str = "sk-4b905783f8ab4fed9f7c1879aaf2ae58"
    model_name: str = "ecnu-max"
    temperature: float = 0.7
    timeout_seconds: int = 300


class LLMClient:
    """
    统一封装 LLM 请求。
    """

    def __init__(self, config: LLMConfig | None = None, openai_client: object | None = None) -> None:
        self.config = config or LLMConfig()
        self.openai_client = openai_client

    def chat(self, messages: list[dict[str, str]], temperature: float | None = None) -> str:
        if self.config.api_key in ["", "YOUR_API_KEY"]:
            return "请先把 LLMConfig 里的 api_key 替换成真实密钥。"

        for attempt in range(2):
            try:
                openai_client = self._get_openai_client()
                completion = openai_client.chat.completions.create(
                    model=self.config.model_name,
                    messages=messages,
                    temperature=self.config.temperature if temperature is None else temperature,
                )
                return completion.choices[0].message.content or ""
            except (socket.timeout, TimeoutError) as error:
                if attempt == 0:
                    logger.warning("Error: 模型调用超时，等待 2 秒后重试一次...")
                    time.sleep(2)
                    continue
                return f"调用模型超时：{error}。可以稍后重试，或减少搜索结果数量。"
            except Exception as error:
                return f"调用模型时出现错误：{error}"

        return "调用模型时出现错误：未知错误。"

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


class WebSearchTool(BaseTool):
    """
    调用 searchfree.site 获取目的地攻略、景点和路线信息。
    """

    name = "web_search"
    function_name = "web_search_travel_guide"
    description = "通过搜索获取目的地攻略、景点和路线信息。"

    def run(self, query_text: str) -> ToolResult:
        query = f"{query_text} 旅游攻略 景点 路线"
        payload = {
            "query": query,
            "max_results": 3,
        }
        request = urllib.request.Request(
            SEARCH_API_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "User-Agent": "TravelPlanAgent/0.4",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="ignore")
            content = f"web_search API 请求失败，状态码：{error.code}\n{detail}"
            return ToolResult(self.name, self.function_name, query_text, self.description, content)
        except Exception as error:
            content = f"web_search 调用失败：{error}"
            return ToolResult(self.name, self.function_name, query_text, self.description, content)

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
        content = "\n".join(lines)
        return ToolResult(self.name, self.function_name, query_text, self.description, content)


class TravelPlanAgent:
    """
    TravelPlanAgent v0.5 tool3：
    每轮只调用 web_search。
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
        self.tool = tool or WebSearchTool()
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
        query_text = city if city != "未识别城市" else user_input
        result = self.tool.run(query_text)
        tool_record = self.build_tool_record(result)

        logger.info("Tool Call: %s(query_text='%s')", result.tool_function, result.tool_input)
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
        for city in SUPPORTED_CITIES:
            if city in text:
                return city
        return "未识别城市"


def main() -> None:
    print("TravelPlanAgent v0.5 tool3：只调用 web_search")
    print("这个版本每轮都会调用 web_search_travel_guide。")
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
