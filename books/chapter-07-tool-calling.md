# 第7章 Tool Calling（工具调用）

## 对 TravelPlanAgent 来说意味着什么

到目前为止，TravelPlanAgent 已经能够：

- 调用 LLM；
- 带着清晰的 Prompt 工作；
- 按固定格式输出；
- 记住最近 5 轮对话。

但它仍然有一个明显局限：**它只能依赖模型已经看到的上下文回答。**

如果用户问：

```text
上海现在天气怎么样？适合步行吗？
```

模型不能只凭训练时学到的知识回答。天气会变化，旧知识不可靠。

如果用户问：

```text
杭州有什么本地出行提醒？
```

我们也许已经在程序里整理了一份小贴士，但模型不会自动知道这些内容。

如果用户问：

```text
帮我搜索一下成都的旅行攻略和路线。
```

模型同样需要连接外部搜索能力，才能获取新的资料。

因此，第 7 章会让 TravelPlanAgent 升级到 v0.5：

```text
能够调用工具，获取外部信息，再结合工具结果回答
```

本章会分三步学习三种常见 Tool：

| 学习顺序 | Tool 类型 
| --- | --- | 
| Tool 1 | 自己写的本地函数 
| Tool 2 | 调用外部天气 API 
| Tool 3 | 使用 web_search 功能 

最后，我们会把三个工具合并进：

```text
Agent-Code/TravelPlanAgent_v0.5.py
```

---

## 本章目标

读完本章，你应该能理解：

1. Tool Calling 为什么是 Agent 连接真实世界的重要方式。
2. 本地函数、外部 API 和 web_search 分别适合解决什么问题。
3. Tool 的输入参数、执行过程和返回结果应怎样组织。
4. 为什么工具结果要重新放回上下文，交给模型继续生成回答。
5. `BaseTool`、具体 Tool 类和 `ToolResult` 如何让代码保持统一结构。

## 为什么 Agent 需要工具

LLM 很擅长理解语言和组织回答，但它并不是万能信息源。

它通常不知道：

- 当前天气；
- 实时票价；
- 最新营业时间；
- 你电脑里的本地文件；
- 程序内部保存的数据；
- 搜索引擎刚刚返回的结果；
- 数据库中的订单状态。

如果 Agent 需要处理这些信息，就要通过 Tool 连接外部世界。

我们可以把关系理解成：

```text
LLM 负责理解和判断
Tool 负责查询和执行
Observation 负责把真实结果带回上下文
```

例如：

```text
用户：上海现在天气怎么样？
Agent：这个问题需要实时数据。
Tool：调用 weather_api。
Observation：上海当前多云，气温 24 摄氏度。
Agent：根据天气结果，给出步行和穿衣建议。
```

这比模型直接猜一个天气答案可靠得多。

## Tool Calling 的共同骨架

三个工具虽然来源不同，但工作流程很像：

```text
用户输入
→ 提取工具需要的参数
→ 执行工具
→ 得到 Tool Result
→ 把 Tool Result 放回 Prompt
→ 调用 LLM 组织最终回答
```

在代码里，我们先用统一的 `ToolResult` 表达工具结果：

<!-- sbs-code -->

```python
@dataclass
class ToolResult:
    tool_name: str
    tool_function: str
    tool_input: str
    tool_action: str
    content: str
```

每个字段都有明确作用：

| 字段 | 含义 | 示例 |
| --- | --- | --- |
| `tool_name` | 工具名称 | `weather_api` |
| `tool_function` | 实际执行的方法 | `get_weather_from_api` |
| `tool_input` | 工具接收的输入 | `上海` |
| `tool_action` | 工具做了什么 | 调用天气 API 获取目的地当前天气 |
| `content` | 工具返回内容 | 上海当前多云，气温 24 摄氏度 |

然后，我们为工具定义一个统一接口：

<!-- sbs-code -->

```python
class BaseTool:
    name: str = "base_tool"
    function_name: str = "run"
    description: str = "基础工具"

    def run(self, tool_input: str) -> ToolResult:
        raise NotImplementedError
```

`BaseTool` 不负责完成具体任务。它更像一个约定：

> 不管后面加入什么工具，都应该提供一个 `run()` 方法，并返回统一格式的 `ToolResult`。

这样，TravelPlanAgent 不需要为每个工具写一套完全不同的处理逻辑。

## Tool 1：自己写的本地函数

最简单的Tool，不是复杂的 API，而是一个自己写的普通 Python 函数。

示例中在程序中保存了一些简单的小贴士示例
```python
LOCAL_TRAVEL_TIPS = {
    "上海": "本地小贴士：上海城市交通方便，地铁适合串联人民广场、南京东路、外滩、陆家嘴。外滩夜景人流较多，建议错峰前往。",
    "北京": "本地小贴士：北京景点预约要求较多，故宫、国家博物馆等建议提前预约。城市尺度大，跨区游玩要给交通留足时间。",
    "杭州": "本地小贴士：杭州西湖适合步行和骑行，但节假日湖滨、断桥一带人流密集。灵隐寺适合安排在上午。",
    "成都": "本地小贴士：成都适合慢节奏旅行。熊猫基地建议上午早点去，宽窄巷子、锦里更适合体验氛围而不是安排过满。",
    "广州": "本地小贴士：广州早茶适合上午体验，老城区适合步行探索。夏季闷热且阵雨多，行程最好留室内备选。",
}
```

对应工具类是：

<!-- sbs-code -->

```python
class LocalTravelTipsTool(BaseTool):
    name = "local_function"
    function_name = "get_local_travel_tips"
    description = "读取代码内置的城市出行提醒和本地小贴士。"

    def run(self, city: str) -> ToolResult:
        content = LOCAL_TRAVEL_TIPS.get(
            city,
            "没有识别到支持城市。"
        )
        return ToolResult(
            self.name,
            self.function_name,
            city,
            self.description,
            content,
        )
```

这个工具没有联网，也没有调用复杂服务。

它只做三件事：

1. 接收城市名；
2. 从本地字典查找小贴士；
3. 返回统一格式的 `ToolResult`。

用户输入：

```text
我去杭州玩两天，有什么本地出行提醒？
```

程序会执行：

```text
get_local_travel_tips(city='杭州')
```

工具结果会被放回 Prompt，模型再基于这些真实内容组织回答。

### 本地函数适合什么场景

本地函数并不“低级”。它是非常实用的工具形式。

适合用本地函数处理：

- 固定业务规则；
- 本地配置；
- 简单计算；
- 日期格式转换；
- 读取小型字典；
- 调用项目内部已有函数。

Tool 不一定要联网。只要它能帮助 Agent 完成任务，就可以成为 Tool。

### 本地函数 Tool 完整调用动画

下面的动画可以看到城市参数怎样从用户问题中被提取出来，如何进入本地字典，查询结果又怎样回填到 Prompt。

```sbs-iframe
src: assets/local-function-tool-demo.html
title: 本地函数 Tool 完整调用动画
height: 620px
```

### 本地函数 Tool 示例

<!-- sbs-code -->

```python
import json
from dataclasses import dataclass, field
from openai import OpenAI


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


class LLMConfig:
    """集中管理模型调用参数，对应课程中的 API 调用基础。"""

    base_url: str = "https://chat.ecnu.edu.cn/open/api/v1"
    api_key: str = ""
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
    """保存最近几轮对话。"""

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
    """工具调用结果的统一格式。"""

    tool_name: str
    tool_function: str
    tool_input: str
    tool_action: str
    content: str


class BaseTool:
    """所有工具的统一接口。"""

    name: str = "base_tool"
    function_name: str = "run"
    description: str = "基础工具"

    def run(self, tool_input: str) -> ToolResult:
        raise NotImplementedError


class LocalTravelTipsTool(BaseTool):
    """读取代码内置的城市出行提醒和本地小贴士。"""

    name = "local_function"
    function_name = "get_local_travel_tips"
    description = "读取代码内置的城市出行提醒和本地小贴士。"

    def run(self, city: str) -> ToolResult:
        content = LOCAL_TRAVEL_TIPS.get(city, f"没有识别到支持城市。当前支持：{', '.join(LOCAL_TRAVEL_TIPS)}。")
        return ToolResult(self.name, self.function_name, city, self.description, content)


class TravelPlanAgent:
    """TravelPlanAgent v0.5 tool1：每轮只调用 local_function。"""

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        memory: ConversationMemory | None = None,
        tool: BaseTool | None = None,
        system_prompt: str = SYSTEM_PROMPT,
    ) -> None:
        self.llm_client = llm_client or LLMClient()
        self.memory = memory or ConversationMemory()
        self.tool = tool or LocalTravelTipsTool()
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

        print(f"[工具调用] {result.tool_function}(city='{result.tool_input}')")
        print(f"[工具作用] {result.tool_action}")
        print(f"[工具结果] {result.content}")

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
        for city in LOCAL_TRAVEL_TIPS:
            if city in text:
                return city
        return "未识别城市"


def main() -> None:
    print("TravelPlanAgent v0.5 tool1：只调用 local_function")
    print("这个版本每轮都会调用 get_local_travel_tips。")
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

```


## Tool 2：调用外部 API

本地函数能处理程序内部信息，但天气这类实时或者外部的数据需要外部服务。

第二个示例是天气 API，对应工具类是：

```text
WeatherApiTool
```

它会调用 [Open-Meteo](https://open-meteo.com/) 获取目的地当前天气。

调用 API 的过程可以拆成几步：

1. 从用户输入中提取城市；
2. 根据城市找到经纬度；
3. 拼接天气 API 请求参数；
4. 发出 HTTP 请求；
5. 读取 JSON 返回值；
6. 整理成适合模型阅读的文本。

核心代码如下：

<!-- sbs-code -->

```python
query = urllib.parse.urlencode(
    {
        "latitude": coordinates["latitude"],
        "longitude": coordinates["longitude"],
        "current": "temperature_2m,relative_humidity_2m,precipitation,weather_code,wind_speed_10m",
        "timezone": "Asia/Shanghai",
    }
)
url = f"https://api.open-meteo.com/v1/forecast?{query}"

with urllib.request.urlopen(url, timeout=20) as response:
    data = json.loads(response.read().decode("utf-8"))
```

天气 API 返回的是结构化数据。程序会整理出：

- 天气状态；
- 温度；
- 相对湿度；
- 降水量；
- 风速。

然后再把结果交给模型。

例如：

```text
上海当前天气：多云，气温 24 摄氏度，相对湿度 68%，降水量 0 mm。
```

### 调用 API 时要考虑失败

外部 API 不一定每次都成功。

可能失败的原因包括：

- 网络不通；
- 超时；
- 城市参数不支持；
- API 服务暂时不可用；
- 返回格式变化；
- 鉴权失败；
- 调用额度不足。

因此，工具不能只写“成功路径”。它还需要返回清晰的错误信息：

```python
except Exception as error:
    content = f"天气 API 调用失败：{error}"
```

这样，即使天气 API 失败，Agent 也能知道发生了什么，而不是直接崩溃。

### 天气 API Tool 完整调用动画

这个动画会展示一次天气查询怎样从城市名出发，经过经纬度转换、HTTP 请求、JSON 返回值解析，最终成为模型可以阅读的 Observation。

```sbs-iframe
src: assets/weather-api-tool-demo.html
title: 天气 API Tool 完整调用动画
height: 620px
```

### 天气 API Tool 完整代码
<!-- sbs-code -->

```python
import json
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from openai import OpenAI


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
    """保存最近几轮对话。"""

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
    """工具调用结果的统一格式。"""

    tool_name: str
    tool_function: str
    tool_input: str
    tool_action: str
    content: str


class BaseTool:
    """所有工具的统一接口。"""

    name: str = "base_tool"
    function_name: str = "run"
    description: str = "基础工具"

    def run(self, tool_input: str) -> ToolResult:
        raise NotImplementedError


class WeatherApiTool(BaseTool):
    """调用 Open-Meteo 天气 API 获取目的地当前天气。"""

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


class TravelPlanAgent:
    """TravelPlanAgent v0.5 tool2：每轮只调用 weather_api。"""

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

        print(f"[工具调用] {result.tool_function}(city='{result.tool_input}')")
        print(f"[工具作用] {result.tool_action}")
        print(f"[工具结果] {result.content}")

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

```


## Tool 3：使用 web_search

有些问题很难靠一个固定 API 回答。

例如：

```text
成都有哪些值得参考的旅行攻略？
杭州最近适合怎样安排路线？
上海有哪些城市漫步区域？
```

这些问题更适合交给搜索工具，对应工具类是：

```text
WebSearchTool
```

### 有些模型已经内嵌了 web_search

在真实开发中，web_search 不一定需要我们从头实现。

许多模型平台已经提供了内嵌的联网搜索能力。开发者只需要在请求中注册 `web_search` 工具，模型就可以在需要最新信息时主动搜索网页，再根据搜索结果生成回答。

以 OpenAI Responses API 为例，可以这样注册内嵌 web_search：

<!-- sbs-code -->

```python
from openai import OpenAI

client = OpenAI()

response = client.responses.create(
    model="gpt-5.5",
    tools=[
        {"type": "web_search"}
    ],
    input="帮我搜索成都旅行攻略，并给出适合三日游的路线建议。",
)

print(response.output_text)
```

这段代码里，我们没有自己调用某个搜索网站，也没有手动解析标题、摘要和链接。模型平台会在内部完成一部分工作：

```text
模型判断需要搜索
→ 调用内嵌 web_search
→ 获取网页资料
→ 根据资料生成回答
→ 返回引用来源
```

根据 OpenAI 官方文档，内嵌 web search 可以让模型访问较新的互联网信息，并在回答中提供来源引用。不同模型、接口和平台的支持方式可能不同，使用前应查看对应平台的最新文档：

[OpenAI Web Search 官方文档](https://platform.openai.com/docs/guides/tools-web-search)

### 为什么课程代码还要自己模拟 web_search

并不是所有模型都已经接入内嵌 web_search。

有些模型只能完成普通对话；有些平台虽然提供兼容接口，但没有开放联网搜索；还有一些教学环境希望学生先看清楚“搜索是怎样发生的”，而不是把整个过程藏在模型平台内部。

因此，本课程使用一种更容易观察和迁移的方式模拟 web_search：

```text
TravelPlanAgent
→ 主动调用一个独立的搜索 API
→ 获取标题、摘要和链接
→ 整理成 ToolResult
→ 把搜索结果回填到 Prompt
→ 交给 LLM 生成最终回答
```

对应代码中，搜索 API 地址是：

```python
SEARCH_API_URL = "https://searchfree.site/api/search"
```

这种写法的目的不是说所有项目都必须使用这个搜索服务，而是让你看见 web_search 的完整结构。以后如果换成其他搜索 API，或者换成模型平台内嵌的 `web_search`，核心思路仍然一致：

> Agent 需要获取网页资料，再根据搜索结果继续完成任务。

这个工具会把用户问题整理成搜索词，例如：

```text
成都 旅游攻略 景点 路线
```

然后调用 web_search 服务，获取前几条搜索结果，并整理出：

- 标题；
- 摘要；
- 链接；
- 可选的 AI 摘要。

核心过程如下：

<!-- sbs-code -->

```python
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
```

搜索结果会被整理成一段文本，再交给 LLM 总结。

### web_search 和模型已有知识有什么区别

模型已有知识来自训练数据，可能过时，也不一定覆盖具体问题。

web_search 的作用是：

- 获取更新的信息；
- 找到可追溯链接；
- 扩展模型当前没有看到的资料；
- 为后续回答提供参考依据。

不过，搜索结果也不能盲目信任。

Agent 需要注意：

- 搜索摘要可能不完整；
- 网页信息可能过时；
- 不同来源可能互相矛盾；
- 涉及票价、营业时间、预约规则时应提醒用户再次核验；
- 最好保留来源链接。

搜索工具让 Agent 看得更远，但不代表每条搜索结果都是真理。

### web_search Tool 完整调用动画

这个动画会展示用户问题怎样变成搜索词，搜索服务如何返回标题、摘要和链接，以及这些结果如何进入 Prompt。

```sbs-iframe
src: assets/web-search-tool-demo.html
title: web_search Tool 完整调用动画
height: 620px
```

### web_search Tool 完整代码

<!-- sbs-code -->

```python
import json
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from openai import OpenAI


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

class LLMConfig:
    """集中管理模型调用参数，对应课程中的 API 调用基础。"""

    base_url: str = "https://chat.ecnu.edu.cn/open/api/v1"
    api_key: str = ""
    model_name: str = "ecnu-plus"
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
            except ImportError:
                return "当前环境缺少 openai 包，请先运行：pip install openai"
            except (socket.timeout, TimeoutError) as error:
                if attempt == 0:
                    print("[模型调用超时] 等待 2 秒后重试一次...")
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

        print(f"[工具调用] {result.tool_function}(query_text='{result.tool_input}')")
        print(f"[工具作用] {result.tool_action}")
        print(f"[工具结果] {result.content}")

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

```


## 三种 Tool 的对比

| Tool 类型 | 数据来源 | 优点 | 需要注意的问题 | 示例 |
| --- | --- | --- | --- | --- |
| 本地函数 | 程序内部 | 简单、稳定、容易测试 | 数据需要自己维护 | 城市小贴士、预算计算 |
| 外部 API | 专门服务 | 结构清晰，适合实时数据 | 网络、超时、参数、额度 | 天气、地图、汇率 |
| web_search | 搜索服务 | 信息范围广，适合开放问题 | 结果质量不稳定，需要核验 | 攻略、景点、路线资料 |

学习 Tool Calling 时，可以先问自己：

```text
这个问题最适合怎样获取答案？
```

如果是简单的计算或者利用已知的内容，可以采用自建函数。

如果有需要调用外部的数据服务以及其他外部数据，用 API。

如果问题比较开放，需要查资料，利用Agent的能力进行汇总和汇报，用 web_search。

## Tool Result 为什么要放回 Prompt

工具执行完以后，LLM 并不会自动知道结果。

程序需要明确把 Observation 放回上下文。

例如本地函数返回：

```text
杭州西湖适合步行和骑行，但节假日湖滨、断桥一带人流密集。
```

我们会把工具记录整理成 JSON：

```json
{
  "tool_name": "local_function",
  "tool_function": "get_local_travel_tips",
  "tool_input": "杭州",
  "tool_action": "读取代码内置的城市出行提醒和本地小贴士。",
  "tool_result": "杭州西湖适合步行和骑行，但节假日湖滨、断桥一带人流密集。"
}
```

再把它拼接到 user message：

<!-- sbs-code -->

```python
return (
    f"{user_input}\n\n"
    "本轮必须调用的 Tool 已经执行。请在回答开头说明 Tool 使用情况：\n"
    f"{json.dumps(tool_record, ensure_ascii=False, indent=2)}"
)
```

模型看到这段结果后，才能基于它回答。

这个过程叫 Observation，也就是“观察工具结果”。

## Tool Schema：让模型知道工具怎么用

工具不仅要能运行，还需要有清楚的说明。

例如天气工具可以描述成：

```json
{
  "name": "weather_api",
  "description": "查询目的地当前天气，适合回答气温、降雨、穿衣和出行风险。",
  "parameters": {
    "city": {
      "type": "string",
      "description": "要查询天气的城市，例如上海、杭州、成都。"
    }
  }
}
```

这就是一种简化的 Tool Schema。

它告诉模型：

- 工具叫什么；
- 工具能做什么；
- 需要哪些参数；
- 每个参数是什么类型；
- 参数应该怎样填写。

在本章代码里，我们先用 `name`、`function_name`、`description` 和 `run()` 表达这些信息。

第 8 章会进一步升级：把工具定义注册给模型，让模型在 Think 阶段自主判断该调用哪个工具。

## 本章实战：从单一工具到综合工具

建议按下面顺序运行代码。

### 第一步：书写本地函数 Tool

自行补充代码中缺失的部分，实现一个本地函数的 Tool

TODO：待补充缺失代码

检验时可以输入：

```text
我去杭州玩两天，有什么本地提醒？
```

观察终端的结果

### 第二步：运行天气 API Tool

自行补充代码中缺失的部分，实现调用其他 API 的 Tool

TODO：待补充缺失代码

检验时可以输入：

```text
上海现在天气怎么样？
```

观察天气 API 返回的实时数据，以及模型如何基于数据调整建议。

### 第三步：运行 web_search Tool

自行补充代码中缺失的部分，实现 web_search Tool

TODO：待补充缺失代码

检验时可以输入：

```text
帮我搜索成都旅行攻略。
```

观察搜索结果中的标题、摘要和链接如何进入 Prompt。

**这里要特别说明**：

TravelPlanAgent v0.5 已经能使用工具，但它还不是“完全自主选择工具”的最终版本。

本章综合版主要使用关键词规则选择工具：

```python
if any(keyword in user_input for keyword in WEATHER_KEYWORDS):
    tools_used.append("weather_api")
```

这样写的好处是：

- 容易理解；
- 容易调试；
- 行为稳定；
- 能先看清 Tool Calling 的基础链路。

下一章，我们会进一步学习 Agent 的工作循环：

```text
Think → Act → Observe → Think Again
```

到那时，模型会参与判断：

- 当前缺少什么信息？
- 应该调用哪个工具？
- 工具结果是否已经足够？
- 是否还需要继续行动？

## 小结

这一章我们给 TravelPlanAgent 增加了三种工具：

- 本地函数：读取代码内部的小贴士；
- 外部 API：查询实时天气；
- web_search：搜索攻略、景点和路线资料。

我们还理解了：

- Tool 是 Agent 连接真实世界的重要方式；
- 不同问题适合不同类型的 Tool；
- Tool 应该有统一输入、输出和失败处理；
- `ToolResult` 让工具返回结果保持一致；
- 工具结果需要作为 Observation 放回 Prompt；
- v0.5 先用关键词规则选择工具，第 8 章再让模型参与自主决策。

下一章，我们会把这些工具真正放进 Agent 的工作循环。
