# 第7章 Tool Calling（工具调用）

## 本章摘要

### 新概念

- **Tool Calling**：让 Agent 调用函数、API 或搜索，把模型自己不知道的信息拿回来。
- **本地函数 Tool**：把普通 Python 函数包装成 Agent 能用的工具。
- **外部 API Tool**：通过网络请求拿到天气这类实时信息。
- **ToolResult**：把工具是否成功、返回了什么、哪里出错整理成统一格式。

### 产品功能增量

- 本章会把 TravelPlanAgent 升级到 v0.5。它能查本地提醒、实时天气和网页资料，再带着工具结果回答用户。


## 对 TravelPlanAgent 来说意味着什么

到现在为止，TravelPlanAgent 已经能调用 LLM、带着 Prompt 工作、按固定格式输出，也能记住最近几轮对话。

但它还有一个明显短板：它只能根据自己看到的上下文回答。

如果用户问：

```text
上海现在天气怎么样？适合步行吗？
```

模型不能靠训练时的旧知识回答。天气每天都在变，旧知识再流畅也不可靠。

如果用户问：

```text
杭州有什么本地出行提醒？
```

我们也许已经在代码里整理了一份小贴士，但模型不会自动知道这些内容，除非程序把它取出来交给模型。

所以第 7 章会让 TravelPlanAgent 升级到 v0.5：

```text
能够调用工具，获取外部信息，再结合工具结果回答
```

这一章先从最简单的本地函数开始，再看天气 API 和 web_search。工具听起来高级，拆开来看，其实就是程序帮模型做一件它自己做不了、或者不该凭空猜的事。


## 本章目标

读完本章，你应该能理解：

1. Tool Calling 为什么是 Agent 连接真实世界的重要方式。
2. 本地函数、外部 API 和 web_search 分别适合解决什么问题。
3. Tool 的输入参数、执行过程和返回结果应怎样组织。
4. 为什么工具结果要重新放回上下文，交给模型继续生成回答。
5. `BaseTool`、具体 Tool 类和 `ToolResult` 如何让代码保持统一结构。

## 为什么 Agent 需要工具

LLM 很擅长理解语言和组织回答，但不能把它当成万能信息源。

它通常不知道：

- 当前天气；
- 实时火车票价；
- 某一些商店的最新营业时间；
- 一些外部数据和信息源等等

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
```python
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

最简单的 Tool 往往不是复杂 API，而是一个自己写的普通 Python 函数。

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
height: 500px
```

### 本地函数 Tool 示例

<!-- sbs-code -->

```python

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
height: 500px
```

### 天气 API Tool 完整代码
<!-- sbs-code -->

```python

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

不是所有模型都已经接入内嵌 web_search。

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

这里使用这个搜索服务，主要是为了让你看见 web_search 的完整结构。以后如果换成其他搜索 API，或者换成模型平台内嵌的 `web_search`，核心思路仍然一致：

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
height: 500px
```

### web_search Tool 完整代码

<!-- sbs-code -->

```python

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

```text
{
  "tool_name": "local_function",
  "tool_function": "get_local_travel_tips",
  "tool_input": "杭州",
  "tool_action": "读取代码内置的城市出行提醒和本地小贴士。",
  "tool_result": "杭州西湖适合步行和骑行，但节假日湖滨、断桥一带人流密集。"
}
```

再把它拼接到 user message：

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

```text
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

这一章我们让 TravelPlanAgent 第一次伸手去拿外部信息。

它新增了三类工具：

- 本地函数：读取代码里准备好的旅行小贴士；
- 外部 API：查询实时天气；
- web_search：搜索攻略、景点和路线资料。

你也看到了，Tool Calling 不只是“能调函数”这么简单。一个好工具还要有清楚的输入、统一的输出、失败处理，以及能放回上下文的 Observation。

v0.5 里我们先用关键词规则选择工具。下一章会更进一步：让模型参与判断什么时候该 Think，什么时候该 Act，拿到结果后又该怎样 Observe。
