import json
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-m3")

BASE_URL = "https://chat.ecnu.edu.cn/open/api/v1"
API_KEY = ""
MODEL_NAME = "ecnu-plus"
CHAT_COMPLETIONS_URL = f"{BASE_URL}/chat/completions"
MAX_HISTORY_ROUNDS = 5
LLM_TIMEOUT_SECONDS = 300
SEARCH_API_URL = "https://searchfree.site/api/search"
MAX_TOOL_ROUNDS = 3



_embedding_model = None
_rag_chunks = None
_rag_embeddings = None

SYSTEM_PROMPT = """
你是 TravelPlanAgent，一个具备工具调用和 RAG 能力的旅行规划助手。
你需要根据用户问题自主决定调用哪些 Tools。

你可以使用四个 Tools：
- get_local_travel_tips：读取代码内置的城市出行提醒。
- get_weather_from_api：调用天气 API 获取实时天气。
- web_search_travel_guide：调用搜索 API 获取网页攻略、景点、路线信息。
- rag_search_travel_knowledge：使用本地 assets/bge-m3 embedding 模型检索课程内置旅游知识库。

当用户询问目的地怎么玩、路线安排、景点建议、雨天安排、亲子/老人/预算等旅行规划问题时，
你应优先调用 rag_search_travel_knowledge 获取本地知识库片段。
如果问题还涉及天气，再额外调用 get_weather_from_api。
如果问题需要最新网页攻略，再额外调用 web_search_travel_guide。

最终回答必须说明：
1. 本轮使用了哪些 Tools。
2. 每个 Tool 做了什么事。
3. RAG 检索到了哪些相关知识，以及这些知识如何影响你的建议。
4. 如果 RAG 或外部工具结果不足，要明确提醒用户补充信息或出行前核验。
"""

LOCAL_TRAVEL_TIPS = {
    "上海": "本地小贴士：上海城市交通方便，地铁适合串联人民广场、南京东路、外滩、陆家嘴。外滩夜景人流较多，建议错峰前往。",
    "北京": "本地小贴士：北京景点预约要求较多，故宫、国家博物馆等建议提前预约。城市尺度大，跨区游玩要给交通留足时间。",
    "杭州": "本地小贴士：杭州西湖适合步行和骑行，但节假日湖滨、断桥一带人流密集。灵隐寺适合安排在上午。",
    "成都": "本地小贴士：成都适合慢节奏旅行。熊猫基地建议上午早点去，宽窄巷子、锦里更适合体验氛围而不是安排过满。",
    "广州": "本地小贴士：广州早茶适合上午体验，老城区适合步行探索。夏季闷热且阵雨多，行程最好留室内备选。",
}

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

TRAVEL_KNOWLEDGE = """
# 上海城市旅行
上海适合第一次到访的经典路线是人民广场、南京东路、外滩、陆家嘴。白天可以看城市建筑和博物馆，晚上适合去外滩或陆家嘴看夜景。喜欢城市文化的游客可以安排武康路、思南路、衡山路、上海博物馆和上海当代艺术博物馆。亲子旅行可以考虑上海科技馆、上海自然博物馆和迪士尼。上海市内公共交通发达，地铁比打车更稳定。

# 北京历史文化旅行
北京适合历史文化主题旅行。经典路线包括天安门、故宫、景山公园、北海公园、什刹海和南锣鼓巷。故宫、国家博物馆等热门景点通常需要提前预约。长城距离市区较远，建议单独安排一天。北京城市尺度大，跨区移动耗时较长，行程不宜过密。

# 杭州西湖与茶文化
杭州适合慢节奏旅行。西湖可以安排断桥、白堤、苏堤、雷峰塔、曲院风荷和湖滨步行区。灵隐寺适合上午前往。喜欢茶文化可以去龙井村、中国茶叶博物馆和满觉陇。雨天可以安排南宋德寿宫遗址博物馆、浙江省博物馆、河坊街和室内茶馆。

# 成都慢旅行与美食
成都适合慢旅行和美食体验。常见路线包括大熊猫繁育研究基地、人民公园、宽窄巷子、武侯祠、锦里和太古里。熊猫基地建议上午早点去。美食可以安排火锅、串串、担担面、钟水饺和蛋烘糕。成都行程要留出喝茶、散步和休息的时间。

# 广州美食与城市文化
广州适合美食和城市文化旅行。常见路线包括陈家祠、沙面、永庆坊、北京路、珠江夜游和广东省博物馆。早茶适合安排在上午。夏季广州闷热且可能阵雨，行程应准备室内备选。广州地铁便利，但老城区步行体验也很好。

# 通用旅行规划原则
旅行规划时，每天不要安排过多景点。城市初访建议每天 2 到 4 个主要点位，并留出交通、排队、用餐和休息时间。亲子、老人同行时要降低步行强度。雨天优先安排博物馆、展览、商场、茶馆和餐饮体验。预算有限时，应优先选择公共交通和集中区域游玩。
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_local_travel_tips",
            "description": "读取代码内置的城市出行提醒，包括交通、预约、人流、节奏和避坑建议。",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "城市名，例如：北京、上海、杭州。"}},
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather_from_api",
            "description": "调用天气 API 获取城市当前天气、气温、湿度、降水和风速。",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "城市名，例如：北京、上海、杭州。"}},
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search_travel_guide",
            "description": "调用搜索 API 搜索目的地旅游攻略、景点、路线和注意事项。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query_text": {"type": "string", "description": "搜索关键词，例如：北京三日游、上海亲子旅行攻略。"}
                },
                "required": ["query_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rag_search_travel_knowledge",
            "description": "使用本地 assets/bge-m3 embedding 模型检索课程内置旅游知识库，返回最相关的知识片段。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "用于 RAG 检索的问题，例如：杭州雨天怎么玩。"},
                    "top_k": {"type": "integer", "description": "返回的知识片段数量，默认 3。"},
                },
                "required": ["query"],
            },
        },
    },
]


def call_llm_response(messages, temperature=0.7, tools=None, tool_choice=None):
    if API_KEY in ["", "YOUR_API_KEY"]:
        return {"role": "assistant", "content": "请先把文件顶部的 API_KEY 替换成真实密钥。"}

    payload = {"model": MODEL_NAME, "messages": messages, "temperature": temperature}
    if tools:
        payload["tools"] = tools
    if tool_choice:
        payload["tool_choice"] = tool_choice

    request = urllib.request.Request(
        CHAT_COMPLETIONS_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"},
        method="POST",
    )

    for attempt in range(2):
        try:
            with urllib.request.urlopen(request, timeout=LLM_TIMEOUT_SECONDS) as response:
                result = json.loads(response.read().decode("utf-8"))
                return result["choices"][0]["message"]
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="ignore")
            return {"role": "assistant", "content": f"API 请求失败，状态码：{error.code}\n{detail}"}
        except (socket.timeout, TimeoutError) as error:
            if attempt == 0:
                print("[模型调用超时] 等待 2 秒后重试一次...")
                time.sleep(2)
                continue
            return {"role": "assistant", "content": f"调用模型超时：{error}。可以稍后重试。"}
        except Exception as error:
            return {"role": "assistant", "content": f"调用模型时出现错误：{error}"}


def load_embedding_model():
    global _embedding_model

    if _embedding_model is not None:
        return _embedding_model

    print("[RAG] 正在加载 bge-m3 embedding 模型...")

    _embedding_model = SentenceTransformer("BAAI/bge-m3")

    return _embedding_model


def embed_texts(texts):
    model = load_embedding_model()
    embeddings = model.encode(
        texts,
        batch_size=8,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(embeddings, dtype=np.float32)


def split_knowledge_into_chunks(text):
    chunks = []
    for block in text.strip().split("\n\n"):
        chunk = block.strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def build_rag_index():
    global _rag_chunks, _rag_embeddings
    if _rag_chunks is not None and _rag_embeddings is not None:
        return _rag_chunks, _rag_embeddings
    _rag_chunks = split_knowledge_into_chunks(TRAVEL_KNOWLEDGE)
    print(f"[RAG] 正在为 {len(_rag_chunks)} 个知识片段生成 embedding...")
    _rag_embeddings = embed_texts(_rag_chunks)
    return _rag_chunks, _rag_embeddings


def rag_search_travel_knowledge(query, top_k=3):
    try:
        top_k = int(top_k)
    except (TypeError, ValueError):
        top_k = 3
    top_k = max(1, min(top_k, 5))

    try:
        chunks, chunk_embeddings = build_rag_index()
        query_embedding = embed_texts([query])[0]
    except Exception as error:
        return f"RAG 检索失败：{error}"

    scores = chunk_embeddings @ query_embedding
    ranked_indices = np.argsort(scores)[::-1][:top_k]

    lines = [
        "RAG 检索工具：rag_search_travel_knowledge",
        f"检索问题：{query}",
        "最相关知识片段：",
    ]
    for rank, index in enumerate(ranked_indices, start=1):
        lines.append(f"{rank}. 相似度：{float(scores[index]):.4f}")
        lines.append(chunks[index])
    return "\n".join(lines)


def get_local_travel_tips(city):
    return LOCAL_TRAVEL_TIPS.get(city, f"没有识别到支持城市。当前支持：{', '.join(LOCAL_TRAVEL_TIPS)}。")


def get_weather_from_api(city):
    coordinates = CITY_COORDINATES.get(city)
    if not coordinates:
        return f"没有识别到支持城市，无法查询天气 API。当前支持：{', '.join(CITY_COORDINATES)}。"

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
        return f"天气 API 调用失败：{error}"

    current = data.get("current", {})
    weather_code = current.get("weather_code")
    weather_text = WEATHER_CODE_MAP.get(weather_code, f"未知天气代码 {weather_code}")
    return (
        f"{city} 当前天气：{weather_text}，"
        f"气温 {current.get('temperature_2m')} 摄氏度，"
        f"相对湿度 {current.get('relative_humidity_2m')}%，"
        f"降水量 {current.get('precipitation')} mm，"
        f"风速 {current.get('wind_speed_10m')} km/h。"
        "数据来自 Open-Meteo 天气 API。"
    )


def web_search_travel_guide(query_text):
    query = f"{query_text} 旅游攻略 景点 路线"
    payload = {"query": query, "max_results": 3}
    request = urllib.request.Request(
        SEARCH_API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "User-Agent": "TravelPlanAgent/0.6"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="ignore")
        return f"web_search API 请求失败，状态码：{error.code}\n{detail}"
    except Exception as error:
        return f"web_search 调用失败：{error}"

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
    return "\n".join(lines)


def parse_tool_arguments(arguments_text):
    try:
        return json.loads(arguments_text or "{}")
    except json.JSONDecodeError:
        return {}


def execute_tool_call(tool_call):
    function_info = tool_call.get("function", {})
    function_name = function_info.get("name", "")
    arguments = parse_tool_arguments(function_info.get("arguments", "{}"))

    if function_name == "get_local_travel_tips":
        city = arguments.get("city", "")
        print(f"[Act] get_local_travel_tips(city='{city}')")
        tool_result = get_local_travel_tips(city)
    elif function_name == "get_weather_from_api":
        city = arguments.get("city", "")
        print(f"[Act] get_weather_from_api(city='{city}')")
        tool_result = get_weather_from_api(city)
    elif function_name == "web_search_travel_guide":
        query_text = arguments.get("query_text", "")
        print(f"[Act] web_search_travel_guide(query_text='{query_text}')")
        tool_result = web_search_travel_guide(query_text)
    elif function_name == "rag_search_travel_knowledge":
        query = arguments.get("query", "")
        top_k = arguments.get("top_k", 3)
        print(f"[Act] rag_search_travel_knowledge(query='{query}', top_k={top_k})")
        tool_result = rag_search_travel_knowledge(query, top_k)
    else:
        tool_result = f"未知工具：{function_name}"

    print(f"[Observe] {tool_result}")
    return tool_result


def trim_history(conversation_history):
    max_messages = MAX_HISTORY_ROUNDS * 2
    return conversation_history[-max_messages:]


def run_agent(user_input, conversation_history):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + conversation_history
    messages.append({"role": "user", "content": user_input})

    print("[Think] 向模型注册 Tools，让模型决定是否调用 RAG、天气、搜索或本地函数...")
    used_tools = []
    for round_index in range(1, MAX_TOOL_ROUNDS + 1):
        response = call_llm_response(messages, temperature=0.2, tools=TOOLS, tool_choice="auto")
        tool_calls = response.get("tool_calls", [])
        if not tool_calls:
            print("[Think] 模型没有继续调用工具。")
            if used_tools:
                break
            return response.get("content", "模型没有返回可用回复。")

        print(f"[Think 结果] 第 {round_index} 轮请求调用 {len(tool_calls)} 个 Tool。")
        messages.append(response)
        for tool_call in tool_calls:
            function_name = tool_call.get("function", {}).get("name", "")
            used_tools.append(function_name)
            tool_result = execute_tool_call(tool_call)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.get("id", ""),
                    "name": function_name,
                    "content": tool_result,
                }
            )

    print("[Final] 将工具 Observe 结果交回模型，生成最终回答...")
    messages.append(
        {
            "role": "user",
            "content": (
                "请停止调用工具，只根据上面的工具观察结果生成自然语言回答。"
                "必须说明本轮使用了哪些工具，并重点说明 RAG 检索到的知识如何影响你的旅行建议。"
            ),
        }
    )
    final_response = call_llm_response(messages, temperature=0.7)
    return final_response.get("content", "模型没有返回可用回复。")


def main():
    print("TravelPlanAgent v0.6：在 v0.5 基础上增加基于 bge-m3 embedding 的 RAG")
    print("可用 Tools：local_function、weather_api、web_search、rag_search_travel_knowledge。")
    print("输入 exit、quit 或 退出 可以结束对话。")

    conversation_history = []

    while True:
        user_input = input("\n你：").strip()
        if user_input.lower() in ["exit", "quit"] or user_input == "退出":
            print("TravelPlanAgent：下次旅行再见！")
            break

        final_answer = run_agent(user_input, conversation_history)
        print(f"TravelPlanAgent：{final_answer}")

        conversation_history.append({"role": "user", "content": user_input})
        conversation_history.append({"role": "assistant", "content": final_answer})
        conversation_history = trim_history(conversation_history)


if __name__ == "__main__":
    main()