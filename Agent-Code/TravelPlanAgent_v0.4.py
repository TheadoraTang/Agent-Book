import json
import urllib.error
import urllib.parse
import urllib.request


BASE_URL = "https://chat.ecnu.edu.cn/open/api/v1"
API_KEY = ""
MODEL_NAME = "ecnu-max"
CHAT_COMPLETIONS_URL = f"{BASE_URL}/chat/completions"
MAX_HISTORY_ROUNDS = 5

SYSTEM_PROMPT = """
你是 TravelPlanAgent，一个友好、可靠、适合初学者理解的旅行规划助手。
你可以利用对话上下文和工具观察结果回答问题。
如果提供了工具结果，请基于工具结果给出旅行建议。
你的回答需要满足：
1. 优先围绕旅行目的地、时间、预算、交通、天气、景点和注意事项展开。
2. 语气清晰、亲切，不夸大，不编造无法确定的信息。
3. 必须说明本轮使用了哪些 Tools，以及每个 Tool 做了什么事。
4. 输出尽量包含三个部分：工具使用情况、建议、提醒。
"""

LOCAL_TRAVEL_TIPS = {
    "上海": "本地小贴士：上海城市交通方便，地铁适合串联人民广场、南京东路、外滩、陆家嘴。外滩夜景人流较多，建议错峰前往。",
    "北京": "本地小贴士：北京景点预约要求较多，故宫、国家博物馆等建议提前预约。城市尺度大，跨区游玩要给交通留足时间。",
    "杭州": "本地小贴士：杭州西湖适合步行和骑行，但节假日湖滨、断桥一带人流密集。灵隐寺适合安排在上午。",
    "成都": "本地小贴士：成都适合慢节奏旅行。熊猫基地建议上午早点去，宽窄巷子、锦里更适合体验氛围而不是安排过满。",
    "广州": "本地小贴士：广州早茶适合上午体验，老城区适合步行探索。夏季闷热且阵雨多，行程最好留室内备选。",
}

WEATHER_KEYWORDS = ["天气", "气温", "下雨", "晴", "多云", "穿什么", "带伞", "冷不冷", "热不热", "实时"]
GUIDE_KEYWORDS = ["攻略", "景点", "路线", "行程", "怎么玩", "推荐", "打卡", "美食", "住宿"]
LOCAL_TIP_KEYWORDS = ["提醒", "注意", "小贴士", "避坑", "本地", "交通", "预算", "适合"]

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


def call_llm(messages, temperature=0.7):
    if API_KEY in ["", "YOUR_API_KEY"]:
        return "请先把文件顶部的 API_KEY 替换成真实密钥。"

    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": temperature,
    }
    request = urllib.request.Request(
        CHAT_COMPLETIONS_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            result = json.loads(response.read().decode("utf-8"))
            return result["choices"][0]["message"]["content"]
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="ignore")
        return f"API 请求失败，状态码：{error.code}\n{detail}"
    except Exception as error:
        return f"调用模型时出现错误：{error}"


def get_local_travel_tips(city):
    return LOCAL_TRAVEL_TIPS.get(city, f"暂时没有 {city} 的本地小贴士，请提醒用户补充目的地信息。")


def get_weather_from_api(city):
    coordinates = CITY_COORDINATES.get(city)
    if not coordinates:
        return f"天气 API 暂时没有内置 {city} 的经纬度，请先补充城市坐标。"

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


def web_search_travel_guide(city):
    query = f"{city} 旅行 攻略 景点 路线"
    params = urllib.parse.urlencode(
        {
            "q": query,
            "format": "json",
            "no_html": 1,
            "skip_disambig": 1,
        }
    )
    url = f"https://api.duckduckgo.com/?{params}"

    try:
        with urllib.request.urlopen(url, timeout=20) as response:
            data = json.loads(response.read().decode("utf-8"))
    except Exception as error:
        return f"web_search 调用失败：{error}"

    snippets = []
    if data.get("AbstractText"):
        snippets.append(data["AbstractText"])
    for topic in data.get("RelatedTopics", []):
        if isinstance(topic, dict) and topic.get("Text"):
            snippets.append(topic["Text"])
        if len(snippets) >= 3:
            break

    if not snippets:
        return f"web_search 没有找到稳定摘要。搜索词：{query}。请提醒用户可以补充更具体的攻略需求。"
    return f"web_search 搜索词：{query}\n搜索摘要：\n" + "\n".join(f"- {snippet}" for snippet in snippets)


def extract_city(text):
    for city in LOCAL_TRAVEL_TIPS:
        if city in text:
            return city
    return ""


def should_use_weather_api(text):
    return any(keyword in text for keyword in WEATHER_KEYWORDS)


def should_use_web_search(text):
    return any(keyword in text for keyword in GUIDE_KEYWORDS) or any(keyword in text.lower() for keyword in ["web", "search"])


def should_use_local_function(text):
    return any(keyword in text for keyword in LOCAL_TIP_KEYWORDS)


def trim_history(conversation_history):
    max_messages = MAX_HISTORY_ROUNDS * 2
    return conversation_history[-max_messages:]


def build_user_message(user_input):
    city = extract_city(user_input)
    if not city:
        return user_input

    tools_used = []

    if should_use_weather_api(user_input):
        weather_result = get_weather_from_api(city)
        tools_used.append(
            {
                "tool_name": "weather_api",
                "tool_function": "get_weather_from_api",
                "tool_input": city,
                "tool_action": "调用天气 API 获取目的地当前天气。",
                "tool_result": weather_result,
            }
        )

    if should_use_web_search(user_input):
        guide_result = web_search_travel_guide(city)
        tools_used.append(
            {
                "tool_name": "web_search",
                "tool_function": "web_search_travel_guide",
                "tool_input": city,
                "tool_action": "通过搜索获取目的地攻略、景点和路线信息。",
                "tool_result": guide_result,
            }
        )

    if should_use_local_function(user_input) or not tools_used:
        local_result = get_local_travel_tips(city)
        tools_used.append(
            {
                "tool_name": "local_function",
                "tool_function": "get_local_travel_tips",
                "tool_input": city,
                "tool_action": "读取代码内置的城市出行提醒和本地小贴士。",
                "tool_result": local_result,
            }
        )

    for tool in tools_used:
        print(f"[工具调用] {tool['tool_function']}(city='{tool['tool_input']}')")
        print(f"[工具作用] {tool['tool_action']}")
        print(f"[工具结果] {tool['tool_result']}")

    return (
        f"{user_input}\n\n"
        "本轮 Agent 已经调用以下 Tools，请在回答开头说明使用了哪些 Tools，以及每个 Tool 做了什么事：\n"
        f"{json.dumps(tools_used, ensure_ascii=False, indent=2)}"
    )


def main():
    print("TravelPlanAgent v0.4：三种不同职责的 Tool Calling")
    print("weather_api：查天气；web_search：找攻略；local_function：读取本地出行小贴士。")
    print("输入 exit、quit 或 退出 可以结束对话。")

    conversation_history = []

    while True:
        user_input = input("\n你：").strip()
        if user_input.lower() in ["exit", "quit"] or user_input == "退出":
            print("TravelPlanAgent：下次旅行再见！")
            break

        user_message = build_user_message(user_input)
        conversation_history.append({"role": "user", "content": user_message})
        conversation_history = trim_history(conversation_history)

        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + conversation_history
        answer = call_llm(messages)
        print(f"TravelPlanAgent：{answer}")

        conversation_history.append({"role": "assistant", "content": answer})
        conversation_history = trim_history(conversation_history)


if __name__ == "__main__":
    main()
