import json
import urllib.error
import urllib.parse
import urllib.request


BASE_URL = "https://chat.ecnu.edu.cn/open/api/v1"
API_KEY = "sk-4b905783f8ab4fed9f7c1879aaf2ae58"
MODEL_NAME = "ecnu-max"
CHAT_COMPLETIONS_URL = f"{BASE_URL}/chat/completions"
MAX_HISTORY_ROUNDS = 5

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


def extract_city(text):
    for city in CITY_COORDINATES:
        if city in text:
            return city
    return "未识别城市"


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


def trim_history(conversation_history):
    max_messages = MAX_HISTORY_ROUNDS * 2
    return conversation_history[-max_messages:]


def build_user_message(user_input):
    city = extract_city(user_input)
    tool_result = get_weather_from_api(city)
    tool_record = {
        "tool_name": "weather_api",
        "tool_function": "get_weather_from_api",
        "tool_input": city,
        "tool_action": "调用天气 API 获取目的地当前天气。",
        "tool_result": tool_result,
    }

    print(f"[工具调用] get_weather_from_api(city='{city}')")
    print(f"[工具作用] {tool_record['tool_action']}")
    print(f"[工具结果] {tool_result}")

    return (
        f"{user_input}\n\n"
        "本轮必须调用的 Tool 已经执行。请在回答开头说明 Tool 使用情况：\n"
        f"{json.dumps(tool_record, ensure_ascii=False, indent=2)}"
    )


def main():
    print("TravelPlanAgent v0.4 tool2：只调用 weather_api")
    print("这个版本每轮都会调用 get_weather_from_api。")
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
