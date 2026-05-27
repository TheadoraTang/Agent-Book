import json
import urllib.error
import urllib.request


BASE_URL = "https://chat.ecnu.edu.cn/open/api/v1"
API_KEY = ""
MODEL_NAME = "ecnu-max"
CHAT_COMPLETIONS_URL = f"{BASE_URL}/chat/completions"
MAX_HISTORY_ROUNDS = 5

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
    for city in LOCAL_TRAVEL_TIPS:
        if city in text:
            return city
    return "未识别城市"


def get_local_travel_tips(city):
    return LOCAL_TRAVEL_TIPS.get(city, f"没有识别到支持城市。当前支持：{', '.join(LOCAL_TRAVEL_TIPS)}。")


def trim_history(conversation_history):
    max_messages = MAX_HISTORY_ROUNDS * 2
    return conversation_history[-max_messages:]


def build_user_message(user_input):
    city = extract_city(user_input)
    tool_result = get_local_travel_tips(city)
    tool_record = {
        "tool_name": "local_function",
        "tool_function": "get_local_travel_tips",
        "tool_input": city,
        "tool_action": "读取代码内置的城市出行提醒和本地小贴士。",
        "tool_result": tool_result,
    }

    print(f"[工具调用] get_local_travel_tips(city='{city}')")
    print(f"[工具作用] {tool_record['tool_action']}")
    print(f"[工具结果] {tool_result}")

    return (
        f"{user_input}\n\n"
        "本轮必须调用的 Tool 已经执行。请在回答开头说明 Tool 使用情况：\n"
        f"{json.dumps(tool_record, ensure_ascii=False, indent=2)}"
    )


def main():
    print("TravelPlanAgent v0.4 tool1：只调用 local_function")
    print("这个版本每轮都会调用 get_local_travel_tips。")
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
