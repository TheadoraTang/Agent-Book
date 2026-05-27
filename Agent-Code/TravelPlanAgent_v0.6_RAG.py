import json
import re
import urllib.error
import urllib.request


BASE_URL = "https://chat.ecnu.edu.cn/open/api/v1"
API_KEY = "sk-4b905783f8ab4fed9f7c1879aaf2ae58"
MODEL_NAME = "ecnu-max"
CHAT_COMPLETIONS_URL = f"{BASE_URL}/chat/completions"
MAX_HISTORY_ROUNDS = 5

SYSTEM_PROMPT = """
你是 TravelPlanAgent，一个友好、可靠、适合初学者理解的旅行规划助手。
你具备 Think-Act-Observe 能力：先判断是否需要工具，再根据工具结果回答。
你可以利用最近几轮对话中的目的地、日期、预算、同行人和偏好来继续回答。
你的回答需要满足：
1. 优先围绕旅行目的地、时间、预算、交通、天气、景点和注意事项展开。
2. 语气清晰、亲切，不夸大，不编造无法确定的信息。
3. 输出尽量包含三个部分：建议、理由、提醒。
"""

THINK_PROMPT = """
请判断用户问题需要调用哪些工具。
你只能输出 JSON，不要输出 Markdown，不要输出解释。
JSON 格式如下：
{
  "need_weather": true 或 false,
  "weather_city": "城市名，如果不需要天气则为空字符串",
  "need_knowledge": true 或 false,
  "knowledge_query": "用于检索旅游知识的问题",
  "reason": "一句话说明判断原因"
}
当前可用天气城市：上海、北京、杭州、成都、广州。
"""

WEATHER_DATA = {
    "上海": "上海明天多云，18-25 摄氏度，东南风 3 级，适合城市漫步，建议带薄外套。",
    "北京": "北京明天晴，12-24 摄氏度，昼夜温差较大，适合参观故宫、颐和园等户外景点。",
    "杭州": "杭州明天小雨，17-22 摄氏度，西湖边可能湿滑，建议带伞并准备防水鞋。",
    "成都": "成都明天阴，16-23 摄氏度，适合室内外结合游玩，建议安排茶馆和博物馆。",
    "广州": "广州明天阵雨，22-29 摄氏度，体感偏闷热，建议带伞并选择透气衣物。",
}

TRAVEL_KNOWLEDGE = """
上海：适合第一次到访的路线可以从人民广场出发，步行到南京东路，再到外滩看城市天际线。喜欢城市文化可以安排武康路、思南路和上海博物馆。亲子旅行可以考虑上海科技馆、自然博物馆和迪士尼。

北京：经典路线包括天安门、故宫、景山公园、什刹海。第一次去故宫建议提前预约，并预留至少半天。长城距离市区较远，建议单独安排一天。北京秋季适合城市漫步，但早晚温差明显。

杭州：西湖适合慢节奏游玩，可以选择断桥、白堤、苏堤、雷峰塔和湖滨步行区。灵隐寺适合上午前往。雨天可以安排中国茶叶博物馆、南宋德寿宫遗址博物馆或河坊街。

成都：适合慢旅行。常见安排包括大熊猫繁育研究基地、宽窄巷子、人民公园、武侯祠和锦里。喜欢美食可以安排火锅、串串、担担面。熊猫基地建议上午早点去。

广州：适合美食和城市文化旅行。常见安排包括陈家祠、沙面、永庆坊、北京路、珠江夜游和广东省博物馆。早茶体验适合安排在上午，夏季需要注意闷热和阵雨。

旅行规划通用原则：每天不要安排过多景点。城市初访建议每天 2 到 4 个主要点位，留出交通和休息时间。亲子、老人同行时应降低步行强度。雨天优先安排博物馆、展览、商场和餐饮体验。
"""


def call_llm(messages, temperature=0.7):
    if API_KEY == "YOUR_API_KEY":
        return "请先把文件顶部的 API_KEY 替换成教师提供的真实密钥。"

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


def get_weather(city):
    return WEATHER_DATA.get(city, f"暂时没有 {city} 的模拟天气数据。请提醒用户确认真实天气。")


def split_knowledge(text):
    chunks = []
    for part in text.strip().split("\n\n"):
        chunk = part.strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def tokenize(text):
    chinese_chars = re.findall(r"[\u4e00-\u9fff]", text)
    english_words = re.findall(r"[A-Za-z0-9]+", text.lower())
    return set(chinese_chars + english_words)


def search_travel_knowledge(query, top_k=3):
    query_tokens = tokenize(query)
    scored_chunks = []
    for chunk in split_knowledge(TRAVEL_KNOWLEDGE):
        chunk_tokens = tokenize(chunk)
        score = len(query_tokens & chunk_tokens)
        if score > 0:
            scored_chunks.append((score, chunk))

    scored_chunks.sort(key=lambda item: item[0], reverse=True)
    results = [chunk for _, chunk in scored_chunks[:top_k]]
    if not results:
        return "没有检索到相关旅游知识。"
    return "\n\n".join(results)


def extract_json(text):
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("模型没有返回 JSON")
    return json.loads(text[start : end + 1])


def think(user_input):
    messages = [
        {"role": "system", "content": THINK_PROMPT},
        {"role": "user", "content": user_input},
    ]
    result = call_llm(messages, temperature=0)
    return extract_json(result)


def act(decision):
    observations = []
    if decision.get("need_weather"):
        city = decision.get("weather_city", "")
        observations.append(
            {
                "tool_name": "get_weather",
                "tool_input": city,
                "tool_result": get_weather(city),
            }
        )
    if decision.get("need_knowledge"):
        query = decision.get("knowledge_query", "")
        observations.append(
            {
                "tool_name": "search_travel_knowledge",
                "tool_input": query,
                "tool_result": search_travel_knowledge(query),
            }
        )
    if not observations:
        observations.append({"tool_name": "none", "tool_input": "", "tool_result": "没有调用工具。"})
    return observations


def observe(user_input, decision, action_results):
    return f"""
用户问题：{user_input}

Think 阶段决策：{json.dumps(decision, ensure_ascii=False)}
Act 阶段结果：{json.dumps(action_results, ensure_ascii=False)}

请基于以上信息回答用户。如果检索结果不足，请明确提醒用户补充目的地、天数、预算或偏好。
"""


def answer(observation, conversation_history):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + conversation_history
    messages.append({"role": "user", "content": observation})
    return call_llm(messages)


def trim_history(conversation_history):
    max_messages = MAX_HISTORY_ROUNDS * 2
    return conversation_history[-max_messages:]


def run_agent(user_input, conversation_history):
    print("[Think] 判断需要天气工具、知识检索，还是直接回答...")
    try:
        decision = think(user_input)
        print(f"[Think 结果] {json.dumps(decision, ensure_ascii=False)}")
    except Exception as error:
        print(f"[Think 失败] {error}")
        return call_llm(
            [{"role": "system", "content": SYSTEM_PROMPT}] + conversation_history + [{"role": "user", "content": user_input}]
        )

    print("[Act] 调用工具...")
    action_results = act(decision)
    print(f"[Act 结果] {json.dumps(action_results, ensure_ascii=False)}")

    print("[Observe] 整理观察结果...")
    observation = observe(user_input, decision, action_results)

    return answer(observation, conversation_history)


def main():
    print("TravelPlanAgent v0.6：具备旅游知识检索能力")
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
