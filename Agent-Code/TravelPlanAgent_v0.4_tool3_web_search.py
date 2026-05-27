import json
import socket
import time
import urllib.error
import urllib.request


BASE_URL = "https://chat.ecnu.edu.cn/open/api/v1"
API_KEY = "sk-4b905783f8ab4fed9f7c1879aaf2ae58"
MODEL_NAME = "ecnu-plus"
CHAT_COMPLETIONS_URL = f"{BASE_URL}/chat/completions"
MAX_HISTORY_ROUNDS = 5
LLM_TIMEOUT_SECONDS = 300
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

    for attempt in range(2):
        try:
            with urllib.request.urlopen(request, timeout=LLM_TIMEOUT_SECONDS) as response:
                result = json.loads(response.read().decode("utf-8"))
                return result["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="ignore")
            return f"API 请求失败，状态码：{error.code}\n{detail}"
        except (socket.timeout, TimeoutError) as error:
            if attempt == 0:
                print("[模型调用超时] 等待 2 秒后重试一次...")
                time.sleep(2)
                continue
            return f"调用模型超时：{error}。可以稍后重试，或减少搜索结果数量。"
        except Exception as error:
            return f"调用模型时出现错误：{error}"


def extract_city(text):
    for city in SUPPORTED_CITIES:
        if city in text:
            return city
    return "未识别城市"


def web_search_travel_guide(query_text):
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


def trim_history(conversation_history):
    max_messages = MAX_HISTORY_ROUNDS * 2
    return conversation_history[-max_messages:]


def build_user_message(user_input):
    city = extract_city(user_input)
    query_text = city if city != "未识别城市" else user_input
    tool_result = web_search_travel_guide(query_text)
    tool_record = {
        "tool_name": "web_search",
        "tool_function": "web_search_travel_guide",
        "tool_input": query_text,
        "tool_action": "通过搜索获取目的地攻略、景点和路线信息。",
        "tool_result": tool_result,
    }

    print(f"[工具调用] web_search_travel_guide(query_text='{query_text}')")
    print(f"[工具作用] {tool_record['tool_action']}")
    print(f"[工具结果] {tool_result}")

    return (
        f"{user_input}\n\n"
        "本轮必须调用的 Tool 已经执行。请在回答开头说明 Tool 使用情况：\n"
        f"{json.dumps(tool_record, ensure_ascii=False, indent=2)}"
    )


def main():
    print("TravelPlanAgent v0.4 tool3：只调用 web_search")
    print("这个版本每轮都会调用 web_search_travel_guide。")
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
