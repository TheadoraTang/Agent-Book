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
你可以利用最近几轮对话中的目的地、日期、预算、同行人和偏好来继续回答。
你的回答需要满足：
1. 优先围绕旅行目的地、时间、预算、交通、天气、景点和注意事项展开。
2. 语气清晰、亲切，不夸大，不编造无法确定的信息。
3. 输出尽量包含三个部分：建议、理由、提醒。
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


def trim_history(conversation_history):
    max_messages = MAX_HISTORY_ROUNDS * 2
    return conversation_history[-max_messages:]


def main():
    print("TravelPlanAgent v0.3：支持最近 5 轮上下文记忆")
    print("输入 exit、quit 或 退出 可以结束对话。")

    conversation_history = []

    while True:
        user_input = input("\n你：").strip()
        if user_input.lower() in ["exit", "quit"] or user_input == "退出":
            print("TravelPlanAgent：下次旅行再见！")
            break

        conversation_history.append({"role": "user", "content": user_input})
        conversation_history = trim_history(conversation_history)

        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + conversation_history
        answer = call_llm(messages)
        print(f"TravelPlanAgent：{answer}")

        conversation_history.append({"role": "assistant", "content": answer})
        conversation_history = trim_history(conversation_history)


if __name__ == "__main__":
    main()
