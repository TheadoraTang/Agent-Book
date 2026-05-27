import json
import urllib.error
import urllib.request


BASE_URL = "https://chat.ecnu.edu.cn/open/api/v1"
API_KEY = ""
MODEL_NAME = "ecnu-max"
CHAT_COMPLETIONS_URL = f"{BASE_URL}/chat/completions"


def call_llm(messages, temperature=0.7):
    if API_KEY == "YOUR_API_KEY":
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


def main():
    print("TravelPlanAgent v0.1：学会调用LLM进行单轮对话")

    user_input = input("\n你：").strip()
    messages = [{"role": "user", "content": user_input}]
    answer = call_llm(messages)
    print(f"TravelPlanAgent：{answer}")


if __name__ == "__main__":
    main()
