import logging
import sys


logger = logging.getLogger("travel_agent")


LOCAL_TRAVEL_TIPS = {
    "北京": "北京景点预约要求较多，故宫、国家博物馆等建议提前预约。",
    "上海": "上海城市交通方便，地铁适合串联人民广场、外滩和陆家嘴。",
    "杭州": "杭州西湖适合步行和骑行，节假日湖滨和断桥一带人流密集。",
    "成都": "成都适合慢节奏旅行，熊猫基地建议上午早点去。",
}

MOCK_WEATHER = {
    "北京": "北京今天晴，气温 18-27 摄氏度，适合户外游览。",
    "上海": "上海今天多云，气温 21-28 摄氏度，晚间江边风较大。",
    "杭州": "杭州今天小雨，气温 20-25 摄氏度，建议带伞。",
    "成都": "成都今天阴，气温 19-26 摄氏度，适合安排室内外结合的路线。",
}


def setup_logging():
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))

    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.propagate = False


def extract_city(user_input):
    for city in LOCAL_TRAVEL_TIPS:
        if city in user_input:
            return city
    return "成都"


def get_local_travel_tips(city):
    logger.info("Tool Call: get_local_travel_tips(city='%s')", city)
    result = LOCAL_TRAVEL_TIPS.get(city, "暂时没有这个城市的本地小贴士。")
    logger.info("Observe: %s", result)
    return result


def get_weather(city):
    logger.info("Tool Call: get_weather(city='%s')", city)
    result = MOCK_WEATHER.get(city, "暂时没有这个城市的天气数据。")
    logger.info("Observe: %s", result)
    return result


def run_agent(user_input):
    logger.info("Think: 正在分析用户需求：%s", user_input)

    city = extract_city(user_input)
    logger.info("Think: 识别到目的地是：%s", city)

    weather_result = get_weather(city)
    tip_result = get_local_travel_tips(city)

    logger.info("Think: 已获得工具观察结果，开始组织最终回答。")

    return (
        f"TravelPlanAgent：为你生成 {city} 一日游建议。\n"
        f"天气参考：{weather_result}\n"
        f"本地提醒：{tip_result}\n"
        "建议上午安排核心景点，下午留出弹性时间，晚上选择交通方便的区域用餐。"
    )


def main():
    setup_logging()

    user_input = "我想去成都玩一天，帮我看看天气和注意事项"
    final_answer = run_agent(user_input)

    print(final_answer)


if __name__ == "__main__":
    main()
