import json
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np


BASE_URL = "https://chat.ecnu.edu.cn/open/api/v1"
API_KEY = ""
MODEL_NAME = "ecnu-max"
CHAT_COMPLETIONS_URL = f"{BASE_URL}/chat/completions"
MAX_HISTORY_ROUNDS = 5
LLM_TIMEOUT_SECONDS = 300
SEARCH_API_URL = "https://searchfree.site/api/search"

BASE_DIR = Path(__file__).resolve().parent
BGE_M3_MODEL_PATH = BASE_DIR / "assets" / "bge-m3"

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


def call_llm(messages, temperature=0.7):
    if API_KEY in ["", "YOUR_API_KEY"]:
        return "请先把文件顶部的 API_KEY 替换成真实密钥。"

    payload = {"model": MODEL_NAME, "messages": messages, "temperature": temperature}
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
                return result["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="ignore")
            return f"API 请求失败，状态码：{error.code}\n{detail}"
        except (socket.timeout, TimeoutError) as error:
            if attempt == 0:
                print("[模型调用超时] 等待 2 秒后重试一次...")
                time.sleep(2)
                continue
            return f"调用模型超时：{error}。可以稍后重试。"
        except Exception as error:
            return f"调用模型时出现错误：{error}"


def extract_json_array(text):
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("模型没有返回 JSON 数组")
    return json.loads(text[start : end + 1])


def extract_city(text):
    for city in LOCAL_TRAVEL_TIPS:
        if city in text:
            return city
    return ""


class BaseAgent:
    name = "BaseAgent"
    role = "基础 Agent"

    def run(self, task):
        raise NotImplementedError


class LocalTipsAgent(BaseAgent):
    name = "LocalTipsAgent"
    role = "读取本地出行提醒"
    local_travel_tips = LOCAL_TRAVEL_TIPS

    def get_local_travel_tips(self, city):
        return self.local_travel_tips.get(city, f"没有识别到支持城市。当前支持：{', '.join(self.local_travel_tips)}。")

    def run(self, task):
        city = task.get("city") or extract_city(task.get("task", ""))
        result = self.get_local_travel_tips(city)
        return {"agent": self.name, "role": self.role, "task": task, "result": result}


class WeatherAgent(BaseAgent):
    name = "WeatherAgent"
    role = "调用天气 API"
    city_coordinates = CITY_COORDINATES
    weather_code_map = WEATHER_CODE_MAP

    def get_weather_from_api(self, city):
        coordinates = self.city_coordinates.get(city)
        if not coordinates:
            return f"没有识别到支持城市，无法查询天气 API。当前支持：{', '.join(self.city_coordinates)}。"

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
        weather_text = self.weather_code_map.get(weather_code, f"未知天气代码 {weather_code}")
        return (
            f"{city} 当前天气：{weather_text}，"
            f"气温 {current.get('temperature_2m')} 摄氏度，"
            f"相对湿度 {current.get('relative_humidity_2m')}%，"
            f"降水量 {current.get('precipitation')} mm，"
            f"风速 {current.get('wind_speed_10m')} km/h。"
            "数据来自 Open-Meteo 天气 API。"
        )

    def run(self, task):
        city = task.get("city") or extract_city(task.get("task", ""))
        result = self.get_weather_from_api(city)
        return {"agent": self.name, "role": self.role, "task": task, "result": result}


class RAGAgent(BaseAgent):
    name = "RAGAgent"
    role = "使用 bge-m3 embedding 做知识库检索"

    def __init__(self):
        self.embedding_model = None
        self.rag_chunks = None
        self.rag_embeddings = None

    def load_embedding_model(self):
        if self.embedding_model is not None:
            return self.embedding_model
        if not BGE_M3_MODEL_PATH.exists():
            raise FileNotFoundError(f"没有找到本地模型目录：{BGE_M3_MODEL_PATH}")
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise ImportError(
                "当前环境缺少 sentence_transformers，无法加载 assets/bge-m3 生成 embedding。"
                "请先安装：pip install sentence-transformers torch"
            ) from error

        print(f"[RAG] 正在加载本地 embedding 模型：{BGE_M3_MODEL_PATH}")
        self.embedding_model = SentenceTransformer(str(BGE_M3_MODEL_PATH))
        return self.embedding_model

    def embed_texts(self, texts):
        model = self.load_embedding_model()
        embeddings = model.encode(
            texts,
            batch_size=8,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def build_rag_index(self):
        if self.rag_chunks is not None and self.rag_embeddings is not None:
            return self.rag_chunks, self.rag_embeddings

        self.rag_chunks = [chunk.strip() for chunk in TRAVEL_KNOWLEDGE.strip().split("\n\n") if chunk.strip()]
        print(f"[RAG] 正在为 {len(self.rag_chunks)} 个知识片段生成 embedding...")
        self.rag_embeddings = self.embed_texts(self.rag_chunks)
        return self.rag_chunks, self.rag_embeddings

    def rag_search_travel_knowledge(self, query, top_k=3):
        try:
            chunks, chunk_embeddings = self.build_rag_index()
            query_embedding = self.embed_texts([query])[0]
        except Exception as error:
            return f"RAG 检索失败：{error}"

        top_k = max(1, min(int(top_k), 5))
        scores = chunk_embeddings @ query_embedding
        ranked_indices = np.argsort(scores)[::-1][:top_k]

        lines = [
            "RAGAgent 检索结果：",
            f"Embedding 模型：{BGE_M3_MODEL_PATH}",
            f"检索问题：{query}",
        ]
        for rank, index in enumerate(ranked_indices, start=1):
            lines.append(f"{rank}. 相似度：{float(scores[index]):.4f}")
            lines.append(chunks[index])
        return "\n".join(lines)

    def run(self, task):
        query = task.get("query") or task.get("task", "")
        result = self.rag_search_travel_knowledge(query, top_k=3)
        return {"agent": self.name, "role": self.role, "task": task, "result": result}


class WebSearchAgent(BaseAgent):
    name = "WebSearchAgent"
    role = "调用搜索 API 获取公开网页攻略"
    search_api_url = SEARCH_API_URL

    def web_search_travel_guide(self, query_text):
        query = f"{query_text} 旅游攻略 景点 路线"
        payload = {"query": query, "max_results": 3}
        request = urllib.request.Request(
            self.search_api_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "User-Agent": "TravelPlanAgent/0.7"},
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
        lines = ["WebSearchAgent 搜索结果：", f"搜索词：{query}"]
        if data.get("answer"):
            lines.append(f"AI 摘要：{data['answer'][:300]}")
        for index, result in enumerate(results, start=1):
            lines.append(f"{index}. 标题：{result.get('title', '无标题')}")
            if result.get("content"):
                lines.append(f"   摘要：{result['content'][:160]}")
            if result.get("url"):
                lines.append(f"   链接：{result['url']}")
        if not results:
            lines.append("没有返回搜索结果。")
        return "\n".join(lines)

    def run(self, task):
        query = task.get("query") or task.get("task", "")
        result = self.web_search_travel_guide(query)
        return {"agent": self.name, "role": self.role, "task": task, "result": result}


class ItineraryAgent(BaseAgent):
    name = "ItineraryAgent"
    role = "根据各副 Agent 结果生成行程草案"

    def run(self, task):
        task_for_prompt = dict(task)
        prompt = f"""
                你是 ItineraryAgent，负责根据多个副 Agent 的观察结果生成旅行计划草案。
                请输出：需求理解、信息依据、每日安排、交通建议、预算提醒、风险提醒、可调整项。
                
                任务：{json.dumps(task_for_prompt, ensure_ascii=False)}
                """
        result = call_llm([{"role": "user", "content": prompt}], temperature=0.4)
        compact_task = {
            "agent": task.get("agent", self.name),
            "task": task.get("task", ""),
            "city": task.get("city", ""),
            "query": task.get("query", ""),
        }
        return {"agent": self.name, "role": self.role, "task": compact_task, "result": result}


class MainAgent:
    def __init__(self):
        self.sub_agents = {
            "LocalTipsAgent": LocalTipsAgent(),
            "WeatherAgent": WeatherAgent(),
            "RAGAgent": RAGAgent(),
            "WebSearchAgent": WebSearchAgent(),
            "ItineraryAgent": ItineraryAgent(),
        }

    def decompose_tasks(self, user_input):
        prompt = f"""
                你是 TravelPlanAgent 的主 Agent。请把用户旅行需求拆解成子任务，并为每个子任务选择一个副 Agent。
                
                可用副 Agent：
                1. LocalTipsAgent：读取本地出行提醒，适合交通、预约、人流、避坑。
                2. WeatherAgent：调用天气 API，适合天气、气温、下雨、穿衣。
                3. RAGAgent：使用本地 bge-m3 embedding 检索旅游知识库，适合景点、路线、雨天安排、亲子/老人/预算等规划知识。
                4. WebSearchAgent：调用搜索 API，适合需要公开网页攻略或最新资料。
                5. ItineraryAgent：根据其他副 Agent 的结果生成完整行程草案。只有当用户明确要求完整行程、攻略、几日游方案时才使用。
                
                只输出 JSON 数组，不要输出 Markdown。
                只选择完成任务所必需的副 Agent，不要默认调用所有副 Agent。
                每个元素格式：
                {{
                  "agent": "副 Agent 名称",
                  "task": "子任务描述",
                  "city": "城市名，如果没有则为空字符串",
                  "query": "给副 Agent 的查询内容"
                }}
                
                用户需求：{user_input}
                """
        result = call_llm([{"role": "user", "content": prompt}], temperature=0.2)
        try:
            tasks = extract_json_array(result)
            if tasks:
                return tasks
        except Exception as error:
            print(f"[Decompose 失败] {error}")
        return self.fallback_tasks(user_input)

    def should_use_sub_agents(self, user_input):
        travel_keywords = [
            "旅游",
            "旅行",
            "出行",
            "攻略",
            "行程",
            "路线",
            "景点",
            "天气",
            "酒店",
            "住宿",
            "交通",
            "预算",
            "美食",
            "门票",
            "预约",
            "几日游",
            "怎么玩",
            "避坑",
        ]
        known_city = extract_city(user_input)
        return bool(known_city) or any(keyword in user_input for keyword in travel_keywords)

    def classify_intent(self, user_input):
        weather_keywords = ["天气", "气温", "下雨", "带伞", "穿什么", "冷不冷", "热不热"]
        plan_keywords = ["攻略", "行程", "路线", "怎么玩", "几日游", "计划", "安排"]
        local_keywords = ["注意", "提醒", "避坑"]
        if any(keyword in user_input for keyword in weather_keywords) and not any(keyword in user_input for keyword in plan_keywords):
            return "weather_only"
        if any(keyword in user_input for keyword in local_keywords) and not any(keyword in user_input for keyword in plan_keywords):
            return "tips_only"
        if any(keyword in user_input for keyword in plan_keywords):
            return "travel_plan"
        return "travel_question"

    def direct_answer(self, user_input, conversation_history):
        prompt = f"""
                用户输入和具体旅行规划任务无关，请不要调用任何副 Agent。
                请用 TravelPlanAgent 的身份自然回应。如果用户只是问候，就简短问候并提示可以帮他规划旅行。
                
                用户输入：{user_input}
                """
        messages = [{"role": "system", "content": "你是 TravelPlanAgent 的主 Agent。"}]
        messages.extend(conversation_history)
        messages.append({"role": "user", "content": prompt})
        return call_llm(messages, temperature=0.6)

    def fallback_tasks(self, user_input):
        city = extract_city(user_input)
        tasks = []
        weather_keywords = ["天气", "气温", "下雨", "带伞", "穿什么", "冷不冷", "热不热"]
        plan_keywords = ["攻略", "行程", "路线", "怎么玩", "几日游", "计划", "安排", "旅游"]
        latest_keywords = ["最新", "开放", "门票", "预约", "搜索", "网上", "网页"]
        local_keywords = ["注意", "提醒", "避坑", "交通", "预算", "人流"]

        if any(keyword in user_input for keyword in plan_keywords):
            tasks.append({"agent": "RAGAgent", "task": "检索本地知识库中的旅行规划知识", "city": city, "query": user_input})
        if any(keyword in user_input for keyword in weather_keywords):
            tasks.append({"agent": "WeatherAgent", "task": "查询目的地天气", "city": city, "query": user_input})
        if any(keyword in user_input for keyword in local_keywords):
            tasks.append({"agent": "LocalTipsAgent", "task": "读取本地出行提醒", "city": city, "query": user_input})
        if any(keyword in user_input for keyword in latest_keywords):
            tasks.append({"agent": "WebSearchAgent", "task": "搜索公开网页攻略或最新信息", "city": city, "query": user_input})
        if any(keyword in user_input for keyword in plan_keywords):
            tasks.append({"agent": "ItineraryAgent", "task": "综合已获得信息生成行程草案", "city": city, "query": user_input})
        return tasks

    def run_sub_tasks(self, tasks, user_input):
        observations = []
        for task in tasks:
            agent_name = task.get("agent", "")
            agent = self.sub_agents.get(agent_name)
            if not agent:
                observations.append(
                    {"agent": agent_name or "UnknownAgent", "role": "未注册副 Agent", "task": task, "result": "没有找到对应副 Agent。"}
                )
                continue

            if agent_name == "ItineraryAgent":
                task = dict(task)
                task["user_input"] = user_input
                task["previous_observations"] = json.loads(json.dumps(observations, ensure_ascii=False))

            print(f"[SubAgent] {agent_name} 执行子任务：{task.get('task', '')}")
            observation = agent.run(task)
            observations.append(observation)
        return observations

    def observe_and_decide_next(self, user_input, tasks, observations):
        prompt = f"""
                你是 TravelPlanAgent 的主 Agent。你刚刚完成了一轮副 Agent 调用。
                请观察已有结果，判断是否还缺少必要信息。
                
                规则：
                - 只在确实缺少完成用户请求的关键信息时，才补充调用副 Agent。
                - 不要为了展示能力而调用所有副 Agent。
                - 如果已有信息足够，输出空数组 []。
                - 如果需要补充，只输出 JSON 数组，不要输出 Markdown。
                
                可用副 Agent：
                LocalTipsAgent、WeatherAgent、RAGAgent、WebSearchAgent、ItineraryAgent。
                
                用户需求：
                {user_input}
                
                已执行子任务：
                {json.dumps(tasks, ensure_ascii=False, indent=2)}
                
                已有观察结果：
                {json.dumps(observations, ensure_ascii=False, indent=2)}
                
                输出格式：
                [
                  {{
                    "agent": "副 Agent 名称",
                    "task": "补充子任务描述",
                    "city": "城市名，如果没有则为空字符串",
                    "query": "给副 Agent 的查询内容"
                  }}
                ]
                """
        result = call_llm([{"role": "user", "content": prompt}], temperature=0.2)
        try:
            next_tasks = extract_json_array(result)
        except Exception as error:
            print(f"[Observe 决策失败] {error}")
            return []

        executed = {(task.get("agent", ""), task.get("task", ""), task.get("query", "")) for task in tasks}
        filtered_tasks = []
        for task in next_tasks:
            key = (task.get("agent", ""), task.get("task", ""), task.get("query", ""))
            if key not in executed:
                filtered_tasks.append(task)
        return filtered_tasks

    def final_answer(self, user_input, tasks, observations, conversation_history):
        intent = self.classify_intent(user_input)
        prompt = f"""
            你是 TravelPlanAgent。请根据内部子任务结果生成面向用户的最终回答。
            子任务拆解和 Multi-Agent 协作过程是系统内部信息，不能出现在最终回答中。
            除非用户明确询问系统如何工作，否则不要提到“主 Agent”“副 Agent”“子任务”“Multi-Agent”“RAGAgent”“WeatherAgent”等内部名称。
            
            注意：最终回答的形式必须匹配用户原始需求。
            - 如果用户只是查询天气，就只回答天气结论、来源和简单出行提醒，不要生成完整旅行规划。
            - 如果用户只是问本地提醒，就只回答提醒，不要生成完整行程。
            - 只有当用户明确要求攻略、行程、路线、几日游、完整计划时，才输出完整旅行规划。
            
            用户需求：
            {user_input}
            
            用户意图分类：
            {intent}
            
            子任务拆解：
            {json.dumps(tasks, ensure_ascii=False, indent=2)}
            
            副 Agent 观察结果：
            {json.dumps(observations, ensure_ascii=False, indent=2)}
            
            如果是完整旅行规划，输出包含：
            1. 需求理解
            2. 信息依据
            3. 推荐行程
            4. 交通与预算提醒
            5. 风险提醒
            6. 还需要用户补充的信息
            
            如果不是完整旅行规划，请用更短的自然语言直接回答，并只说明必要的信息来源，例如天气 API、搜索结果或本地知识库。
            """
        messages = [{"role": "system", "content": "你是主 Agent，负责汇总多个副 Agent 的结果。"}]
        messages.extend(conversation_history)
        messages.append({"role": "user", "content": prompt})
        return call_llm(messages, temperature=0.5)

    def run(self, user_input, conversation_history):
        if not self.should_use_sub_agents(user_input):
            print("[Think] MainAgent 判断这是闲聊或非旅行任务，不调用副 Agent。")
            return self.direct_answer(user_input, conversation_history)

        print("[Think] MainAgent 拆解子任务并选择必要副 Agent...")
        tasks = self.decompose_tasks(user_input)
        if not tasks:
            print("[Think] 没有必要的子任务，直接回答。")
            return self.direct_answer(user_input, conversation_history)
        print(f"[MainAgent] 子任务：{json.dumps(tasks, ensure_ascii=False)}")

        print("[Act] MainAgent 调度已选择的副 Agent 执行任务...")
        observations = self.run_sub_tasks(tasks, user_input)

        print("[Observe] MainAgent 观察结果，判断是否需要补充调用副 Agent...")
        next_tasks = self.observe_and_decide_next(user_input, tasks, observations)
        if next_tasks:
            print(f"[Observe] 需要补充子任务：{json.dumps(next_tasks, ensure_ascii=False)}")
            more_observations = self.run_sub_tasks(next_tasks, user_input)
            tasks.extend(next_tasks)
            observations.extend(more_observations)
        else:
            print("[Observe] 已有结果足够，不再调用额外副 Agent。")

        print("[MainAgent] 汇总副 Agent 结果，生成最终旅行规划...")
        return self.final_answer(user_input, tasks, observations, conversation_history)


def trim_history(conversation_history):
    max_messages = MAX_HISTORY_ROUNDS * 2
    return conversation_history[-max_messages:]


def main():
    print("TravelPlanAgent v0.7：子任务拆解 + Multi-Agent 集成")
    print("主 Agent 会调度 LocalTipsAgent、WeatherAgent、RAGAgent、WebSearchAgent、ItineraryAgent。")
    print("输入 exit、quit 或 退出 可以结束对话。")

    main_agent = MainAgent()
    conversation_history = []

    while True:
        user_input = input("\n你：").strip()
        if user_input.lower() in ["exit", "quit"] or user_input == "退出":
            print("TravelPlanAgent：下次旅行再见！")
            break

        answer = main_agent.run(user_input, conversation_history)
        print(f"TravelPlanAgent：{answer}")

        conversation_history.append({"role": "user", "content": user_input})
        conversation_history.append({"role": "assistant", "content": answer})
        conversation_history = trim_history(conversation_history)


if __name__ == "__main__":
    main()
