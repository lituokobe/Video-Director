import asyncio
from datetime import datetime
from itertools import zip_longest
from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from pymilvus import AsyncMilvusClient
from config.constant_config import TASK_TYPE_CN, MATERIAL_TYPE, TOP_K_RECOMMENDATION, MATERIAL_CN, \
    DEFAULT_TOTAL_DURATION, TOP_K_MULT, DEFAULT_QUERY, DEFAULT_BGM_DESC, DEFAULT_FOOTAGE_DESC, DEFAULT_DURATION, \
    DEFAULT_IMAGE_DESC, DEFAULT_TTS_DESC
from config.path_config import MILVUS_URL, TTS_DATA_PATH
from config.schema_config import RecommendationRequest, RecommendationResult, TaskRequest, QueryResult, DecideTTSResult, \
    VideoPlansResponseNoScript, VideoPlansResponseScript, SelectScriptResult
from functionals.logger import video_director_logger
from functionals.utils import user_id_to_collection_name, embed_query
from models.llm_models import llm_reasoning, llm_regular

ADDITIONAL_QUERY_SYSTEM_PROMPT = """
# === 你的角色 ===
你是一个查询语句的生成专家，专门为精准匹配生成用于向量数据库检索的查询语句。

# === 你的核心任务 === 
向量数据库目前含有各种视频/图片/背景音乐的描述信息的向量。这些描述信息的内容包括**该视频/图片/背景音乐体现的内容和情绪，以及适用于何种短视频的制作**。
现在用户需要制作一段展会的宣传短视频，并提供展会的相关信息和具体需求。
**请根据所提供的【展会名称】、【展会活动描述】、【视频任务描述】、【用户具体需求】，生成一段查询语句，可以在向量数据中找到最匹配的视频/图片/背景音乐**。

# === 输出要求 ===
- **必须输出仅含一个键的JSON**：
    - `query`: 生成的查询语句，不超过100字
- **不得输出其他格式，或其他JSON键值，不得输出自然语言对话，不得输出换行符或其他任何无意义的符号**

# === 重要指示 ===
- 查询语句必须包括并完整体现展会行业和用户要求，**不得减少或更改**
- 查询语句会同时用于搜素视频/图片/背景音乐，所以内容**不用包括具体的媒体格式**

# === 输出示例 ===
用户输入: "【展会名称】：北京国际车展\n【展会活动描述】：2026北京国际车展，今年全国最大的车展，有很多新的车型和科技都将亮相\n【视频任务描述】：30秒的短视频\n【用户具体需求】：突出热闹、科技感"
你的输出:
{{"query": "汽车、人群、科技、氛围热烈、充满科技感。适合大型车展、新车发布会、新汽车科技发布会的短视频制作"}}

----------------------

用户输入: "【展会名称】：沧州福居家博会\n【展会活动描述】：很多大牌家装品牌参展，目标定位有装修需求的家庭\n【视频任务描述】：\n【用户具体需求】：希望能吸引家庭客户"
你的输出:
{{"query": "家具品牌、卫浴品牌、电器品牌、人群、热闹、全家参观。适合家装博览会、家电展销会、家具销售会会的短视频制作"}}
"""

DECIDE_ADDITIONAL_TTS_SYSTEM_PROMPT = """
# === 你的角色 ===
你是一个展会短视频配音音色策划专家，擅长根据展会行业属性、视频风格与目标受众，精准匹配最合适的TTS音色。

# === 你的核心任务 === 
用户需要制作一段展会的宣传短视频，并提供展会的行业和短视频的制作要求。
**请根据所提供的【展会名称】、【展会活动描述】、【视频任务描述】、【用户具体需求】，从下方提供的音色库中，精心挑选出最匹配的{top_k}种音色**。

# === 输出要求 ===
- **必须输出仅含两个键的JSON**：
    - `material_ids`: 长度为{top_k}的整数列表，对应所选音色的ID
    - `material_descs`: 长度为{top_k}的字符串列表，必须与下方音色描述**逐字完全一致**，顺序与material_ids一一对应
- **不得输出其他格式，或其他JSON键值，不得输出自然语言对话，不得输出换行符或其他任何无意义的符号**
- 严禁捏造、修改、拼接或使用音色库之外的ID与描述

# === 可选音色 ===
{formatted_tts_data}

# === 输出示例， **仅参考格式** ===
{{"material_ids": [138, 139, 149], "material_descs":["语调平稳、咬字柔和、自带治愈安抚力的女声音色", "声线甜美有活力的妹妹，活泼开朗，笑容明媚。", "声线阳光温暖、语气亲切，活力满满的少年音"]}}
"""

SELECT_SCRIPT_SYSTEM_PROMPT = """
# === 你的角色 ===
你是资深短视频文案选材师，擅长从各种文案中，筛选选出符合宣传展会短视频的最佳文案。

# === 你的核心任务 === 
根据展会信息，从备选文案中，严格按照输出要求，筛选出 {video_count} 个最适合制作宣传短视频的文案。
展会信息：{city_name}{show_title} | {show_desc} | {task_desc} | {ai_director}

# === 备选文案 ===
{script_candidates}

# === 输出要求 ===
- **必须输出一个含 `results` 键的 JSON 对象**
- `results` 的值是包含{video_count}个字典的列表，每个字典代表一个所选的文案。
- 每个文案字典必须包含且只能包含两个字段：
    - `material_id`: 1个整数，文案ID
    - `script_content`: 字符串，仅一字不差的文案内容
- **不得输出其他格式，或其他JSON键值，不得输出自然语言对话，不得输出换行符或其他任何无意义的符号**

# === 输出示例， **仅参考格式** ===
[{{"material_id": 1, "script_content": "2026北京国际汽车文化节来了，6月12日-15日，就在首钢会展中心，百余款新车齐亮相，车模表演现场抽奖high翻天，快带上你的家人朋友来逛展吧！"}}, {{"material_id": 3, "script_content": "2026北京国际汽车文化节将在6月12日-15日登录首钢会展中心，大牌新车云集，车模表演现场抽奖氛围热烈，热爱汽车的你千万不要错过！"}}]
"""

SCRIPT_SUB_SYSTEM_PROMPT = """
# === 你的角色 ===
你是资深短视频文案字幕调整师，擅长把已经生成好的短视频文案，调整成适合生成字幕的格式。

# === 你的核心任务 === 
把以下短视频文案分割成若干段，用 | 隔开，**每段不超过10个字**
## 分段逻辑：
- 标点符号（逗号、句号、顿号、感叹号、问号等）必需分段，同时删除原标点符号，只保留文本
- 标点符号分段后，如果有段落仍然超过十个字，合理地把该段落分成不超过十个字的若干段落，每个段落不少于五个字，**不得破坏单词结构，尽量尊重自然语义**

# === 输出要求 ===
- **必须输出分好段的字符串**
- 不要打断活动日期，尽量保证活动日期在一个段落内，该段落可略超过十个字
- 原文案的文本必需完全保留，除此之外不要有任何其他文字
- **不得输出其他任何格式**，不得聊天，不得输出 | 以外的任何符号

# === 输出示例 ===
用户输入：2026北京国际汽车文化节来了，6月12日-15日，就在首钢会展中心，百余款新车齐亮相，车模表演现场抽奖high翻天，快带上你的家人朋友来逛展吧！
你的输出：2026|北京国际汽车文化节|来了|6月12日-15日|就在首钢会展中心|百余款新车齐亮相|车模表演|现场抽奖high翻天|快带上你的家人|朋友来逛展吧
----------------------
用户输入：秋日的温情从一颗板栗开始。2025唐山板栗展将于8月13日至17日在唐山会展中心盛大开启。带上家人一起品尝香甜软糯的正宗唐山板栗，感受舌尖上的幸福滋味。
你的输出：秋日的温情|从一颗板栗开始|2025唐山板栗展|将于8月13日至17日|在唐山会展中心|盛大开启|带上家人一起|品尝香甜软糯的|正宗唐山板栗|感受舌尖上的幸福滋味
----------------------
用户输入：走进历史，感受文明。阳泉文物展将于5月1日至7日，在阳泉国际会展中心盛大开启。珍贵文物齐聚一堂，带您领略中华文化的深厚底蕴与无穷魅力。
你的输出：走进历史|感受文明|阳泉文物展|将于5月1日至7日|在阳泉国际会展中心|盛大开启|珍贵文物齐聚一堂|带您领略中华文化的|深厚底蕴与无穷魅力
"""

class Recommend:
    def __init__(self, request: RecommendationRequest, tts_data:list):
        self.task_id: int = request.task_id
        self.show_id: int = request.show_id
        self.org_id: int = request.org_id
        self.industry_id: int = request.industry_id
        self.task: TaskRequest = request.task
        self.result: dict = request.result
        self.collection_name = user_id_to_collection_name(self.org_id)
        self.total_duration:int = DEFAULT_TOTAL_DURATION if self.task.video_type in [0, 1] else DEFAULT_TOTAL_DURATION + 3
        # O(1) lookup map instead of repeated list scanning
        self.tts_data = tts_data
        self.tts_data_map = {t.get("material_id"): t for t in self.tts_data}

        # additional video footage, image, bgm and tts may need to be obtained
        self.additional_query_chain = self._generate_additional_query_chain()
        self.decide_additional_tts_chain = self._generate_decide_additional_tts_chain()

        # select scripts if required video to generate is less than scripts (regular recommend - prioritize selection)
        self.select_script_chain = self._generate_select_script_chain()

        # convert regular script to the version suitable for subtitle
        self.script_sub_chain = self._generate_script_sub_chain()

        self.milvus_client = AsyncMilvusClient(uri=MILVUS_URL, secure=False)

        # return requirement - there should be a key of retry_video returned
        self.retry_video = 0
        retry_source = self.task.retry_source
        if isinstance(retry_source, list) and retry_source:
            retry_source1 = retry_source[0]
            if isinstance(retry_source1, dict) and isinstance(retry_source1.get("source_video_id"), int):
                self.retry_video = retry_source1["source_video_id"]

    @staticmethod
    def _generate_additional_query_chain():
        additional_query_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", ADDITIONAL_QUERY_SYSTEM_PROMPT),
                ("human", "【展会名称】：{city_name}{show_title}\n【展会活动描述】：{show_desc}\n"
                          "【视频任务描述】：{task_desc}\n【用户具体需求】：{ai_director}")
            ]
        )
        llm_with_structured_output_query = llm_reasoning.with_structured_output(QueryResult)
        return additional_query_prompt | llm_with_structured_output_query

    @staticmethod
    def _generate_decide_additional_tts_chain():
        decide_additional_tts_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", DECIDE_ADDITIONAL_TTS_SYSTEM_PROMPT),
                ("human", "【展会名称】：{city_name}{show_title}\n【展会活动描述】：{show_desc}\n"
                          "【视频任务描述】：{task_desc}\n【用户具体需求】：{ai_director}")
            ]
        )
        llm_with_structured_output_decide_tts = llm_reasoning.with_structured_output(DecideTTSResult)
        return decide_additional_tts_prompt | llm_with_structured_output_decide_tts

    @staticmethod
    def _generate_select_script_chain():
        select_script_prompt = ChatPromptTemplate.from_messages([("system", SELECT_SCRIPT_SYSTEM_PROMPT)])
        llm_with_structured_output_select_script = llm_reasoning.with_structured_output(SelectScriptResult)
        return select_script_prompt | llm_with_structured_output_select_script

    @staticmethod
    def _generate_script_sub_chain():
        script_sub_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SCRIPT_SUB_SYSTEM_PROMPT),
                ("human", "用户输入：{script_reg}")
            ]
        )
        return script_sub_prompt | llm_regular

    def _generate_script_sub(self, scripts:dict)->str:
        try:
            script_reg = scripts["script_content"]
            script_sub = self.script_sub_chain.invoke({"script_reg": script_reg})
            return str(script_sub.content)
        except Exception as e:
            raise RuntimeError(e)

    async def _fetch_desc_from_milvus(self, m_id: int, partition: str) -> dict | None:
        """Safely query Milvus and extract desc_json."""
        try:
            res = await self.milvus_client.query(
                collection_name=self.collection_name,
                partition_names=[partition],
                filter=f"material_id == {m_id} and status == 1",
                output_fields=["desc_json"]
            )
            if not res:
                return None
            return res[0].get("desc_json")
        except Exception as e:
            video_director_logger.error(f"{e}")
            return None

    async def _fetch_vector_from_milvus(self, m_id: int, partition: str) -> list | None:
        """Safely query Milvus and extract vector."""
        try:
            res = await self.milvus_client.query(
                collection_name=self.collection_name,
                partition_names=[partition],
                filter=f"material_id == {m_id} and status == 1",
                output_fields=["vector"]
            )
            if not res:
                return None
            return res[0].get("vector")
        except Exception as e:
            video_director_logger.error(f"{e}")
            return None

    async def _additional_search(self)->dict:
        video_director_logger.info("搜索向量数据库开始")
        additional_search_results = {}
        query = DEFAULT_QUERY

        # -------- Generate query ----------
        try:
            video_director_logger.info("搜索向量数据库-生成查询语句开始")
            query_result = await self.additional_query_chain.ainvoke(
                {
                    'city_name': self.task.city_name,
                    'show_title':self.task.show_title,
                    'show_desc': self.task.show_desc,
                    'task_desc': self.task.task_desc,
                    'ai_director': self.task.ai_director,
                }
            )
            query = query_result.query
            video_director_logger.info("搜索向量数据库-生成查询语句完成")
            if not query or not isinstance(query, str):
                e_m = f"❌ 查询语句格式有误"
                video_director_logger.error(e_m)

        except Exception as e:
            e_m = f"❌ 查询语句生成有误: {e}"
            video_director_logger.error(e_m)

        # -------- Embed the Query ----------
        video_director_logger.info("搜索向量数据库-嵌入查询语句开始")
        query_emb:list = await embed_query(query)
        video_director_logger.info("搜索向量数据库-嵌入查询语句完成")

        # -------- Search from vector database ----------
        video_director_logger.info("搜索向量数据库-异步搜索开始")
        # Async concurrent loop
        search_tasks = []
        for material_type in MATERIAL_TYPE:
            search_tasks.append(self._search_one_type(material_type, query_emb, TOP_K_RECOMMENDATION))
        async_results = await asyncio.gather(*search_tasks, return_exceptions=True)

        for material_type, result in zip(MATERIAL_TYPE, async_results):
            if isinstance(result, Exception):
                video_director_logger.error(f"❌ '{query}'在向量数据库: {self.collection_name}查询{MATERIAL_CN[material_type]}失败: {result}")
                additional_search_results[material_type] = []
            elif not result or not result[0]:
                video_director_logger.warning(f"⚠️ '{query}'在向量数据库: {self.collection_name}查询{MATERIAL_CN[material_type]}无结果")
                additional_search_results[material_type] = []
            else:
                additional_search_results[material_type] = [
                    {
                        "material_id":int(m.get("material_id")),
                        "material_path": str(m.get("material_path")),
                        "material_desc": str(m.get("desc_json").get("overall_summary", "")),
                        "duration":float(m.get("desc_json").get("duration", DEFAULT_DURATION)),
                        "mandatory":False
                    }
                    for m in result[0]
                    if m.get("material_id") is not None
                ]
        video_director_logger.info("搜索向量数据库-异步搜索完成")
        return additional_search_results

    async def _search_one_type(self, material_type:Literal[*MATERIAL_TYPE], query_emb: list, limit: int):
        filter_expr = f"industry_id in [0, {int(self.industry_id)}] and status == 1"  # make filters
        return await self.milvus_client.search(
            collection_name=self.collection_name,
            partition_names=[material_type],
            data=[query_emb],
            filter=filter_expr,
            limit=limit,
            output_fields=["material_id", "material_path", "desc_json"],
            timeout=8.0
        )

    async def _decide_additional_tts(self)->list:
        video_director_logger.info("选择配音音色开始")
        formatted_tts_data = ""
        if isinstance(self.tts_data, list) and self.tts_data:
            for t_d in self.tts_data:
                formatted_tts_data += f"- ID: {t_d.get('material_id')}, 音色描述: {t_d.get('material_desc')}\n"

        try:
            decide_tts_results = await self.decide_additional_tts_chain.ainvoke(
                {
                    'top_k': TOP_K_RECOMMENDATION,
                    'city_name': self.task.city_name,
                    'show_title': self.task.show_title,
                    'show_desc': self.task.show_desc,
                    'task_desc': self.task.task_desc,
                    'ai_director': self.task.ai_director,
                    'formatted_tts_data': formatted_tts_data,
                }
            )
            # slightly validate the ids
            decide_tts_results.material_ids = [int(_id) for _id in decide_tts_results.material_ids]

            material_ids = decide_tts_results.material_ids
            material_descs = decide_tts_results.material_descs

            len_material_ids = len(material_ids)

            if len_material_ids != len(material_descs):
                e_m = f"❌ 符合要求的音色ID和描述数量不一致"
                video_director_logger.error(e_m)
                raise ValueError(e_m)

            video_director_logger.info("选择配音音色完成")

            return [
                {
                    "material_id":material_ids[i],
                    "material_desc":material_descs[i],
                    "mandatory":False
                }
                for i in range(len_material_ids)
            ]

        except Exception as e:
            e_m = f"❌ 选择配音音色有误: {e}"
            video_director_logger.error(e_m)
            raise Exception(e_m) from e

    @staticmethod
    def _merge_with_dedup(mandatory: list[dict], additional: list[dict]) -> list[dict]:
        """Merge mandatory pre-selected items with additional retrieved items.
        Automatically filters out duplicates from the retrieved list.
        Auto-filtered out "material_id"==0 in mandatory
        """
        mandatory_ids = {item["material_id"] for item in mandatory if item["material_id"] != 0}
        filtered_additional = [item for item in additional if item.get("material_id") not in mandatory_ids]
        return mandatory + filtered_additional

    @staticmethod
    def _deduplicate_by_material_id(data: list[dict]) -> list[dict]:
        seen = set()
        unique_data = []
        for item in data:
            mid = item["material_id"]  # or item.get("material_id") if key might be missing
            if mid not in seen:
                seen.add(mid)
                unique_data.append(item)
        return unique_data

    @staticmethod
    def _resolve(pool: dict, m_id: int, field: str) -> dict | None:
        mat = pool.get(m_id)
        if mat:
            m_path = mat.get("material_path") if isinstance(mat, dict) else ""
            return {
                "material_id": m_id,
                "material_path": m_path
            }
        else:  # sometimes LLM can hallucinate
            e_m1 = f"❌ LLM 返回了不存在的 {field} material_id: {m_id}"
            video_director_logger.error(e_m1)
            return None

    async def _gather_retry_data(self, retry_source: list) -> tuple[list, list, list, list, list, list]:
        # Define the default lists
        (retry_footage_opening, retry_footage_regular, retry_image,
         retry_bgm, retry_tts, retry_scripts) = [], [], [], [], [], []

        for item in retry_source:
            try:
                # footage:
                for video in item.get("videos"):
                    m_id = int(video.get('material_id'))
                    m_path = str(video.get('material_path'))
                    is_opening = video.get('is_opening')
                    if is_opening == 1:  # opening footage
                        partition, target = "footage_opening", retry_footage_opening
                    elif is_opening == 0:
                        partition, target = "footage_regular", retry_footage_regular
                    else:
                        continue
                    # Get the retrieved desc
                    desc_json = await self._fetch_desc_from_milvus(m_id, partition)
                    if not desc_json:
                        continue
                    target.append({
                        "material_id": m_id,
                        "material_path": m_path,
                        "material_desc": str(desc_json.get("overall_summary", "")),
                        "duration": float(desc_json.get("duration", DEFAULT_DURATION)),
                        "mandatory": False  # retry material only for reference
                    })

                # image - only one item in the list, directly execute:
                r_t = item.get("templates")
                r_t_m_id = int(r_t.get('material_id'))
                r_t_m_path = str(r_t.get('material_path'))
                r_t_desc_json = await self._fetch_desc_from_milvus(r_t_m_id, "image")
                if r_t_desc_json:
                    retry_image.append({
                        "material_id": r_t_m_id,
                        "material_path": r_t_m_path,
                        "material_desc": str(r_t_desc_json.get("overall_summary", "")),
                        "duration": 0.0,
                        "mandatory": False
                    })

                # bgm:
                for bgm in item.get("bgms"):
                    m_id = int(bgm.get('material_id'))
                    m_path = str(bgm.get('material_path'))
                    desc_json = await self._fetch_desc_from_milvus(m_id, "bgm")
                    if not desc_json:
                        continue
                    retry_bgm.append({
                        "material_id": m_id,
                        "material_path": m_path,
                        "material_desc": str(desc_json.get("overall_summary", "")),
                        "duration": float(desc_json.get("duration", DEFAULT_DURATION)),
                        "mandatory": False
                    })

                # tts - only one item in the list, directly execute:
                r_v_m_id = int(item.get("voices").get('material_id'))
                tts_info = self.tts_data_map.get(r_v_m_id)
                if tts_info:
                    retry_tts.append({
                        "material_id": r_v_m_id,
                        "material_desc": tts_info.get("material_desc") if isinstance(tts_info, dict) else DEFAULT_TTS_DESC,
                        # in tts_data, "material_desc" is already the description string
                        "mandatory": False
                    })

                # scripts
                r_s = item.get("scripts")
                if isinstance(r_s, dict) and r_s:
                    retry_scripts.append(r_s)

            except Exception as e:
                e_m = f"❌ 获取重试资源时出错: {e}"
                video_director_logger.error(e_m)
                continue

        return (retry_footage_opening, retry_footage_regular, retry_image,
                retry_bgm, retry_tts, retry_scripts)

    async def _gather_mult_data(self, mult_source: list) -> tuple[list, list, list, list, list, list]:
        # Define the default lists
        (mult_footage_opening, mult_footage_regular, mult_image,
         mult_bgm, mult_tts, mult_scripts) = [], [], [], [], [], []

        for item in mult_source:
            try:
                # footage:
                for video in item.get("videos"):
                    m_id = int(video.get('material_id'))
                    is_opening = video.get('is_opening')
                    if is_opening == 1:  # opening footage
                        partition, target = "footage_opening", mult_footage_opening
                    elif is_opening == 0:
                        partition, target = "footage_regular", mult_footage_regular
                    else:
                        continue
                    # Get the retrieved vector
                    vector = await self._fetch_vector_from_milvus(m_id, partition)
                    if not vector:
                        continue

                    # Multiplication: get similar
                    result = await self._search_one_type(partition, vector, TOP_K_MULT)
                    for m in result[0]:
                        if m.get("material_id") is not None:
                            target.append(
                                {
                                    "material_id": int(m.get("material_id")),
                                    "material_path": str(m.get("material_path")),
                                    "material_desc": str(m.get("desc_json").get("overall_summary", "")),
                                    "duration": float(m.get("desc_json").get("duration", DEFAULT_DURATION)),
                                    "mandatory": False
                                }
                            )

                # image - only one item in the list, directly execute:
                r_t = item.get("templates")
                r_t_m_id = int(r_t.get('material_id'))
                r_t_vector = await self._fetch_vector_from_milvus(r_t_m_id, "image")
                if r_t_vector:
                    result = await self._search_one_type("image", r_t_vector, TOP_K_MULT)
                    for m in result[0]:
                        if m.get("material_id") is not None:
                            mult_image.append(
                                {
                                    "material_id": int(m.get("material_id")),
                                    "material_path": str(m.get("material_path")),
                                    "material_desc": str(m.get("desc_json").get("overall_summary", "")),
                                    "duration": float(m.get("desc_json").get("duration", DEFAULT_DURATION)),
                                    "mandatory": False
                                }
                            )

                # bgm:
                for bgm in item.get("bgms"):
                    m_id = int(bgm.get('material_id'))
                    vector = await self._fetch_vector_from_milvus(m_id, "bgm")
                    if not vector:
                        continue

                    # Multiplication: get similar
                    result = await self._search_one_type("bgm", vector, TOP_K_MULT)
                    for m in result[0]:
                        if m.get("material_id") is not None:
                            mult_bgm.append(
                                {
                                    "material_id": int(m.get("material_id")),
                                    "material_path": str(m.get("material_path")),
                                    "material_desc": str(m.get("desc_json").get("overall_summary", "")),
                                    "duration": float(m.get("desc_json").get("duration", DEFAULT_DURATION)),
                                    "mandatory": False
                                }
                            )

                # tts - only one item in the list, directly execute, no vector database, no need to multiply:
                r_v_m_id = int(item.get("voices").get('material_id'))
                tts_info = self.tts_data_map.get(r_v_m_id)
                if tts_info:
                    mult_tts.append({
                        "material_id": r_v_m_id,
                        "material_desc": tts_info.get("material_desc"),
                        # in tts_data, "material_desc" is already the description string
                        "mandatory": False
                    })

                # scripts
                r_s = item.get("scripts")
                if isinstance(r_s, dict) and r_s:
                    mult_scripts.append(r_s)

            except Exception as e:
                e_m = f"❌ 获取裂变资源时出错: {e}"
                video_director_logger.error(e_m)
                continue
        # For data retrieved from vector database, there may be duplicates, only keep the first one with same material_id
        mult_footage_opening = self._deduplicate_by_material_id(mult_footage_opening) if mult_footage_opening else []
        mult_footage_regular = self._deduplicate_by_material_id(mult_footage_regular) if mult_footage_regular else []
        mult_image = self._deduplicate_by_material_id(mult_image) if mult_image else []
        mult_bgm = self._deduplicate_by_material_id(mult_bgm) if mult_bgm else []

        print(f"裂变结果:\n"
              f"  - mult_footage_opening长度: {len(mult_footage_opening)}\n"
              f"  - mult_footage_regular长度: {len(mult_footage_regular)}\n"
              f"  - mult_image长度: {len(mult_image)}\n"
              f"  - mult_bgm长度: {len(mult_bgm)}")

        return (mult_footage_opening, mult_footage_regular, mult_image,
                mult_bgm, mult_tts, mult_scripts)

    @staticmethod
    def _build_info(items: list[dict], include_duration: bool = False) -> str:
        if not items:
            return ""
        lines = []
        for item in items:
            dur_part = f"时长{item.get('duration', DEFAULT_DURATION)}秒, " if include_duration else ""
            mand_part = "必需" if item.get("mandatory") else "非必需"
            desc_part = item.get("material_desc", "").replace("\n",
                                                              "\n    ")  # add some indention for line changing
            lines.append(f"- ID:{item.get('material_id')}, {dur_part}{mand_part}, {desc_part}")
        return "\n".join(lines)

    def _select_best_prompt_no_script(self, footage_opening: list, footage_regular: list,
                                      image: list, bgm: list)->str:
        fo_info = self._build_info(footage_opening, include_duration=True)
        fr_info = self._build_info(footage_regular, include_duration=True)
        img_info = self._build_info(image)
        bgm_info = self._build_info(bgm, include_duration=True)

        return f"""
# === 你的角色 ===
你是资深短视频导演，擅长从素材库中挑选最佳组合，制作符合展会调性的高质量短视频。

# === 你的核心任务 === 
制作 {self.total_duration} 秒的展会宣传短视频，尽量生成 {self.task.video_count} 个版本 (如果提供的视频、贴图、音乐等素材不够生成足够多的不同版本，则能生成多少就生成多少)。
展会信息：{self.task.city_name}{self.task.show_title} | {self.task.show_desc} | {self.task.task_desc} | {self.task.ai_director}

# === 版本构成要求 ===
每个版本必须包含：
- 1个开场白视频素材
- 1个或多个普通视频素材
- 1个文字贴图
- 1段背景音乐。但当选材库没有背景音乐时，可以没有背景音乐
⚠️ 开场白 + 所有普通视频素材的总时长**至少{self.total_duration}秒**，可添加多段普通视频素材增加总时长。
⚠️ 开场白 + 所有普通视频素材的总时长**最多{self.total_duration + 10}秒**。
⚠️ 素材使用规则：
   - 标记为【必需】的素材：必须在【至少一个】版本中出现（不需要每个版本都有）
   - 非必需素材：鼓励在不同版本中轮换使用，增加多样性
   - 各版本之间至少有两个元素不同（如：不同的开场视频+不同的贴图）

# === 选材库 ===
## 开场白视频
{fo_info if fo_info else "（无可用素材）"}
## 普通视频
{fr_info if fr_info else "（无可用素材）"}
## 文字贴图
{img_info if img_info else "（无可用素材）"}
## 背景音乐
{bgm_info if bgm_info else "（无可用素材）"}

# === 输出要求 ===
- **必须输出一个含 `plans` 键的 JSON 对象**
- `plans` 的值是包含{self.task.video_count}个字典的列表，每个字典代表一个版本 (如果提供的视频、贴图、音乐等素材不够生成足够多的不同版本，则能生成多少就生成多少)。
- 每个版本字典必须包含且只能包含以下字段：
    - `footage_opening`: 1个整数，开场白视频素材的material_id
    - `footage_regular`: 包含整数的数组，整数为1个或多个普通视频素材的material_id
    - `image`: 1个整数，文字贴图的material_id
    - `bgm`: 1个整数或null，背景音乐的material_id，无可用素材时可为空
- **不得输出其他格式，或其他JSON键值，不得输出自然语言对话，不得输出换行符或其他任何无意义的符号**

# === 输出示例， **仅参考格式** ===
{{"plans": [{{"footage_opening": 4, "footage_regular": [326], "image": 145,"bgm": 400}}, {{"footage_opening": 15, "footage_regular": [327, 323], "image": 17, "bgm": 5}}]}}
"""

    def _select_best_prompt_script(self, footage_opening: list, footage_regular: list,
                                   image: list, bgm: list, tts: list, script:dict, version_per_script: int) -> str:

        fo_info = self._build_info(footage_opening, include_duration=True)
        fr_info = self._build_info(footage_regular, include_duration=True)
        img_info = self._build_info(image)
        bgm_info = self._build_info(bgm, include_duration=True)
        tts_info = self._build_info(tts)

        return f"""
# === 你的角色 ===
你是资深短视频导演，擅长从素材库中挑选最佳组合，制作符合展会调性的高质量短视频。

# === 你的核心任务 === 
制作 {self.total_duration} 秒的展会宣传短视频，尽量生成 {version_per_script} 个版本 (如果提供的视频、贴图、音乐、语音音色等素材不够生成足够多的不同版本，则能生成多少就生成多少)。
展会信息：{self.task.city_name}{self.task.show_title} | {self.task.show_desc} | {self.task.task_desc} | {self.task.ai_director}
短视频口播旁白：{script.get("script_content")}

# === 版本构成要求 ===
每个版本必须包含：
- 1个开场白视频素材
- 1个或多个普通视频素材
- 1个文字贴图
- 1段背景音乐。但当选材库没有背景音乐时，可以没有背景音乐
- 1个语音音色，用于解说口播旁白
⚠️ 开场白 + 所有普通视频素材的总时长**至少{self.total_duration}秒**，可添加多段普通视频素材增加总时长。
⚠️ 开场白 + 所有普通视频素材的总时长**最多{self.total_duration + 10}秒**。
⚠️ 素材使用规则：
   - 标记为【必需】的素材：必须在【至少一个】版本中出现（不需要每个版本都有）
   - 非必需素材：鼓励在不同版本中轮换使用，增加多样性
   - 各版本之间至少有两个元素不同（如：不同的开场视频+不同的贴图）

# === 选材库 ===
## 开场白视频
{fo_info if fo_info else "（无可用素材）"}
## 普通视频
{fr_info if fr_info else "（无可用素材）"}
## 文字贴图
{img_info if img_info else "（无可用素材）"}
## 背景音乐
{bgm_info if bgm_info else "（无可用素材）"}
## 语音音色
{tts_info if tts_info else "（无可用素材）"}

# === 输出要求 ===
- **必须输出一个含 `plans` 键的 JSON 对象**
- `plans` 的值是包含 {version_per_script} 个字典的列表，每个字典代表一个版本 (如果提供的视频、贴图、音乐、语音音色等素材不够生成足够多的不同版本，则能生成多少就生成多少)。
- 每个版本字典必须包含且只能包含以下字段：
    - `footage_opening`: 1个整数，开场白视频素材的material_id
    - `footage_regular`: 包含整数的数组，整数为1个或多个普通视频素材的material_id
    - `image`: 1个整数，文字贴图的material_id
    - `bgm`: 1个整数或null，背景音乐的material_id，无可用素材时可为空
    - `tts`: 1个整数，语音音色的material_id
- **不得输出其他格式，或其他JSON键值，不得输出自然语言对话，不得输出换行符或其他任何无意义的符号**

# === 输出示例， **仅参考格式** ===
{{"plans": [{{"footage_opening": 4, "footage_regular": [328], "image": 145, "bgm": 400, "tts": 141}}, {{"footage_opening": 15, "footage_regular": [327, 323], "image": 5, "bgm": 6, "tts": 142}}]}}
"""

    async def _select_best(self, footage_opening: list, footage_regular: list,
                           image: list, bgm: list, tts: list, scripts:list
                           )->RecommendationResult:
        """
        select the appropriate materials and give the final recommendations
        :param footage_opening: [{"material_id": 1, "material_path": "xx", "material_desc": "xx", "duration": 1.0, "mandatory": False}...]
        :param footage_regular: [{"material_id": 1, "material_path": "xx", "material_desc": "xx", "duration": 1.0, "mandatory": False}...]
        :param image: [{"material_id": 1, "material_path": "xx", "material_desc": "xx", "duration": 0.0,"mandatory": False}...]
        :param bgm: [{"material_id": 1, "material_path": "xx", "material_desc": "xx", "duration": 1.0,"mandatory": False}...]
        :param tts: [{"material_id": 1, "material_desc": "xx", "mandatory": False}...]
        :param scripts: ["xxx", "xxx", "xxx"]
        :return:
        """
        if not scripts:
            video_director_logger.info("推荐的最终选择范围: \n"
                                       f"  - footage_opening长度: {len(footage_opening)}\n"
                                       f"  - footage_regular长度: {len(footage_regular)}\n"
                                       f"  - image长度: {len(image)}\n"
                                       f"  - bgm长度: {len(bgm)}\n"
                                       f"  - tts长度: {len(tts)}\n"
                                       f"  - scripts长度: {len(scripts)}\n"
                                       )
            select_best_prompt = self._select_best_prompt_no_script(footage_opening, footage_regular, image, bgm)
            print(f"无台词推荐提示词:{select_best_prompt}")

            # Build O(1) lookup maps for fast material resolution
            material_pool = {
                "footage_opening": {m["material_id"]: m for m in footage_opening},
                "footage_regular": {m["material_id"]: m for m in footage_regular},
                "image": {m["material_id"]: m for m in image},
                "bgm": {m["material_id"]: m for m in bgm}
            }

            recommended_plans = []
            try:
                llm_response = llm_reasoning.with_structured_output(VideoPlansResponseNoScript).invoke(select_best_prompt)

                if not llm_response or not llm_response.plans:
                    e_m = "❌ LLM 返回了空结果或未生成有效版本。"
                    video_director_logger.error(e_m)
                    raise ValueError(e_m)

                # 3. Map IDs back to full material objects & validate
                for plan in llm_response.plans:
                    resolved = {
                        "retry_video" : self.retry_video,
                        "video_opening_id": self._resolve(material_pool["footage_opening"], plan.footage_opening,"开场白视频") or {},
                        "video_regular_ids": [
                            res for mid in plan.footage_regular
                            if (res := self._resolve(material_pool["footage_regular"], mid, "普通视频")) is not None
                        ],  # := if res is None, skip. Only keep res in the list when it is not None
                        "template_id": self._resolve(material_pool["image"], plan.image, "贴图") or {},
                        "bgm_ids": (self._resolve(material_pool["bgm"], plan.bgm,"背景音乐") or {}) if plan.bgm is not None else None,
                        # bgm id can be None
                        "voice_ids": {},
                        "scripts": {}
                    }
                    recommended_plans.append(resolved)

            except Exception as e:
                e_m = f"❌ 短视频组合推荐失败: {e}"
                video_director_logger.error(e_m)
                raise RuntimeError(e_m) from e

            return RecommendationResult(
                task_id=self.task_id,
                show_id=self.show_id,
                org_id=self.org_id,
                industry_id=self.industry_id,
                results=recommended_plans
            )

        else:
            # Tell if video count is less than the mount of scripts
            if self.task.video_count<len(scripts):
                # Create script candidates
                script_candidates = ""
                for item in scripts:
                    if isinstance(item, dict):
                        script_candidates += f"- 文案ID: {item.get('material_id')} - 文案内容: {item.get('script_content')}\n"
                try:
                    selected_results = await self.select_script_chain.ainvoke(
                        {
                            'video_count': self.task.video_count,
                            'city_name': self.task.city_name,
                            'show_title': self.task.show_title,
                            'show_desc': self.task.show_desc,
                            'task_desc': self.task.task_desc,
                            'ai_director': self.task.ai_director,
                            'script_candidates': script_candidates,
                        }
                    )
                    # Use selected results to replace scripts
                    scripts = []
                    for item in selected_results.results:
                        scripts.append({
                            "material_id": item.material_id,
                            "script_content": item.script_content
                        })
                    video_director_logger.info(f"🎁 有文案推荐: 仅使用{len(scripts)}套文案推荐方案")
                except Exception as e:
                    video_director_logger.error(f"❌ 选择文案出错: {e}")

                version_per_script = 1
            else:
                version_per_script = int(((self.task.video_count-1)//len(scripts)) + 1)
                video_director_logger.info(f"🎁 有文案推荐: {len(scripts)}段文案，每段生成{version_per_script}套推荐方案")

            video_director_logger.info("推荐的最终选择范围: \n"
                                       f"  - footage_opening长度: {len(footage_opening)}\n"
                                       f"  - footage_regular长度: {len(footage_regular)}\n"
                                       f"  - image长度: {len(image)}\n"
                                       f"  - bgm长度: {len(bgm)}\n"
                                       f"  - tts长度: {len(tts)}\n"
                                       f"  - scripts长度: {len(scripts)}\n"
                                       )

            script_plans_lookup = {}

            #iterate each script
            len_scripts = len(scripts)
            for i in range(len_scripts):
                video_director_logger.info(f"开始为第{i+1}/{len_scripts}段文案推荐")
                script = scripts[i]

                # generate script for subtitles
                try:
                    script_sub = self._generate_script_sub(script)
                except Exception as e:
                    video_director_logger.error(f"生成字幕剧本有误: {e}")
                    script_sub = script

                select_best_prompt = self._select_best_prompt_script(footage_opening, footage_regular, image, bgm, tts,
                                                                     script, version_per_script)
                video_director_logger.info(f"第{i+1}/{len_scripts}段文案推荐提示词:{select_best_prompt}")
                # Build O(1) lookup maps for fast material resolution
                material_pool = {
                    "footage_opening": {m["material_id"]: m for m in footage_opening},
                    "footage_regular": {m["material_id"]: m for m in footage_regular},
                    "image": {m["material_id"]: m for m in image},
                    "bgm": {m["material_id"]: m for m in bgm},
                    "tts": {m["material_id"]: m for m in tts}
                }

                try:
                    llm_response = llm_reasoning.with_structured_output(VideoPlansResponseScript).invoke(select_best_prompt)

                    if not llm_response or not llm_response.plans:
                        e_m = "❌ LLM 返回了空结果或未生成有效版本。"
                        video_director_logger.error(e_m)
                        continue

                    # 3. Map IDs back to full material objects & validate
                    recommended_plans = []
                    for plan in llm_response.plans:
                        resolved = {
                            "retry_video": self.retry_video,
                            "video_opening_id": self._resolve(material_pool["footage_opening"], plan.footage_opening,
                                                         "开场白视频") or {},
                            "video_regular_ids": [
                                res for mid in plan.footage_regular
                                if (res := self._resolve(material_pool["footage_regular"], mid, "普通视频")) is not None
                            ], # := if res is None, skip. Only keep res in the list when it is not None
                            "template_id": self._resolve(material_pool["image"], plan.image, "贴图") or {},
                            "bgm_ids": (self._resolve(material_pool["bgm"], plan.bgm,"背景音乐") or {}) if plan.bgm is not None else None,  # bgm id can be None
                            "voice_ids": self._resolve(material_pool["tts"], plan.tts, "语音音色") or {},
                            "scripts": script,
                            "script_sub": script_sub
                        }
                        recommended_plans.append(resolved)

                    # Attach the script plan to the lookup
                    if script.get("material_id"):
                        script_plans_lookup[script["material_id"]] = recommended_plans

                    video_director_logger.info(f"完成第{i + 1}/{len_scripts}段文案推荐")

                except Exception as e:
                    e_m = f"❌ 第{i + 1}/{len_scripts}段文案推荐失败: {e}"
                    video_director_logger.error(e_m)
                    continue

            # Now we have a full lookup table of plans by script, integrate them and output the final results
            integrated_recommended_plans = []
            if script_plans_lookup:
                sorted_keys = sorted(script_plans_lookup.keys())
                lists_in_order = [script_plans_lookup[k] for k in sorted_keys]
                integrated_recommended_plans = [
                    plan for group in zip_longest(*lists_in_order)
                    for plan in group if plan is not None
                ]
                integrated_recommended_plans = integrated_recommended_plans[:int(self.task.video_count)]

            return RecommendationResult(
                task_id=self.task_id,
                show_id=self.show_id,
                org_id=self.org_id,
                industry_id=self.industry_id,
                results=integrated_recommended_plans
            )

    async def _regular_recommend(self)-> RecommendationResult:
        """Directly generate"""
        start_time = datetime.now()
        video_director_logger.info("🔹 ▶️ 普通成片推荐方案开始")
        video_director_logger.info(f"用户指定模板(result): {self.result}")
        # ------- Recommend plans -------
        if self.task.template_strategy==1: # Prioritize selected, no need to have additional materials
            # ------- Process Footage (Opening & Regular) -------
            footage_opening, footage_regular = [], []
            for item in self.result.get("videos", []):
                m_id = int(item.get('material_id'))
                m_path = str(item.get('material_path'))
                is_opening = item.get('is_opening')
                if is_opening == 1: # opening footage
                    partition, target = "footage_opening", footage_opening
                elif is_opening == 0:
                    partition, target = "footage_regular", footage_regular
                else:
                    continue

                # Get the retrieved desc
                desc_json = await self._fetch_desc_from_milvus(m_id, partition)
                if not desc_json: # Sometimes the footage is not in Milvus yet, we use the default desc
                    target.append({
                        "material_id": m_id,
                        "material_path": m_path,
                        "material_desc": DEFAULT_FOOTAGE_DESC,
                        "duration": DEFAULT_DURATION,
                        "mandatory": True
                    })
                else:
                    target.append({
                        "material_id": m_id,
                        "material_path": m_path,
                        "material_desc": str(desc_json.get("overall_summary", "")),
                        "duration":float(desc_json.get("duration", DEFAULT_DURATION)),
                        "mandatory": True
                    })

            # footage_opening = self._merge_with_dedup(footage_opening, additional_results.get("footage_opening", []))
            # footage_regular = self._merge_with_dedup(footage_regular, additional_results.get("footage_regular", []))

            # ------- Process Images -------
            image = []
            for item in self.result.get("templates", []):
                m_id = int(item.get('material_id'))
                m_path = str(item.get('material_path'))
                desc_json = await self._fetch_desc_from_milvus(m_id, "image")
                if not desc_json:  # Sometimes the image is not in Milvus yet, we use the default desc
                    image.append({
                        "material_id": m_id,
                        "material_path": m_path,
                        "material_desc": DEFAULT_IMAGE_DESC,
                        "duration": 0.0,
                        "mandatory": True
                    })
                else:
                    image.append({
                        "material_id": m_id,
                        "material_path": m_path,
                        "material_desc": str(desc_json.get("overall_summary", "")),
                        "duration": 0.0,
                        "mandatory": True
                    })
            # image = self._merge_with_dedup(image, additional_results.get("image", []))

            # ------- Process BGM -------
            bgm = []
            bgm_need_search = False
            for item in self.result.get("bgms", []):
                m_id = int(item.get('material_id'))
                m_path = str(item.get('material_path'))

                if m_id != 0:
                    desc_json = await self._fetch_desc_from_milvus(m_id, "bgm")
                    if not desc_json: # sometimes the given bgm is not in vector database, we directly use what is given
                        bgm.append({
                            "material_id": m_id,
                            "material_path": m_path,
                            "material_desc": DEFAULT_BGM_DESC, # Default bgm description
                            "duration": DEFAULT_DURATION,
                            "mandatory": True
                        })
                    else:
                        bgm.append({
                            "material_id": m_id,
                            "material_path": m_path,
                            "material_desc": str(desc_json.get("overall_summary", "")),
                            "duration": float(desc_json.get("duration", DEFAULT_DURATION)),
                            "mandatory": True
                        })

                else: # Special rule: if BGM material id == 0 (there will be only one item in fact), directly use retrieval results
                    bgm_need_search = True

            # ------- Search additional materials from vector database -------
            if (not footage_regular) or (not footage_opening) or (not image) or bgm_need_search:
                # We must gurantee that there are content in footage_regular, footage_opening and image.
                # Also when there are id=0 in bgm, we need to guarantee bgm as well.
                video_director_logger.info("🔹 🔎 普通成片-选择优先，搜索向量数据库")
                try:
                    additional_results = await self._additional_search()
                    # {
                    #     "footage_regular":[{"material_id": 1, "material_path": "xx", "material_desc": "xx", "duration": 1.0,"mandatory": False}...],
                    #     "footage_opening":[{"material_id": 1, "material_path": "xx", "material_desc": "xx", "duration": 1.0,"mandatory": False}...],
                    #     "image":[{"material_id": 1, "material_path": "xx", "material_desc": "xx", "duration": 0.0,"mandatory": False}...],
                    #     "bgm":[{"material_id": 1, "material_path": "xx", "material_desc": "xx", "duration": 1.0,"mandatory": False}...],
                    # }
                except Exception as e:
                    e_m = f"❌ 向量数据库搜索失败: {e}"
                    video_director_logger.error(e_m)
                    additional_results = {
                        "footage_regular": [],
                        "footage_opening": [],
                        "image": [],
                        "bgm": []
                    }
                if not footage_regular:
                    footage_regular = additional_results.get("footage_regular", [])
                if not footage_opening:
                    footage_opening = additional_results.get("footage_opening", [])
                if not image:
                    image = additional_results.get("image", [])
                if bgm_need_search:
                    bgm = additional_results.get("bgm", [])

            # Special rule: BGM remains empty if no pre-selected items exist
            # bgm = self._merge_with_dedup(bgm, additional_results.get("bgm", [])) if self.result.get("bgms", []) else []

            # ------- Process TTS -------
            tts = []
            for item in self.result.get("voices", []):
                m_id = int(item.get('material_id'))
                if m_id != 0:
                    tts_info = self.tts_data_map.get(m_id)
                    if not tts_info:
                        tts.append({
                            "material_id": m_id,
                            "material_desc": DEFAULT_TTS_DESC,
                            "mandatory": True
                        })
                    else:
                        tts.append({
                            "material_id": m_id,
                            "material_desc": tts_info.get("material_desc", DEFAULT_TTS_DESC) if isinstance(tts_info, dict) else DEFAULT_TTS_DESC,
                            # in tts_data, "material_desc" is already the description string
                            "mandatory": True
                        })
                else: # Special rule: if tts material id == 0 (there will be only one item in fact), directly use retrieval results
                    # ------- Decide additional voices -------
                    try:
                        additional_tts = await self._decide_additional_tts()
                        # [{"material_id": 1, "material_desc": "xx", "mandatory": False}...]
                    except Exception as e:
                        e_m = f"❌ 选择配音音色失败: {e}"
                        video_director_logger.error(e_m)
                        additional_tts = []
                    tts = additional_tts
            # Add additional search results after processing
            # tts = self._merge_with_dedup(tts, additional_tts)

            results = await self._select_best(footage_opening, footage_regular, image, bgm, tts,
                                           scripts = self.result.get("scripts", []))
            duration = (datetime.now() - start_time).total_seconds()
            video_director_logger.info(f"🔹 ✅ 普通成片推荐方案结束, 优先已选, 耗时{round(duration, 2)}秒\n"
                                       f"推荐结果:{results}")
            return results

        elif self.task.template_strategy==2:
            # ------- Search additional materials from vector database -------
            video_director_logger.info("🔹 🔎 普通成片-智能匹配，搜索向量数据库")
            try:
                additional_results = await self._additional_search()
                # {
                #     "footage_regular":[{"material_id": 1, "material_path": "xx", "material_desc": "xx", "duration": 1.0,"mandatory": False}...],
                #     "footage_opening":[{"material_id": 1, "material_path": "xx", "material_desc": "xx", "duration": 1.0,"mandatory": False}...],
                #     "image":[{"material_id": 1, "material_path": "xx", "material_desc": "xx", "duration": 0.0,"mandatory": False}...],
                #     "bgm":[{"material_id": 1, "material_path": "xx", "material_desc": "xx", "duration": 1.0,"mandatory": False}...],
                # }
            except Exception as e:
                e_m = f"❌ 向量数据库搜索失败: {e}"
                video_director_logger.error(e_m)
                additional_results = {
                    "footage_regular": [],
                    "footage_opening": [],
                    "image": [],
                    "bgm": []
                }

            # ------- Decide additional voices -------
            try:
                additional_tts = await self._decide_additional_tts()
                # [{"material_id": 1, "material_desc": "xx", "mandatory": False}...]
            except Exception as e:
                e_m = f"❌ 选择配音音色失败: {e}"
                video_director_logger.error(e_m)
                additional_tts = []

            results = await self._select_best(
                additional_results.get("footage_opening", []),
                additional_results.get("footage_regular", []),
                additional_results.get("image", []),
                additional_results.get("bgm", []) if self.result.get("bgms", []) else [],
                additional_tts,
                self.result.get("scripts", [])
            )
            duration = (datetime.now() - start_time).total_seconds()
            video_director_logger.info(f"🔹 ✅ 普通成片推荐方案结束, 智能匹配，耗时{round(duration, 2)}秒\n"
                                       f"推荐结果:{results}")
            return results
        else:
            e_m = f"❌ 成片策略输入值有误：只能是1或2，当前输入{self.task.template_strategy}"
            video_director_logger.error(e_m)
            raise ValueError(e_m)

    async def _regular_recommend_with_retry(self)-> RecommendationResult:
        """Generate with reference of retry sources"""
        video_director_logger.info("🔄 ▶️ 普通成片重新生成推荐方案开始")
        start_time = datetime.now()

        # ------- If no retry_source, do _regular_recommend -------
        retry_source = self.task.retry_source
        if not isinstance(retry_source, list) or not retry_source:
            return await self._regular_recommend()

        # ------- Gather retry sources for further process -------
        (retry_footage_opening, retry_footage_regular, retry_image, retry_bgm,
         retry_tts, retry_scripts) = await self._gather_retry_data(retry_source)

        # ------- Search additional materials from vector database -------
        video_director_logger.info("🔄 🔎 普通成片重新生成搜索向量数据库")
        try:
            additional_results = await self._additional_search()
            # {
            #     "footage_regular":[{"material_id": 1, "material_path": "xx", "material_desc": "xx", "duration": 1.0,"mandatory": False}...],
            #     "footage_opening":[{"material_id": 1, "material_path": "xx", "material_desc": "xx", "duration": 1.0,"mandatory": False}...],
            #     "image":[{"material_id": 1, "material_path": "xx", "material_desc": "xx", "duration": 0.0,"mandatory": False}...],
            #     "bgm":[{"material_id": 1, "material_path": "xx", "material_desc": "xx", "duration": 1.0,"mandatory": False}...],
            # }
        except Exception as e:
            e_m = f"❌ 向量数据库搜索失败: {e}"
            video_director_logger.error(e_m)
            additional_results = {
                "footage_regular":[],
                "footage_opening":[],
                "image":[],
                "bgm":[]
            }

        # ------- Decide additional voices -------
        try:
            additional_tts = await self._decide_additional_tts()
            # [{"material_id": 1, "material_desc": "xx", "mandatory": False}...]
        except Exception as e:
            e_m = f"❌ 选择配音音色失败: {e}"
            video_director_logger.error(e_m)
            additional_tts = []

        # ------- Recommend plans -------
        footage_opening = self._merge_with_dedup(additional_results.get("footage_opening", []), retry_footage_opening)
        footage_regular = self._merge_with_dedup(additional_results.get("footage_regular", []), retry_footage_regular)
        image = self._merge_with_dedup(additional_results.get("image", []), retry_image)
        bgm = self._merge_with_dedup(additional_results.get("bgm", []), retry_bgm) if retry_bgm else [] # Only have bgm when retry bgm exists
        tts = self._merge_with_dedup(additional_tts, retry_tts)
        scripts = self._merge_with_dedup(self.result.get("scripts", []), retry_scripts) if retry_scripts else [] # Only have scripts when retry or mult scripts exists
        print(f"scripts: {scripts}, retry_scripts: {retry_scripts}")

        results = await self._select_best(footage_opening, footage_regular, image, bgm, tts, scripts)
        duration = (datetime.now() - start_time).total_seconds()
        video_director_logger.info(f"🔄 ✅ 普通成片重新生成推荐方案结束, 耗时{round(duration, 2)}秒\n"
                                   f"推荐结果:{results}")
        return results

    async def _mult_recommend(self) -> RecommendationResult:
        """Generate with multiplication with or without retry, same logic"""
        video_director_logger.info("⚛️ ▶️ 爆款裂变推荐方案开始")
        start_time = datetime.now()

        # ------- If no retry_source and mult_source, do _regular_recommend -------
        retry_source = self.task.retry_source
        mult_source = self.task.mult_source

        retry_valid = isinstance(retry_source, list) and retry_source
        mult_valid = isinstance(mult_source, list) and mult_source

        if not retry_valid and not mult_valid:
            return await self._regular_recommend()

        # ------- Gather retry sources for further process -------
        (retry_footage_opening, retry_footage_regular, retry_image, retry_bgm,
         retry_tts, retry_scripts) = await self._gather_retry_data(retry_source)

        # ------- Gather mult sources for further process -------
        (mult_footage_opening, mult_footage_regular, mult_image, mult_bgm,
         mult_tts, mult_scripts) = await self._gather_mult_data(mult_source)

        # ------- Search additional materials from vector database -------
        video_director_logger.info("⚛️ 🔎 爆款裂变搜索向量数据库")
        try:
            additional_results = await self._additional_search()
            # {
            #     "footage_regular":[{"material_id": 1, "material_path": "xx", "material_desc": "xx", "duration": 1.0,"mandatory": False}...],
            #     "footage_opening":[{"material_id": 1, "material_path": "xx", "material_desc": "xx", "duration": 1.0,"mandatory": False}...],
            #     "image":[{"material_id": 1, "material_path": "xx", "material_desc": "xx", "duration": 0.0,"mandatory": False}...],
            #     "bgm":[{"material_id": 1, "material_path": "xx", "material_desc": "xx", "duration": 1.0,"mandatory": False}...],
            # }
        except Exception as e:
            e_m = f"❌ 向量数据库搜索失败: {e}"
            video_director_logger.error(e_m)
            additional_results = {
                "footage_regular": [],
                "footage_opening": [],
                "image": [],
                "bgm": []
            }

        # ------- Decide additional voices -------
        try:
            additional_tts = await self._decide_additional_tts()
            # [{"material_id": 1, "material_desc": "xx", "mandatory": False}...]
        except Exception as e:
            e_m = f"❌ 选择配音音色失败: {e}"
            video_director_logger.error(e_m)
            additional_tts = []

        # ------- Recommend plans -------
        footage_opening = self._merge_with_dedup(additional_results.get("footage_opening", []), retry_footage_opening)
        footage_opening = self._merge_with_dedup(footage_opening, mult_footage_opening)

        footage_regular = self._merge_with_dedup(additional_results.get("footage_regular", []), retry_footage_regular)
        footage_regular = self._merge_with_dedup(footage_regular, mult_footage_regular)

        image = self._merge_with_dedup(additional_results.get("image", []), retry_image)
        image = self._merge_with_dedup(image, mult_image)

        if retry_bgm or mult_bgm: #Only have bgm when retry or mult bgm exists
            bgm = self._merge_with_dedup(additional_results.get("bgm", []), retry_bgm)
            bgm = self._merge_with_dedup(bgm, mult_bgm)
        else:
            bgm = []

        tts = self._merge_with_dedup(additional_tts, retry_tts)
        tts = self._merge_with_dedup(tts, mult_tts)

        if retry_scripts or mult_scripts:  # Only have bgm when retry or mult scripts exists
            scripts = self._merge_with_dedup(self.result.get("scripts", []), retry_scripts)
            scripts = self._merge_with_dedup(scripts, mult_scripts)
        else:
            scripts = []
        print(f"scripts: {scripts}, retry_scripts: {retry_scripts}, mult_scripts: {mult_scripts}")

        results = await self._select_best(footage_opening, footage_regular, image, bgm, tts, scripts)
        duration = (datetime.now() - start_time).total_seconds()
        video_director_logger.info(f"⚛️ ✅ 爆款裂变推荐方案结束, 耗时{round(duration, 2)}秒\n"
                                   f"推荐结果:{results}")
        return results

    async def execute(self)->RecommendationResult:
        task_type = self.task.task_type  # 1, 2, 3, or 4
        video_director_logger.info(
            f"💡 {TASK_TYPE_CN[task_type]}任务:\n"
            f"  - 任务ID: {self.task_id}\n"
            f"  - 展会活动ID: {self.show_id}\n"
            f"  - 组织ID: {self.org_id}\n"
            f"  - 行业ID: {self.industry_id}")

        if task_type == 1:
            return await self._regular_recommend()
        elif task_type == 2:
            return await self._regular_recommend_with_retry()
        elif task_type in (3, 4):
            return await self._mult_recommend()
        else:
            e_m = f"❌ 不支持任务类型: {task_type}"
            video_director_logger.error(e_m)
            raise ValueError(e_m)


if __name__ == "__main__":
    task1 = TaskRequest(
        task_desc = "生成短视频",  # user's description on task, can be empty
        task_type = 1,  # 1 普通成片  2 普通成片重新生成  3 爆款裂变   4 爆款裂变重新生成
        video_type = 1, # 1 口播从开场白视频开始 2 口播在开场白视频结束后开始，视频时长限制要稍微尝一下
        ai_director = "突出人多热闹，商业感强",  # user requirements, can be empty
        template_strategy = 1,  # 1使用已选中优先 2 智能匹配
        retry_type = 0,
        video_count = 2,
        city_name =  "北京",
        show_title = "国际汽车文化节",
        show_address = "首钢会展中心",
        show_desc = None,  # user's description on the show, can be empty
        mult_source = [
            {
                "source_video_id": 2903,
                "type": 2949,
                "videos": [
                    {
                        "material_id": 211,
                        "material_path": "https://www.pexels.com/zh-cn/download/video/36033073.mp4",
                        "is_opening": 1
                    },
                    {
                        "material_id": 330,
                        "material_path": "https://www.pexels.com/zh-cn/download/video/8490544.mp4",
                        "is_opening": 0
                    }
                ],
                "templates":
                    {
                        "material_id": 153,
                        "material_path": "https://xiuxiu-pro.meitudata.com/posters/d708ecd87c7a32f4309144b2f57a4a97.jpeg"
                    },
                "bgms": [
                    {
                        "material_id": 405,
                        "material_path": "https://freepd.cn/api/music/576f726c642f4e6f72746875722e6d7033.mp3"
                    }
                ],
                "voices":
                    {
                        "material_id": 173,
                        "robot_show_name": "唐僧"
                    },
                "scripts":
                    {
                        "material_id": 1148,
                        "script_content": "我实在是太喜欢北京国际汽车文化节了！"
                    }
            }
        ],
        retry_source = [
            {
                "source_video_id": 290,
                "type": 2999,
                "videos": [
                    {
                        "material_id": 356,
                        "material_path": "https://www.pexels.com/zh-cn/download/video/13929678.mp4",
                        "is_opening": 1
                    },
                    {
                        "material_id": 329,
                        "material_path": "https://www.pexels.com/zh-cn/download/video/31319337.mp4",
                        "is_opening": 0
                    }
                ],
                "templates":
                    {
                        "material_id": 152,
                        "material_path": "https://xiuxiu-pro.meitudata.com/posters/3eb50419b73499bf738eef9a3d3b25c9.jpeg"
                    },
                "bgms": [
                    {
                        "material_id": 407,
                        "material_path": "https://freepd.cn/api/music/436f6d6564792f48656c6c6f21204d6120426162792e6d7033.mp3"
                    }
                ],
                "voices":
                    {
                        "material_id": 198,
                        "robot_show_name": "小辉"
                    },
                "scripts":
                    {
                        "material_id": 1198,
                        "script_content": "北京国际汽车文化节真是太棒了！"
                    }
            }
        ]
    )
    result1 = {
        "videos": [
            {
                "material_id": 126,
                "material_path": "https://www.pexels.com/zh-cn/download/video/35715727.mp4",
                "is_opening": 1
            },
            {
                "material_id": 110,
                "material_path": "https://www.pexels.com/zh-cn/download/video/33749020.mp4",
                "is_opening": 1
            },
            {
                "material_id": 104,
                "material_path": "https://www.pexels.com/zh-cn/download/video/33749020.mp4",
                "is_opening": 1
            },
            {
                "material_id": 126,
                "material_path": "https://www.pexels.com/zh-cn/download/video/35715727.mp4",
                "is_opening": 0
            },
            {
                "material_id": 110,
                "material_path": "https://www.pexels.com/zh-cn/download/video/33749020.mp4",
                "is_opening": 0
            },
            {
                "material_id": 104,
                "material_path": "https://www.pexels.com/zh-cn/download/video/33749020.mp4",
                "is_opening": 0
            },
            {
                "material_id": 106,
                "material_path": "https://www.pexels.com/zh-cn/download/video/35715727.mp4",
                "is_opening": 1
            },
            {
                "material_id": 112,
                "material_path": "https://www.pexels.com/zh-cn/download/video/33749020.mp4",
                "is_opening": 1
            },
            {
                "material_id": 116,
                "material_path": "https://www.pexels.com/zh-cn/download/video/33749020.mp4",
                "is_opening": 1
            }
        ],
        "templates": [
            {
                "material_id": 114,
                "material_path": "https://xiuxiu-pro.meitudata.com/posters/b5ff9c7bdcd0800d735bb9cf27cf82a6.jpg"
            }
        ],
        "bgms": [
            {
                "material_id": 111,
                "material_path": "https://freepd.cn/api/music/486f72726f722f4372656570792048616c6c6f772e6d7033.mp3"
            }
        ],
        "voices": [
            {
                "material_id": 1,
                "robot_show_name": "小美"
            }
        ],
        "scripts": [
            {
                "material_id": 1139,
                "script_content": "2026北京国际汽车文化节来了，6月12日-15日，就在首钢会展中心，百余款新车齐亮相，车模表演现场抽奖high翻天，快带上你的家人朋友来逛展吧！"
            },
            {
                "material_id": 1131,
                "script_content": "2026北京国际汽车文化节将在6月12日-15日登录首钢会展中心，大牌新车云集，车模表演现场抽奖氛围热烈，热爱汽车的你千万不要错过！"
            }
        ]
  }
    from functionals.utils import load_tts_data
    tts_data1 = load_tts_data(TTS_DATA_PATH)
    rcm = Recommend(
        request =  RecommendationRequest(
            task_id = 22,
            show_id = 33,
            org_id = 1,
            industry_id = 1,
            task = task1,
            result = result1
        ),
        tts_data=tts_data1
    )
    print(asyncio.run(rcm.execute()))
    # print(rcm._generate_script_sub(
    #     {
    #         "script_content":"2026北京全民购车节，视频要轻松欢快。活动时间是三月14日和15日，地点在北京农业展览馆。这是今年的第一个全民购车节突出今天最后七小时免费抢门票。车展上还有国家，市区县的购车补贴。车展上有进口合资国产新能源各种车型，还有车模表演，带礼品的的游戏。"
    #     }
    # )
    # )
