import asyncio
from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from pymilvus import AsyncMilvusClient
from config.constant_config import TOP_K, MATERIAL_TYPE, MATERIAL_CN, TOP_K_FOOTAGE_REGULAR, TOP_K_FOOTAGE_OPENING, \
    N_SCRIPT
from config.path_config import MILVUS_URL
from config.schema_config import QueryResult, SelectMaterialRequest, DecideTTSResult, WriteScriptResult
from functionals.logger import video_director_logger
from functionals.utils import embed_query, user_id_to_collection_name
from models.llm_models import llm_regular

QUERY_SYSTEM_PROMPT = """
# === 你的角色 ===
你是一个查询语句的生成专家，专门为精准匹配生成用于向量数据库检索的查询语句。

# === 你的核心任务 === 
向量数据库目前含有各种视频/图片/背景音乐的描述信息的向量。这些描述信息的内容包括**该视频/图片/背景音乐体现的内容和情绪，以及适用于何种短视频的制作**。
现在用户需要制作一段展会的宣传短视频，并提供展会的行业和短视频的制作要求。
**请根据所提供的【展会行业】和【用户要求】，生成一段查询语句，可以在向量数据中找到最匹配的视频/图片/背景音乐**。

# === 输出要求 ===
- **必须输出仅含一个键的JSON**：
    - `query`: 生成的查询语句，不超过100字
- **不得输出其他格式**，或其他JSON键值，不得输出自然语言对话

# === 重要指示 ===
- 查询语句必须包括并完整体现展会行业和用户要求，**不得减少或更改**
- 查询语句会同时用于搜素视频/图片/背景音乐，所以内容**不用包括具体的媒体格式**

# === 输出示例 ===
用户输入: "【展会行业】：车展 \n【用户要求】：2026北京国际车展，今年全国最大的车展，有很多新的车型和科技都将亮相"
你的输出:
{{
  "query": "汽车、人群、科技、氛围热烈、充满科技感。适合大型车展、新车发布会、新汽车科技发布会的短视频制作"
}}

----------------------

用户输入: "【展会行业】：家博会 \n【用户要求】：绵阳第三届美好家博会有很多大牌家装品牌参展，目标定位有装修需求的家庭"
你的输出:
{{
  "query": "家具品牌、卫浴品牌、电器品牌、人群、热闹、全家参观。适合家装博览会、家电展销会、家具销售会会的短视频制作"
}}
"""

DECIDE_TTS_SYSTEM_PROMPT = """
# === 你的角色 ===
你是一个展会短视频配音音色策划专家，擅长根据展会行业属性、视频风格与目标受众，精准匹配最合适的TTS音色。

# === 你的核心任务 === 
用户需要制作一段展会的宣传短视频，并提供展会的行业和短视频的制作要求。
**请根据所提供的【展会行业】和【用户要求】，从下方提供的音色库中，精心挑选出最匹配的{top_k}种音色**。

# === 输出要求 ===
- **必须输出仅含两个键的JSON**：
    - `material_ids`: 长度为{top_k}的整数列表，对应所选音色的ID
    - `material_descs`: 长度为{top_k}的字符串列表，必须与下方音色描述**逐字完全一致**，顺序与material_ids一一对应
- **不得输出其他格式**，或其他JSON键值，不得输出自然语言对话
- 严禁捏造、修改、拼接或使用音色库之外的ID与描述

# === 可选音色 ===
{formatted_tts_data}

# === 输出示例 ===
{{
  "material_ids": [138, 139, 149, ......],
  "material_descs":["语调平稳、咬字柔和、自带治愈安抚力的女声音色",
                    "声线甜美有活力的妹妹，活泼开朗，笑容明媚。", 
                    "声线阳光温暖、语气亲切，活力满满的少年音",
                    ......]
}}
"""

WRITE_SCRIPT_SYSTEM_PROMPT = """
# === 你的角色 ===
你是一个展会短视频的文案专家，擅长根据展会行业属性、视频风格与目标受众，创作最合适的口播文案。

# === 你的核心任务 === 
用户需要制作一段展会的宣传短视频，并提供展会的行业和短视频的制作要求。
**请根据所提供的【展会行业】和【用户要求】，创作{n_script}套最能宣传该展会，让视频充满商业吸引力的口播文案。**。
每套文案都必须包括以下信息：{city_name}，{show_title}，{show_address}，{show_time}，
但是**直接把这些字段名加上大括号即可，不用写具体的展会信息，无论用户是否提供这些信息。**

# === 输出要求 ===
- **必须输出仅含一个键的JSON**：
    - `scripts`: 长度为{n_script}的字符串列表，对应创作的文案
- **每套文案不得超过100字。**
- **不得输出其他格式**，或其他JSON键值，不得输出自然语言对话
- 不得使用“第一、最佳、顶尖、国家级、世界级”等面向商品或服务的绝对化表述
- 不得使用错别字、拼音、字母、emoji等方式代替原有文本
- 文案中有涉及“先到先得”“限时限量”的，需在标注具体的活动时间和商品数量
- 可以适当发挥创造力，提升文案的吸引力，但**不得更改活动信息，不得凭空创造信息，不得聊天**。

# === 输出示例 ===
{{
  "scripts": [
        "2026{{{city_name}}}{{{show_title}}}，{{{show_time}}}，锁定{{{show_address}}}。免费门票抢购仅剩最后7小时！买车直接省到底。
        各大品牌全阵容亮相，名额有限，赶紧抢票，周末直接冲！",
        "周末去哪儿？2026{{{city_name}}}{{{show_title}}}来了！{{{show_time}}}，{{{show_address}}}等你来逛。免费领票只剩最后7小时啦！
        除了看车，还有车模表演和趣味小游戏，玩着把礼品带回家！",
        "关注购车的朋友请注意，2026{{{city_name}}}{{{show_title}}}正式定档{{{show_time}}}，地点设于{{{show_address}}}。
        优惠力度空前，免费观展门票进入最后7小时倒计时！",
        ......
  ]
}}
"""

# input = {
#     "city_name": "展会城市",
#     "show_title": "展会名称",
#     "show_address": "展会地点",
#     "show_time": "展会日期",
#     "user_input": "2026年最大的车展将在5月1日.....",
#     "industry_name": "车展",
#     "industry_id":2,
#     "org_id":23
# }
# output = {
#     "video_ids": [1,2,3,4,5,6,7],
#     "template_id": [1,2,3,4,5,6,7],
#     "bgm_ids": [1,2,3,4,5,6,7],
#     "voice_ids": [1,2,3,4,5,6,7],
#     "scripts": [
#         {"script_content": "欢迎来到北京车展"},
#         {"script_content": "欢迎来到北京车展"},
#         {"script_content": "欢迎来到北京车展"},
#         {"script_content": "欢迎来到北京车展"},
#         {"script_content": "欢迎来到北京车展"},
#         {"script_content": "欢迎来到北京车展"}
#     ]
# }

class SelectMaterial:
    def __init__(self):
        self.query_chain = self._generate_query_chain()
        self.decide_tts_chain = self._generate_decide_tts_chain()
        self.write_script_chain = self._generate_write_script_chain()
        self.milvus_client = AsyncMilvusClient(uri = MILVUS_URL, secure=False)

    @staticmethod
    def _generate_query_chain():
        query_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", QUERY_SYSTEM_PROMPT),
                ("human", "【展会行业】：{industry_name}\n【用户要求】：{user_input}")
            ]
        )
        llm_with_structured_output_query = llm_regular.with_structured_output(QueryResult)
        return query_prompt | llm_with_structured_output_query

    @staticmethod
    def _generate_decide_tts_chain():
        # decide_tts
        decide_tts_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", DECIDE_TTS_SYSTEM_PROMPT),
                ("human", "【展会行业】：{industry_name}\n【用户要求】：{user_input}")
            ]
        )
        llm_with_structured_output_decide_tts = llm_regular.with_structured_output(DecideTTSResult)
        return decide_tts_prompt | llm_with_structured_output_decide_tts

    @staticmethod
    def _generate_write_script_chain():
        write_script_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", WRITE_SCRIPT_SYSTEM_PROMPT),
                ("human", "【展会行业】：{industry_name}\n【用户要求】：{user_input}")
            ]
        )
        llm_with_structured_output_write_script = llm_regular.with_structured_output(WriteScriptResult)
        return write_script_prompt | llm_with_structured_output_write_script

    async def search(self, request: SelectMaterialRequest)->dict:
        collection_name = user_id_to_collection_name(request.org_id)
        search_results = {}
        # -------- Generate query ----------
        try:
            query_result = await self.query_chain.ainvoke({'industry_name': request.industry_name, 'user_input':request.user_input})
            query = query_result.query
            if not query or not isinstance(query, str):
                e_m = f"❌ 查询语句格式有误"
                video_director_logger.error(e_m)
                raise ValueError(e_m)
        except Exception as e:
            e_m = f"❌ 查询语句生成有误: {e}"
            video_director_logger.error(e_m)
            raise Exception(e_m) from e

        # -------- Search from vector database ----------
        # Generate query embedding
        query_emb:list = await embed_query(query)
        filter_expr = f"industry_id in [0, {int(request.industry_id)}] and status == 1" # make filters

        # Async concurrent loop
        search_tasks = []
        for material_type in MATERIAL_TYPE:
            search_tasks.append(self._search_one_type(material_type, collection_name, query_emb, filter_expr))
        async_results = await asyncio.gather(*search_tasks, return_exceptions=True)

        for material_type, result in zip(MATERIAL_TYPE, async_results):
            if isinstance(result, Exception):
                video_director_logger.error(f"❌ '{query}'在向量数据库: {collection_name}查询{MATERIAL_CN[material_type]}失败: {result}")
                search_results[material_type] = []
            elif not result or not result[0]:
                video_director_logger.warning(f"⚠️ '{query}'在向量数据库: {collection_name}查询{MATERIAL_CN[material_type]}无结果")
                search_results[material_type] = []
            else:
                search_results[material_type] = [
                    int(m.get("material_id"))
                    for m in result[0]
                    if m.get("material_id") is not None
                ]
        return search_results

    async def _search_one_type(
            self, 
            material_type:Literal[*MATERIAL_TYPE], 
            collection_name: str, 
            query_emb: list, 
            filter_expr: str):
        # Search the top n
        top_n = TOP_K_FOOTAGE_REGULAR if material_type == "footage_regular" else \
            TOP_K_FOOTAGE_OPENING if material_type == "footage_opening" else TOP_K
        return await self.milvus_client.search(
            collection_name=collection_name,
            partition_names=[material_type],
            data=[query_emb],
            filter=filter_expr,
            limit=top_n,
            output_fields=["material_id", "material_path", "industry_id", "status", "desc_json", "version"],
            timeout=8.0
        )
    
    async def decide_tts(self, request: SelectMaterialRequest, tts_data: list)->DecideTTSResult:
        formatted_tts_data = ""
        if isinstance(tts_data, list) and tts_data:
            for t_d in tts_data:
                formatted_tts_data += f"- ID: {t_d.get('material_id')}, 音色描述: {t_d.get('material_desc')}\n"

        try:
            decide_tts_results = await self.decide_tts_chain.ainvoke({
                'top_k': TOP_K,
                'formatted_tts_data':formatted_tts_data,
                'industry_name': request.industry_name,
                'user_input':request.user_input
            })

            # slightly validate the ids
            decide_tts_results.material_ids = [int(_id) for _id in decide_tts_results.material_ids]
            return decide_tts_results

        except Exception as e:
            e_m = f"❌ 选择配音音色有误: {e}"
            video_director_logger.error(e_m)
            raise Exception(e_m) from e

    async def write_script(self, request: SelectMaterialRequest)->WriteScriptResult:
        try:
            write_script_results = await self.write_script_chain.ainvoke({
                'n_script': N_SCRIPT,
                'city_name': request.city_name,
                'show_title': request.show_title,
                'show_address': request.show_address,
                'show_time': request.show_time,
                'industry_name': request.industry_name,
                'user_input': request.user_input,
            })

            # slightly validate the ids
            write_script_results.scripts = [str(script) for script in write_script_results.scripts]
            return write_script_results

        except Exception as e:
            e_m = f"❌ 生成文案有误: {e}"
            video_director_logger.error(e_m)
            raise Exception(e_m) from e


if __name__ == '__main__':
    select_material = SelectMaterial()
    # print(asyncio.run(select_material.search(
    #     SelectMaterialRequest(
    #         user_input = "2026年5月1日-3日，石家庄车展在正定会展中心开幕。这是中国历史上最大的车展，优惠空前。喜欢汽车的朋友一定要来看一看。",
    #         industry_name = "车展",
    #         industry_id=1,
    #         org_id=1
    #     )
    #
    # )))
    # with open(TTS_DATA_PATH, 'r', encoding='utf-8') as f:
    #     tts_data = json.load(f)
    # print(asyncio.run(select_material.decide_tts(
    #     SelectMaterialRequest(
    #         user_input="2026年9月1日-3日，石家庄车展在正定会展中心开幕。这是中国历史上最大的车展，优惠空前。喜欢汽车的朋友一定要来看一看。",
    #         industry_name="车展",
    #         industry_id=1,
    #         org_id=1
    #     ),
    #     tts_data
    # )))
    print(asyncio.run(select_material.write_script(
        SelectMaterialRequest(
            user_input="2027年1月1日-5日，沧州家博会在沧州会展中心开幕。各种家电家具品牌一应俱全，欢迎有宝宝的家庭来看看。",
            industry_name="车展",
            industry_id=1,
            org_id=1
        )
    )))