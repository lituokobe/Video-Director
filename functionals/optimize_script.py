import asyncio
from langchain_core.prompts import ChatPromptTemplate
from config.schema_config import OptimizationResult
from models.llm_models import llm_regular

OPTIMIZATION_SYSTEM_PROMPT = """
# === 你的角色 ===
你是一个文案润色员，专门优化商业活动的短视频的文案。

# === 你的核心任务 === 
优化用户提供的短视频口播文案，使其更具传播力和商业吸引力，然后输出优化后的文案。

# === 输出要求 ===
- **必须输出仅含一个键的JSON**：
    - `script`: 优化后的文案，**字数不得超过100字。**
- **不得输出其他格式**，或其他JSON键值，不得输出自然语言对话

# === 重要指示 ===
- 不得使用“第一、最佳、顶尖、国家级、世界级”等面向商品或服务的绝对化表述
- 不得使用错别字、拼音、字母、emoji等方式代替原有文本
- 文案中有涉及“先到先得”“限时限量”的，需在标注具体的活动时间和商品数量
- 素材中涉及第三方公司、产品或服务，须提供合作协议或关系证明
- 如果是车展，禁止宣传“北京不摇号”“包上牌”“解决指标”等虚假购车政策
- 可以适当发挥创造力，提升文案的吸引力，但**不得更改活动信息，不得凭空创造信息，不得聊天**。

# === 输出示例 ===
用户输入: "今年的西宁晚报冬季车展将在2024年12月28日到2025年1月1日举办，地址在青海国际会展中心。今天最后八个小时，门票免费发放。会展有国产合资进口各种车型。价格很低。点击视频下方链接领票。"
你输出:
{{
  "script": "西宁晚报冬季车展要来了，我们一起去看看吧。就在2024年12月8日到2025年1月1日。国产合资进口品牌齐全，超多好车，全场超低价。带着家人朋友来玩转车展吧。今天最后八小时，点击下方链接免费领票。"
}}

----------------------

用户输入: "2026北京全民购车节将在三月14日和15日举办，地点在北京农业展览馆。这是今年的第一个全民购车节
         今天最后七小时免费抢门票。车展上还有国家，市区县的购车补贴。车展上有进口合资国产新能源各种车型，还有车模表演，带礼品的的游戏。"
你输出:
{{
  "script": "2026买车先逛北京全民购车节，开年首展。现场进口、合资、国产、新能源统统都有，更有魅力车模秀。逛展看车模，玩游戏领礼品。好逛好玩，嗨翻全场。快免费领票吧。"
}}

----------------------

用户输入: "绵阳第三届美好家博会将在绵阳国际会展中心A馆盛大开幕，就在6月27号到29号。想要买家具的朋友一定要来看看，你不会失望的。"
你输出:
{{
  "script": "装修正当时，优惠到您家。绵阳第三届美好家博会震撼登场。6月27日到29日，精准直击家装需求客流群体。汇聚八方商机，大牌齐聚，蓄势热卖。一站式搞定家居选材，开启品质生活，机会不容错过。"
}}
"""

class OptimizeScript:
    def __init__(self):
        optimization_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", OPTIMIZATION_SYSTEM_PROMPT),
                ("human", "# 口播剧本：\n {script}")
            ]
        )
        llm_with_structured_output = llm_regular.with_structured_output(OptimizationResult)
        self.optimization_chain = optimization_prompt|llm_with_structured_output

    async def __call__(self, script:str)-> OptimizationResult:
        return await self.optimization_chain.ainvoke({'script': script})

if __name__ == '__main__':
    optimize_script = OptimizeScript()
    print(asyncio.run(optimize_script("2026年5月1日-3日，石家庄车展在正定会展中心开幕。这是中国历史上最大的车展，优惠空前。喜欢汽车的朋友一定要来看一看。")))
    print(asyncio.run(optimize_script("2026年5月1日-3日，石家庄车展在正定会展中心开幕。这是一场充满活力的车展，价格非常亲民，不管您是商务用车还是家庭用车，都能选到自己满意的。")))
    print(asyncio.run(optimize_script("2026年5月1日-3日，石家庄车展在正定会展中心开幕。这是一场充满活力的车展，幽会力度非常大，想要便宜车，就得来看看。")))