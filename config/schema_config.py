# API response schema
from typing import TypeVar, Generic, Literal
from pydantic import BaseModel, Field

T = TypeVar("T")
class APIResponse(BaseModel, Generic[T]):
    """Unified API response wrapper"""
    success: bool = Field(..., description="Whether the request succeeded")
    data: T|None = Field(default=None, description="Response payload on success")
    error: str|None = Field(default=None, description="Error message on failure")
    error_code: str|None = Field(default=None, description="Machine-readable error code")

    @classmethod
    def ok(cls, data: T) -> "APIResponse[T]":
        return cls(success=True, data=data)

    @classmethod
    def fail(cls, error: str, error_code: str = "UNKNOWN") -> "APIResponse[T]":
        return cls(success=False, error=error, error_code=error_code)

class ComplianceRequest(BaseModel):
    script: str = Field(min_length=1)

class ComplianceResult(BaseModel):
    compliant: bool = Field(default=False)
    reason: str = Field(default='')

class OptimizationRequest(BaseModel):
    script: str = Field(min_length=1)

class OptimizationResult(BaseModel):
    script: str = Field(default='')

class SelectMaterialRequest(BaseModel):
    city_name: str = Field(default="展会城市")
    show_title: str = Field(default="展会名称")
    show_address: str = Field(default="展会地点")
    show_time: str = Field(default="展会日期")
    user_input: str = Field(min_length=1)
    industry_name: str = Field(min_length=1)
    industry_id: int = Field(default=0)
    org_id: int

class SelectMaterialResult(BaseModel):
    video_ids: list[int]
    template_ids: list[int]
    bgm_ids: list[int]
    voice_ids: list[int]
    scripts: list[str]

class QueryResult(BaseModel):
    query: str = Field(default='')

class DecideTTSResult(BaseModel):
    material_ids: list = Field(default=[])
    material_descs: list = Field(default=[])

class WriteScriptResult(BaseModel):
    scripts: list = Field(default=[])

class TaskRequest(BaseModel):
    task_desc: str = str|None # user's description on task, can be empty
    task_type: Literal[1,2,3,4] #1 普通成片  2 普通成片得 重新生成  3 爆款裂变   4 爆款裂变得重新生成
    video_type: Literal[1, 2] # 1 口播从开场白视频开始 2 口播在开场白视频结束后开始，视频时长限制要稍微尝一下
    ai_director: str|None # user requirements, can be empty
    template_strategy: Literal[1,2] #1使用已选中优先 2 智能匹配
    retry_type: Literal[0,1,2]
    video_count: int #10,
    city_name: str = Field(min_length=1) # "北京"
    show_title: str = Field(min_length=1) #"国际汽车文化节"
    show_address: str = Field(min_length=1) # "北京国际展览中心"
    show_time: str = Field(min_length=1) # "2026-05-01到2026-05-03"
    show_desc: str|None # user's description on the show, can be empty
    mult_source: list[dict|None]
    retry_source: list[dict|None]

class RecommendationRequest(BaseModel):
    task_id:int
    show_id:int
    org_id:int
    industry_id: int = Field(default=0)
    task: TaskRequest
    result: dict

class RecommendationResult(BaseModel):
    task_id: int
    show_id: int
    org_id: int
    industry_id: int = Field(default=0)
    results: list[dict]

class VideoPlanNoScript(BaseModel):
    footage_opening: int = Field(description="开场白视频素材 material_id")
    footage_regular: list[int] = Field(description="普通视频素材 material_id 列表")
    image: int = Field(description="文字贴图 material_id")
    bgm: int|None = Field(description="背景音乐 material_id")

class VideoPlansResponseNoScript(BaseModel):
    plans: list[VideoPlanNoScript] = Field(description="生成的短视频版本列表")

class VideoPlanScript(BaseModel):
    footage_opening: int = Field(description="开场白视频素材 material_id")
    footage_regular: list[int] = Field(description="普通视频素材 material_id 列表")
    image: int = Field(description="文字贴图 material_id")
    bgm: int|None = Field(description="背景音乐 material_id")
    tts: int = Field(description="语音音色 material_id")

class VideoPlansResponseScript(BaseModel):
    plans: list[VideoPlanScript] = Field(description="生成的短视频版本列表")