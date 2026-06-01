from contextlib import asynccontextmanager
from datetime import datetime
from fastapi import FastAPI, Request
from pymilvus import AsyncMilvusClient
from config.path_config import TTS_DATA_PATH, MILVUS_URL
from config.schema_config import APIResponse, ComplianceRequest, ComplianceResult, OptimizationRequest, \
    OptimizationResult, SelectMaterialRequest, SelectMaterialResult, RecommendationResult, RecommendationRequest
from functionals.check_compliance import CheckCompliance
from functionals.optimize_script import OptimizeScript
from functionals.logger import video_director_logger
from functionals.recommend import Recommend
from functionals.select_material import SelectMaterial
from functionals.utils import load_tts_data


# Create a lifespan to run while the app is active
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load config once at startup
    app.state.tts_data = load_tts_data(TTS_DATA_PATH)
    if not isinstance(app.state.tts_data, list):
        e_m = f"配音音色描述文件应为列表，实际为{type(app.state.tts_data).__name__}"
        video_director_logger.error(e_m)
        raise TypeError(e_m)

    # Validate Milvus connection
    try:
        test_client = AsyncMilvusClient(uri=MILVUS_URL, secure=False)
        await test_client.has_collection(collection_name="health_check_dummy")
        await test_client.close()
        video_director_logger.info("🕸️ Milvus向量数据库链接可用")
    except Exception as e:
        video_director_logger.warning(f"⚠️ Milvus 向量数据库链接报错: {e}")

    app.state.check_compliance = CheckCompliance()
    app.state.optimize_script = OptimizeScript()
    app.state.select_material = SelectMaterial()

    video_director_logger.info("✅ AI导演API服务成功启动")
    yield
    video_director_logger.info("👋 AI导演API服务关闭")

app = FastAPI(
    title="Video Director API",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health")
async def health_check():
    """Quick endpoint to verify the server is running before using Postman."""
    return {"status": "healthy", "service": "video-director", "timestamp": datetime.now().isoformat()}

@app.post("/check_compliance", response_model=APIResponse[ComplianceResult])
async def check_compliance(request: Request, payload: ComplianceRequest) -> APIResponse[ComplianceResult]:
    """Check whether the scripts meet the compliance requirements from Douyin and Tencent"""
    try:
        start_time = datetime.now()
        video_director_logger.info("⚖️ 合规检查任务开始")

        result = await request.app.state.check_compliance(payload.script)

        duration = (datetime.now() - start_time).total_seconds()
        video_director_logger.info(f"⚖️ 合规检查任务完成, 耗时{round(duration, 2)}秒")

        return APIResponse.ok(data=result)
    except Exception as e:
        e_m = f"❌ 合规检查失败: {e}"
        video_director_logger.error(e_m)
        return APIResponse[ComplianceResult].fail(e_m, error_code="COMPLIANCE_ERROR")

@app.post("/optimize_script", response_model=APIResponse[OptimizationResult])
async def optimize_script(request: Request, payload: OptimizationRequest) -> APIResponse[OptimizationResult]:
    """Optimize current script"""
    try:
        start_time = datetime.now()
        video_director_logger.info("🪄 文案优化任务开始")

        result = await request.app.state.optimize_script(payload.script)

        duration = (datetime.now() - start_time).total_seconds()
        video_director_logger.info(f"🪄 文案优化任务完成, 耗时{round(duration, 2)}秒")

        return APIResponse.ok(data=result)
    except Exception as e:
        e_m = f"❌ 文案优化失败: {e}"
        video_director_logger.error(e_m)
        return APIResponse[OptimizationResult].fail(e_m, error_code="OPTIMIZATION_ERROR")

@app.post("/select_material", response_model=APIResponse[SelectMaterialResult])
async def select_material(request: Request, payload: SelectMaterialRequest) -> APIResponse[SelectMaterialResult]:
    """Select matching footage, BGM, TTS voice, and generate scripts"""
    start_time = datetime.now()
    video_director_logger.info("📚 选材任务开始")

    # ------- Search materials from vector database -------
    try:
        search_results = await request.app.state.select_material.search(payload)

        duration = (datetime.now() - start_time).total_seconds()
        video_director_logger.info(f"📚 🔍 选材任务-搜索向量数据库完成, 已耗时{round(duration, 2)}秒")
    except Exception as e:
        e_m = f"❌ 向量数据库搜索失败: {e}"
        video_director_logger.error(e_m)
        return APIResponse[SelectMaterialResult].fail(e_m, error_code="MATERIAL_SEARCH_ERROR")

    # ------- Decide voice -------
    try:
        tts_data = request.app.state.tts_data
        decide_tts_results = await request.app.state.select_material.decide_tts(payload, tts_data)

        duration = (datetime.now() - start_time).total_seconds()
        video_director_logger.info(f"📚 🎙️ 选材任务-选择配音音色完成, 已耗时{round(duration, 2)}秒")
    except Exception as e:
        e_m = f"❌ 选择配音音色失败: {e}"
        video_director_logger.error(e_m)
        return APIResponse[SelectMaterialResult].fail(e_m, error_code="DECIDE_TTS_ERROR")

    # ------- Write script -------
    try:
        write_script_results = await request.app.state.select_material.write_script(payload)

        duration = (datetime.now() - start_time).total_seconds()
        video_director_logger.info(f"📚 ✍️ 选材任务-创作文案完成, 已耗时{round(duration, 2)}秒")
    except Exception as e:
        e_m = f"❌ 创作文案失败: {e}"
        video_director_logger.error(e_m)
        return APIResponse[SelectMaterialResult].fail(e_m, error_code="WRITE_SCRIPT_ERROR")

    duration = (datetime.now() - start_time).total_seconds()
    video_director_logger.info(f"📚 选材任务完成, 共耗时{round(duration, 2)}秒")

    return APIResponse.ok(data=SelectMaterialResult(
        video_ids=search_results.get("footage_opening",[]) + search_results.get("footage_regular",[]),
        template_ids=search_results.get("image",[]),
        bgm_ids=search_results.get("bgm",[]),
        voice_ids=decide_tts_results.material_ids,
        scripts=write_script_results.scripts
    ))

@app.post("/recommend", response_model=APIResponse[RecommendationResult])
async def recommend(request: Request, payload: RecommendationRequest) -> APIResponse[RecommendationResult]:
    try:
        start_time = datetime.now()
        video_director_logger.info("👍 推荐任务开始")

        tts_data = request.app.state.tts_data
        recommend_instance = Recommend(request=payload, tts_data=tts_data)
        result = await recommend_instance.execute()

        duration = (datetime.now() - start_time).total_seconds()
        video_director_logger.info(f"👍 推荐任务完成, 耗时{round(duration, 2)}秒")

        return APIResponse.ok(data=result)
    except Exception as e:
        e_m = f"❌ 推荐服务执行失败: {e}"
        video_director_logger.error(e_m, exc_info=True)
        return APIResponse[RecommendationResult].fail(e_m, error_code="RECOMMENDATION_ERROR")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("video_director:app", host="0.0.0.0", port=8014)

