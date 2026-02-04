
import os
import sys
print(f"DEBUG SYSTARGV: {sys.argv}")
import argparse
import logging
import time
import numpy as np
from typing import Optional, Generator
from concurrent.futures import ThreadPoolExecutor
import threading
import torch
import torchaudio
from datetime import datetime
from fastapi.responses import FileResponse

# 方法1.1：使用时间戳+随机数
import uuid

def generate_filename_with_timestamp(prefix="", suffix="", extension="txt"):
    """生成基于时间戳的唯一文件名"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]  # 取UUID前8位
    
    # 确保 temp_output 目录存在
    output_dir = os.path.join(SCRIPT_DIR, '../temp_output')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    filename = os.path.join(output_dir, f"{prefix}{timestamp}_{unique_id}{suffix}.{extension}")
    return filename

# 设置路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, '..'))
sys.path.insert(0, os.path.join(SCRIPT_DIR, '../third_party', 'Matcha-TTS'))

from fastapi import FastAPI, Form, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# 多音色配置 (用户可修改此部分)
# ============================================================================
# 格式: {"id": "音色ID", "file": "文件名", "prompt_text": "音频中说的话"}
# 文件放在 deploy/cosyvoice/asset/ 目录下
# 要求音频清晰（5~15秒为佳，不要过长），保存为wav格式，采集率16kHz,单声道，内容任意，但是和prompt_text里的内容完全一致，必须一字不差！
# CosyVoice3 的 prompt_text 必须以 "You are a helpful assistant.<|endofprompt|>" 开头
# ============================================================================
VOICE_CONFIGS = [
    {
        "id": "default",  # 默认音色
        "file": "zero_shot_prompt.wav",  # asset/zero_shot_prompt.wav
        "prompt_text": "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。",
        "type":"female"
    },
    # 添加更多音色示例 (取消注释并修改):
    {
        "id": "longyingcheng",
        "file": "longyingcheng_man.wav",
        "prompt_text": "You are a helpful assistant.<|endofprompt|>真不好意思，从小至今，他还从来没有被哪一位异性朋友亲吻过呢。",
        "type":"male"
    },
    {
        "id": "longyingwan",
        "file": "longyingwan_woman.wav", 
        "prompt_text": "You are a helpful assistant.<|endofprompt|>我们将为全球城市的可持续发展贡献力量。",
        "type":"female"
    },
    {
        "id": "longyingmu",
        "file": "longyingmu_woman.wav",
        "prompt_text": "You are a helpful assistant.<|endofprompt|>您好，我是智能电话助手，很高兴为您服务。请问您需要咨询业务预约办理还是查询信息？",
        "type":"female"
    
    },
    {
        "id": "nanami",
        "file": "nanami.wav",
        "prompt_text": "You are a helpful assistant.<|endofprompt|>三百多万也不便宜，哎呀你要想上海的房价，你想上海平均都是多少。",
        "type":"female"
    
    },
    {
        "id": "female_sunny",
        "file": "female_sunny.wav",
        "type":"female",
        "prompt_text": "You are a helpful assistant.<|endofprompt|>快来,帮我看看今天下午茶喝什么?"
    },
    {
        "id": "female_hi",
        "file": "female_hi.wav",
        "type":"female",
        "prompt_text": "You are a helpful assistant.<|endofprompt|>日光有些刺眼,这顶帽子就借给你吧!放心,不会有鸽子飞出来的。"
    },
    {
        "id": "female_lazy",
        "file": "female_lazy.wav",
        "type":"female",
        "prompt_text": "You are a helpful assistant.<|endofprompt|>睡不着的话,就一起玩个通宵!"
    },
    {
        "id": "female_middle",
        "file": "female_middle.wav",
        "type":"female",
        "prompt_text": "You are a helpful assistant.<|endofprompt|>吃过午饭了吗?没有的话,要不要和我一起?"
    },
    {
        "id": "female_maid",
        "file": "female_maid.wav",
        "type":"female",
        "prompt_text": "You are a helpful assistant.<|endofprompt|>早安,魔王大人,请问您需要茶还是咖啡......"
    },
    {
        "id": "female_loli",
        "file": "female_loli.wav",
        "type":"female",
        "prompt_text": "You are a helpful assistant.<|endofprompt|>今天的午市生意特别好,我也要加油咯!"
    },
    {
        "id": "female_reader",
        "file": "female_reader.wav",
        "type":"female",
        "prompt_text": "You are a helpful assistant.<|endofprompt|>宝贝们，这几天我特别想告诉你一个天大的秘密，关系到你下半年的财运。记住2月31号这天，一定要来找我，我悄悄告诉你。现在赶紧点赞收藏，让我知道你在，别让好运从身边溜走啊！"
    },
    {
        "id": "male_low",
        "file": "male_low.wav",
        "type":"male",
        "prompt_text": "You are a helpful assistant.<|endofprompt|>人世间有太多无奈，如想去周游世界、但又没有钱。言到此处，我想起《金瓶梅》中的那句话，凡事都要听听豆包的，唯有豆包，是认真听我们讲话的。"
    },
    {
        "id": "midmale",
        "file": "midmale.wav",
        "prompt_text": "You are a helpful assistant.<|endofprompt|>至于她的生活来源，是在网络上写一些小短文，赚些钱。",
        "type":"male"
    
    }
]
# ============================================================================

# 全局变量
cosyvoice = None
inference_lock = threading.Lock()

# 多音色缓存: {voice_id: {"file": path, "prompt_text": text}}
voice_cache = {}
default_voice_id = "default"  # 默认使用的音色ID

# 输出采样率 (可通过 --output_sample_rate 配置)
# 默认 16000 以兼容小智平台
output_sample_rate = 24000

# [注意] CosyVoice API 不支持传入 Tensor，必须传路径或文件对象，因此移除 resampler_cache 优化


app = FastAPI(title="CosyVoice TTS Server", version="1.0.0")

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

def generate_instruct2(
    text: str,
    voice_id: str,
    motionctrl: str = None,
):
    global cosyvoice, voice_cache
    logger.info(f"参数: voice_id{voice_id} {text} {motionctrl}")
    # 确定使用哪个音色
    wavfile =  generate_filename_with_timestamp(prefix=voice_id, extension="wav")
    if voice_id not in voice_cache:
        raise ValueError(f"音色 '{voice_id}' 不存在")
    spk_id = voice_id
    voice_info = voice_cache[voice_id]
    actual_prompt_text = motionctrl
    actual_prompt_wav = voice_info["file"]
    for i, j in enumerate(cosyvoice.inference_instruct2(text,actual_prompt_text,actual_prompt_wav, stream=False)):
       torchaudio.save(wavfile.format(i), j['tts_speech'], cosyvoice.sample_rate)
    return wavfile


def generate_zs_sync(
    text: str,
    voice_id: str,
):
    global cosyvoice, voice_cache
    logger.info(f"参数: voice_id{voice_id} {text}")
    # 确定使用哪个音色
    wavfile =  generate_filename_with_timestamp(prefix=voice_id, extension="wav")
    if voice_id not in voice_cache:
        raise ValueError(f"音色 '{voice_id}' 不存在")
    voice_info = voice_cache[voice_id]
    actual_prompt_text = voice_info["prompt_text"]
    actual_prompt_wav = voice_info["file"]
    for i, j in enumerate(cosyvoice.inference_zero_shot(text,actual_prompt_text,actual_prompt_wav, stream=False)):
       torchaudio.save(wavfile.format(i), j['tts_speech'], cosyvoice.sample_rate)
    return wavfile

def generate_cross_lingual_sync(
    text: str,
    voice_id: str,
):
    global cosyvoice, voice_cache
    logger.info(f"参数: voice_id{voice_id} {text}")
    # 确定使用哪个音色
    wavfile =  generate_filename_with_timestamp(prefix=voice_id, extension="wav")
    if voice_id not in voice_cache:
        raise ValueError(f"音色 '{voice_id}' 不存在")
    voice_info = voice_cache[voice_id]
    actual_prompt_text = voice_info["prompt_text"]
    actual_prompt_wav = voice_info["file"]
    for i, j in enumerate(cosyvoice.inference_cross_lingual(f"You are a helpful assistant.<|endofprompt|>{text}",actual_prompt_wav, stream=False)):
       torchaudio.save(wavfile.format(i), j['tts_speech'], cosyvoice.sample_rate)
    return wavfile




def generate_audio_stream(
    text: str,
    voice_id: str = None,
    prompt_text: str = None,
    prompt_wav = None,
    stream: bool = True
) -> Generator[bytes, None, None]:
    """
    生成流式音频数据
    
    Args:
        text: 要合成的文本
        voice_id: 音色ID (使用预加载的缓存音色，零延迟)
        prompt_text: 自定义音色的提示文本 (与 prompt_wav 配合使用)
        prompt_wav: 自定义音色的参考音频 (与 prompt_text 配合使用)
        stream: 是否流式输出
    """
    global cosyvoice, voice_cache
    logger.info(f"⚡ 参数: voice_id{voice_id}(零I/O/计算)")
    with inference_lock:
        try:
            # 确定使用哪个音色
            spk_id = ""
            actual_prompt_text = prompt_text
            actual_prompt_wav = prompt_wav
            
            # 优先使用 voice_id (预缓存音色)
            if voice_id and voice_id in voice_cache:
                spk_id = voice_id
                voice_info = voice_cache[voice_id]
                actual_prompt_text = voice_info["prompt_text"]
                actual_prompt_wav = voice_info["file"]
                logger.debug(f"⚡ 使用预缓存音色: {voice_id} (零I/O/计算)")
            elif voice_id and voice_id not in voice_cache:
                logger.warning(f"音色 '{voice_id}' 不存在，使用默认音色")
                if default_voice_id in voice_cache:
                    spk_id = default_voice_id
                    voice_info = voice_cache[default_voice_id]
                    actual_prompt_text = voice_info["prompt_text"]
                    actual_prompt_wav = voice_info["file"]
            elif prompt_text and prompt_wav:
                # 使用自定义音色 (无缓存，需实时计算)
                logger.debug("使用自定义音色 (实时计算特征)")
            else:
                # 使用默认音色
                if default_voice_id in voice_cache:
                    spk_id = default_voice_id
                    voice_info = voice_cache[default_voice_id]
                    actual_prompt_text = voice_info["prompt_text"]
                    actual_prompt_wav = voice_info["file"]
                    logger.debug(f"⚡ 使用默认音色: {default_voice_id}")
                
            for result in cosyvoice.inference_zero_shot(
                text, 
                actual_prompt_text, 
                actual_prompt_wav,
                stream=stream,
                zero_shot_spk_id=spk_id
            ):
                audio_tensor = result['tts_speech']
                
                # [GPU 重采样] 如果输出采样率与模型原生不同，进行重采样
                if output_sample_rate != cosyvoice.sample_rate:
                    audio_tensor = torchaudio.functional.resample(
                        audio_tensor, 
                        orig_freq=cosyvoice.sample_rate, 
                        new_freq=output_sample_rate
                    )
                
                # [GPU 版本] PCM 16bit 转换
                # 1. GPU 进行乘法 (* 32768)
                # 2. GPU 进行类型转换 (float -> int16)
                # 3. 传输 int16 (2 bytes) 到 CPU
                yield (audio_tensor * 32768).to(torch.int16).cpu().numpy().tobytes()
        except Exception as e:
            logger.error(f"TTS 生成失败: {e}")
            raise
        
@app.get("/tts/voices")
async def voices():
    return JSONResponse(VOICE_CONFIGS)


@app.get("/health")
async def health_check():
    mps_is_built = bool(getattr(getattr(torch, "backends", None), "mps", None) and torch.backends.mps.is_built())
    mps_is_available = bool(getattr(getattr(torch, "backends", None), "mps", None) and torch.backends.mps.is_available())
    mps_memory = {}
    if mps_is_available and hasattr(torch, "mps"):
        try:
            allocated = getattr(torch.mps, "current_allocated_memory", None)
            driver_allocated = getattr(torch.mps, "driver_allocated_memory", None)
            if callable(allocated):
                mps_memory["allocated_bytes"] = int(allocated())
            if callable(driver_allocated):
                mps_memory["driver_allocated_bytes"] = int(driver_allocated())
        except Exception:
            mps_memory = {}

    cuda_info = {"is_available": bool(torch.cuda.is_available())}
    if torch.cuda.is_available():
        cuda_info.update(
            {
                "device_count": int(torch.cuda.device_count()),
                "device_name": torch.cuda.get_device_name(0),
                "memory_allocated_bytes": int(torch.cuda.memory_allocated(0)),
                "memory_reserved_bytes": int(torch.cuda.memory_reserved(0)),
            }
        )

    model_device = None
    if cosyvoice is not None:
        if hasattr(cosyvoice, "model") and hasattr(cosyvoice.model, "device"):
            model_device = str(cosyvoice.model.device)
        else:
            try:
                model_device = str(next(cosyvoice.parameters()).device)
            except Exception:
                model_device = None

    return JSONResponse(
        {
            "status": "ok",
            "platform": sys.platform,
            "python_version": sys.version.split(" ")[0],
            "torch_version": torch.__version__,
            "model": "Fun-CosyVoice3-0.5B-2512",
            "model_sample_rate": cosyvoice.sample_rate if cosyvoice else None,
            "output_sample_rate": output_sample_rate,
            "model_device": model_device,
            "available_voices": list(voice_cache.keys()),
            "default_voice": default_voice_id,
            "accelerators": {
                "cuda": cuda_info,
                "mps": {
                    "is_built": mps_is_built,
                    "is_available": mps_is_available,
                    "memory": mps_memory,
                },
            },
        }
    )


@app.post("/tts/stream")
async def tts_stream(
    text: str = Form(..., description="要合成的文本"),
    voice_id: Optional[str] = Form(default=None, description="音色ID (使用预加载的音色，零延迟)"),
    prompt_text: Optional[str] = Form(default=None, description="自定义音色的提示文本"),
    prompt_wav: Optional[UploadFile] = File(default=None, description="自定义音色的参考音频")
):
    """
    流式 TTS 接口
    
    参数优先级:
    1. voice_id: 使用预加载的音色 (推荐，零延迟)
    2. prompt_text + prompt_wav: 自定义音色 (需实时计算特征)
    3. 都不传: 使用默认音色
    
    返回: 流式 PCM 音频数据 (采样率由 --output_sample_rate 控制, 16bit, Mono)
    """
    if not text or len(text.strip()) == 0:
        raise HTTPException(status_code=400, detail="文本不能为空")
    
    # 处理自定义音色的参考音频
    prompt_wav_data = None
    if prompt_wav is not None:
        prompt_wav_data = prompt_wav.file
    
    # 记录日志
    if voice_id:
        logger.info(f"TTS 请求: text='{text[:50]}...', voice_id='{voice_id}'")
    elif prompt_text:
        logger.info(f"TTS 请求: text='{text[:50]}...', prompt_text='{prompt_text[:30]}...' (自定义音色)")
    else:
        logger.info(f"TTS 请求: text='{text[:50]}...', voice_id='default'")
    
    start_time = time.time()
    
    def stream_generator():
        first_chunk = True
        total_bytes = 0
        for chunk in generate_audio_stream(
            text, 
            voice_id=voice_id,
            prompt_text=prompt_text, 
            prompt_wav=prompt_wav_data, 
            stream=True
        ):
            if first_chunk:
                logger.info(f"⚡ 首帧延迟: {(time.time() - start_time) * 1000:.0f}ms")
                first_chunk = False
            total_bytes += len(chunk)
            yield chunk
        logger.info(f"✅ TTS 完成: 总耗时 {(time.time() - start_time) * 1000:.0f}ms, 数据量 {total_bytes / 1024:.1f}KB")
    
    return StreamingResponse(
        stream_generator(),
        media_type="application/octet-stream",
        headers={
            "X-Sample-Rate": str(output_sample_rate),
            "X-Channels": "1",
            "X-Bits": "16"
        }
    )

@app.post("/tts/instruct2")
async def instruct2(
    text: str = Form(..., description="要合成的文本"),
    voice_id: Optional[str] = Form(default=None, description="音色ID (使用预加载的音色，零延迟)"),
    motionctrl: str = Form(..., description="情感控制"),
): 
    if not voice_id:
        raise HTTPException(status_code=400, detail="voice_id 不能为空")
    if voice_id not in voice_cache:
        raise HTTPException(status_code=404, detail=f"音色 '{voice_id}' 不存在")
    fns = generate_instruct2(text,voice_id,motionctrl)
    
    return FileResponse(
        path=fns,
        filename=fns,  # 客户端下载时显示的文件名
        media_type="audio/wav"  # 可选的 MIME 类型
    )

@app.post("/tts/zs_sync")
async def instruct2(
    text: str = Form(..., description="要合成的文本"),
    voice_id: Optional[str] = Form(default=None, description="音色ID (使用预加载的音色，零延迟)"),
): 
    if not voice_id:
        raise HTTPException(status_code=400, detail="voice_id 不能为空")
    if voice_id not in voice_cache:
        raise HTTPException(status_code=404, detail=f"音色 '{voice_id}' 不存在")
    fns = generate_zs_sync(text,voice_id)
    logger.info(f"参数: voice_id{voice_id} {text} ,文件路径: {fns}")
    
    return FileResponse(
        path=fns,
        filename=fns,  # 客户端下载时显示的文件名
        media_type="audio/wav"  # 可选的 MIME 类型
    )    

@app.post("/tts/cross_lingual_sync")
async def cross_lingual_sync(
    text: str = Form(..., description="要合成的文本"),
    voice_id: Optional[str] = Form(default=None, description="音色ID (使用预加载的音色，零延迟)"),
): 
    if not voice_id:
        raise HTTPException(status_code=400, detail="voice_id 不能为空")
    if voice_id not in voice_cache:
        raise HTTPException(status_code=404, detail=f"音色 '{voice_id}' 不存在")
    fns = generate_cross_lingual_sync(text,voice_id)
    logger.info(f"参数: voice_id{voice_id} {text} ,文件路径: {fns}")
    return FileResponse(
        path=fns,
        filename=fns,  # 客户端下载时显示的文件名
        media_type="audio/wav"  # 可选的 MIME 类型
    )    


@app.post("/tts/zero_shot")
async def tts_zero_shot(
    text: str = Form(..., description="要合成的文本"),
    prompt_text: str = Form(..., description="参考音频对应的文本"),
    prompt_wav: UploadFile = File(..., description="参考音频文件 (WAV, 16kHz)")
):
    """
    Zero-shot 音色克隆接口
    
    返回: 流式 PCM 音频数据
    """
    if not text or len(text.strip()) == 0:
        raise HTTPException(status_code=400, detail="文本不能为空")
    if not prompt_text or len(prompt_text.strip()) == 0:
        raise HTTPException(status_code=400, detail="提示文本不能为空")
    
    try:
        # 直接传入文件对象 (SpooledTemporaryFile)
        prompt_wav_data = prompt_wav.file
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"音频文件加载失败: {e}")
    
    logger.info(f"Zero-shot TTS: text='{text[:50]}...', prompt='{prompt_text[:30]}...'")
    
    def stream_generator():
        for chunk in generate_audio_stream(text, prompt_text, prompt_wav_data, stream=True):
            yield chunk
    
    return StreamingResponse(
        stream_generator(),
        media_type="application/octet-stream",
        headers={
            "X-Sample-Rate": str(cosyvoice.sample_rate),
            "X-Channels": "1",
            "X-Bits": "16"
        }
    )


def load_model(model_dir: str, device: str = "mps", fp16: bool = False, use_vllm: bool = False):
    """加载 CosyVoice 模型"""
    global cosyvoice, voice_cache
    from cosyvoice.cli.cosyvoice import AutoModel
    
    logger.info(f"正在加载模型: {model_dir}")
    logger.info(f"设备: {device}, FP16: {fp16}, vLLM加速: {use_vllm}")
    
    if use_vllm:
        try:
            import vllm
        except ImportError:
            logger.error("启用 vLLM 失败: 未找到 vllm 库。请先安装: pip install vllm==0.9.0")
            sys.exit(1)
            
    start_time = time.time()
    try:
        # load_vllm 参数会自动触发 CosyVoice2/3 模型的 vLLM 加载逻辑
        # 注意: 库内部默认 gpu_memory_utilization=0.2 (适配 24G 显存)
        cosyvoice = AutoModel(model_dir=model_dir, fp16=fp16, load_vllm=use_vllm)
    except TypeError as e:
        if "load_vllm" in str(e):
             logger.error("当前 CosyVoice 版本似乎不支持 vLLM，请确保使用最新代码")
        raise e
        
    logger.info(f"模型加载完成，耗时: {time.time() - start_time:.1f}s")
    logger.info(f"模型采样率: {cosyvoice.sample_rate}Hz, 输出采样率: {output_sample_rate}Hz")
    
    # ========== 加载多音色配置 ==========
    asset_dir = os.path.join(SCRIPT_DIR, "../asset")
    official_asset_dir = os.path.join(SCRIPT_DIR, "official", "asset")
    
    logger.info(f"⚡ 正在加载 {len(VOICE_CONFIGS)} 个音色配置...")
    
    for voice_config in VOICE_CONFIGS:
        voice_id = voice_config["id"]
        voice_file = voice_config["file"]
        prompt_text = voice_config["prompt_text"]
        
        # 查找音频文件
        voice_path = os.path.join(asset_dir, voice_file)
        if not os.path.exists(voice_path):
            # 尝试从官方目录查找
            voice_path = os.path.join(official_asset_dir, voice_file)
        
        if not os.path.exists(voice_path):
            logger.warning(f"❌ 音色 '{voice_id}' 的文件未找到: {voice_file}")
            continue
        
        try:
            # 缓存音色特征
            cosyvoice.add_zero_shot_spk(prompt_text, voice_path, voice_id)
            
            # 保存到 voice_cache
            voice_cache[voice_id] = {
                "file": voice_path,
                "prompt_text": prompt_text
            }
            logger.info(f"✅ 音色 '{voice_id}' 加载成功: {voice_file}")
        except Exception as e:
            logger.warning(f"❌ 音色 '{voice_id}' 加载失败: {e}")
    
    logger.info(f"⚡ 音色加载完成，共 {len(voice_cache)} 个可用音色: {list(voice_cache.keys())}")
    
    # 预热推理 (使用第一个可用音色)
    first_voice_path = None
    if voice_cache:
        first_voice_id = list(voice_cache.keys())[0]
        first_voice_path = voice_cache[first_voice_id]["file"]
    warmup_model(first_voice_path, first_voice_id if voice_cache else None)
    
    return cosyvoice


def warmup_model(prompt_wav_path: str = None, voice_id: str = None):
    """预热模型，减少首次请求延迟"""
    global cosyvoice
    
    if cosyvoice is None:
        return
    
    logger.info("🔥 正在预热模型...")
    start_time = time.time()
    
    warmup_text = "预热测试"
    warmup_prompt_text = "预热"
    
    # 如果有参考音频，使用 zero-shot 预热
    if prompt_wav_path and os.path.exists(prompt_wav_path):
        try:
            # 使用指定的 voice_id 进行预热
            spk_id = voice_id if voice_id else "default"
            for _ in cosyvoice.inference_zero_shot(
                warmup_text, 
                warmup_prompt_text, 
                prompt_wav_path,
                stream=False,
                zero_shot_spk_id=spk_id
            ):
                pass
            logger.info(f"✅ 模型预热完成，耗时: {time.time() - start_time:.1f}s")
        except Exception as e:
            logger.warning(f"预热失败 (不影响正常使用): {e}")
    else:
        logger.info("⏭ 跳过预热 (无参考音频)")


def main():
    global output_sample_rate
    
    parser = argparse.ArgumentParser(description="CosyVoice TTS Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="监听地址")
    parser.add_argument("--port", type=int, default=9662, help="监听端口")
    parser.add_argument(
        "--model_dir", 
        type=str, 
        default="../pretrained_models/Fun-CosyVoice3-0.5B",
        help="模型目录路径"
    )
    parser.add_argument("--device", type=str, default="cuda", help="运行设备: cuda 或 cpu")
    parser.add_argument("--fp16", action="store_true", help="使用 FP16 推理 (节省显存)")
    parser.add_argument("--use_vllm", action="store_true", help="[优化] 使用 vLLM 加速推理 (需 pip install vllm)")
    parser.add_argument(
        "--output_sample_rate", 
        type=int, 
        default=16000, 
        choices=[16000, 24000],
        help="输出采样率: 16000 (兼容小智平台) 或 24000 (原生高质量)"
    )
    args = parser.parse_args()
    
    # 设置输出采样率
    output_sample_rate = args.output_sample_rate
    
    # 处理相对路径
    if not os.path.isabs(args.model_dir):
        args.model_dir = os.path.join(SCRIPT_DIR, args.model_dir)
    
    # 加载模型
    load_model(args.model_dir, args.device, args.fp16, args.use_vllm)
    
    # 启动服务
    logger.info(f"服务已启动: http://{args.host}:{args.port}")
    logger.info(f"健康检查: http://{args.host}:{args.port}/health")
    logger.info(f"TTS 接口: POST http://{args.host}:{args.port}/tts/stream")
    logger.info(f"📢 输出采样率: {output_sample_rate}Hz (模型原生: 24000Hz)")
    
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
