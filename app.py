import os
import json
import tempfile
import threading
from typing import Optional, List, Dict, Any

import torch
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from transformers import AutoModelForImageTextToText, AutoProcessor, TextIteratorStreamer
from dotenv import load_dotenv

# ---------- 載入 .env ----------
load_dotenv()
QPIKEY = os.getenv("QPIKEY", None)
if not QPIKEY:
    raise RuntimeError(".env 中沒有設定 QPIKEY")

MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
MODEL_CACHE_DIR = "/app/models"

app = FastAPI(title="Qwen3-VL-8B Vision-Language API")

print("Loading model...")
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    cache_dir=MODEL_CACHE_DIR,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(MODEL_NAME, cache_dir=MODEL_CACHE_DIR)
print("Model loaded.")


# ---------- 核心推論函式 1：同步/阻塞模式 (原始) ----------
def run_qwen_vl_blocking(messages, video_path=None, image_path=None, max_new_tokens=256):
    """原始的推論方式，等待生成結束後一次回傳結果"""
    prompt_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=[prompt_text],
        videos=[video_path] if video_path else None,
        images=[image_path] if image_path else None,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
        )

    input_len = inputs["input_ids"].shape[1]
    generated_ids_trimmed = generated_ids[:, input_len:]

    result = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    return result


# ---------- 核心推論函式 2：串流模式 (新增) ----------
def stream_qwen_vl(messages, video_path=None, image_path=None, max_new_tokens=256):
    """使用 Thread 與 Streamer 進行串流輸出"""
    prompt_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=[prompt_text],
        videos=[video_path] if video_path else None,
        images=[image_path] if image_path else None,
        return_tensors="pt",
    ).to(model.device)

    # 初始化 Streamer
    streamer = TextIteratorStreamer(
        processor, 
        skip_prompt=True, 
        skip_special_tokens=True
    )

    # 設定參數
    generation_kwargs = dict(
        inputs,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
    )

    # 啟動執行緒進行生成 (避免阻塞)
    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    # Generator: 逐步回傳
    for new_text in streamer:
        if new_text:
            yield new_text


# ---------- API Key Middleware ----------
@app.middleware("http")
async def verify_api_key(request: Request, call_next):
    if request.url.path in ["/"]:
        return await call_next(request)

    client_key = request.headers.get("X-API-Key")
    if client_key != QPIKEY:
        return JSONResponse(
            status_code=401,
            content={"detail": "Invalid or missing API Key."},
        )

    return await call_next(request)


# ---------- Endpoint 1: 原始 POST /chat-vl (JSON 回應) ----------
@app.post("/chat-vl")
async def chat_vl(
    file: Optional[UploadFile] = File(None),
    messages: str = Form(...),
    max_new_tokens: int = Form(256),
):
    try:
        msgs = json.loads(messages)
    except Exception as e:
        raise HTTPException(400, f"messages JSON 錯誤: {e}")

    need_video = False
    need_image = False
    for m in msgs:
        content = m.get("content")
        if isinstance(content, list):
            for c in content:
                if c.get("type") == "video": need_video = True
                if c.get("type") == "image": need_image = True

    if (need_video or need_image) and file is None:
        raise HTTPException(400, "messages 指定 video/image，但沒有上傳 file")

    temp_path = None
    video_path = None
    image_path = None

    try:
        if file:
            ext = os.path.splitext(file.filename)[1].lower() or ".bin"
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                tmp.write(await file.read())
                temp_path = tmp.name

            if need_video: video_path = temp_path
            if need_image: image_path = temp_path

        # 呼叫「阻塞式」推論函式
        output = run_qwen_vl_blocking(
            msgs,
            video_path=video_path,
            image_path=image_path,
            max_new_tokens=max_new_tokens,
        )

        return {"text": output}

    except Exception as e:
        raise HTTPException(500, str(e))

    finally:
        # 立即刪除暫存檔
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


# ---------- Endpoint 2: 新增 POST /chat-vl2 (串流回應) ----------
@app.post("/chat-vl2")
async def chat_vl_streaming(
    file: Optional[UploadFile] = File(None),
    messages: str = Form(...),
    max_new_tokens: int = Form(256),
):
    try:
        msgs = json.loads(messages)
    except Exception as e:
        raise HTTPException(400, f"messages JSON 錯誤: {e}")

    need_video = False
    need_image = False
    for m in msgs:
        content = m.get("content")
        if isinstance(content, list):
            for c in content:
                if c.get("type") == "video": need_video = True
                if c.get("type") == "image": need_image = True

    if (need_video or need_image) and file is None:
        raise HTTPException(400, "messages 指定 video/image，但沒有上傳 file")

    temp_path = None
    video_path = None
    image_path = None

    # 處理檔案落地 (邏輯同上，但為了獨立性再寫一次，或可封裝成 function)
    if file:
        ext = os.path.splitext(file.filename)[1].lower() or ".bin"
        # delete=False 必須稍後手動刪除
        f = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
        try:
            content = await file.read()
            f.write(content)
            temp_path = f.name
        finally:
            f.close()

        if need_video: video_path = temp_path
        if need_image: image_path = temp_path

    # 定義 Generator，負責包裝串流與清理檔案
    async def response_generator():
        try:
            for chunk in stream_qwen_vl(
                msgs,
                video_path=video_path,
                image_path=image_path,
                max_new_tokens=max_new_tokens,
            ):
                yield chunk
        except Exception as e:
            yield f"[Error: {str(e)}]"
        finally:
            # 確保串流結束後刪除暫存檔
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

    # 回傳 StreamingResponse
    return StreamingResponse(response_generator(), media_type="text/plain")


@app.get("/")
def root():
    return {
        "status": "ok", 
        "endpoints": {
            "/chat-vl": "Original blocking JSON API",
            "/chat-vl2": "New streaming text API"
        }
    }