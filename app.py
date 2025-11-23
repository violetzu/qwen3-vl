import os
import json
import tempfile
from typing import Optional, List, Dict, Any

import torch
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse
from transformers import AutoModelForImageTextToText, AutoProcessor
from dotenv import load_dotenv

# ---------- 載入 .env ----------
load_dotenv()
QPIKEY = os.getenv("QPIKEY", None)
if not QPIKEY:
    raise RuntimeError(" .env 中沒有設定 QPIKEY")

MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
MODEL_CACHE_DIR = "/workspace/models"

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


def run_qwen_vl(messages, video_path=None, image_path=None, max_new_tokens=256):
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
        )

    input_len = inputs["input_ids"].shape[1]
    generated_ids_trimmed = generated_ids[:, input_len:]

    result = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    return result


# ---------- API Key Middleware：統一在這裡檢查 X-API-Key ----------
@app.middleware("http")
async def verify_api_key(request: Request, call_next):
    # 這邊可以排除某些路徑不用驗證
    if request.url.path in ["/"]:
        return await call_next(request)

    client_key = request.headers.get("X-API-Key")
    if client_key != QPIKEY:
        return JSONResponse(
            status_code=401,
            content={"detail": "Invalid or missing API Key."},
        )

    return await call_next(request)


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
                if c.get("type") == "video":
                    need_video = True
                if c.get("type") == "image":
                    need_image = True

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

            if need_video:
                video_path = temp_path
            if need_image:
                image_path = temp_path

        output = run_qwen_vl(
            msgs,
            video_path=video_path,
            image_path=image_path,
            max_new_tokens=max_new_tokens,
        )

        return {"text": output}

    except Exception as e:
        raise HTTPException(500, str(e))

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@app.get("/")
def root():
    return {"status": "ok", "message": "Qwen3-VL-8B API is running"}
