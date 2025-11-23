import os
import json
import tempfile
from typing import Optional, List, Dict, Any

import torch
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse
from transformers import AutoModelForImageTextToText, AutoProcessor

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



@app.post("/chat-vl")
async def chat_vl(
    file: Optional[UploadFile] = File(None),
    messages: str = Form(...),
    max_new_tokens: int = Form(256),
):
    # 讀 messages JSON
    try:
        msgs = json.loads(messages)
    except Exception as e:
        raise HTTPException(400, f"messages JSON 錯誤: {e}")

    # 掃描是否包含 video/image
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

    # 如果需要 file 但沒附檔 → error
    if (need_video or need_image) and file is None:
        raise HTTPException(400, "messages 指定 video/image，但沒有上傳 file")

    temp_path = None
    video_path = None
    image_path = None

    try:
        # 如果有檔案 → 存到 tmp
        if file:
            ext = os.path.splitext(file.filename)[1].lower() or ".bin"
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                tmp.write(await file.read())
                temp_path = tmp.name

            # 根據 messages 判斷是 video 還是 image
            if need_video:
                video_path = temp_path
            if need_image:
                image_path = temp_path

        # 執行推理
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


# ---------- 全域擋瀏覽器 User-Agent ----------
@app.middleware("http")
async def block_browsers(request: Request, call_next):
    ua = request.headers.get("user-agent", "")
    if any(k in ua for k in ["Mozilla", "Chrome", "Safari", "Firefox", "Edge", "Opera"]):
        return JSONResponse(
            status_code=403,
            content={"detail": "Browser access is not allowed."},
        )
    return await call_next(request)


@app.get("/")
def root():
    return {"status": "ok", "message": "Qwen3-VL-8B API is running"}
