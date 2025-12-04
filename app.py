import os
import json
import tempfile
import threading
from typing import Optional

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


# ---------- Helper ----------

def parse_messages(messages_str: str):
    try:
        return json.loads(messages_str)
    except Exception as e:
        raise HTTPException(400, f"messages JSON 錯誤: {e}")


def analyze_media_requirements(msgs):
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
    return need_video, need_image


async def save_upload_tempfile(file: Optional[UploadFile]) -> Optional[str]:
    if file is None:
        return None
    ext = os.path.splitext(file.filename)[1].lower() or ".bin"
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        content = await file.read()
        tmp.write(content)
        return tmp.name


def build_media_paths(temp_path: Optional[str], need_video: bool, need_image: bool):
    video_path = temp_path if need_video else None
    image_path = temp_path if need_image else None
    return video_path, image_path


def _build_inputs(messages, video_path=None, image_path=None):
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

    return inputs


# ---------- 核心推論：阻塞版 ----------

def run_qwen_vl_blocking(messages, video_path=None, image_path=None, max_new_tokens=256):
    inputs = _build_inputs(messages, video_path, image_path)

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


# ---------- 核心推論：串流版 ----------

def run_qwen_vl_streaming(messages, video_path=None, image_path=None, max_new_tokens=256):
    inputs = _build_inputs(messages, video_path, image_path)

    streamer = TextIteratorStreamer(
        processor,
        skip_prompt=True,
        skip_special_tokens=True
    )

    generation_kwargs = dict(
        inputs,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
    )

    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

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

# ---------- 單一 Endpoint，同時支援 JSON / 串流 ----------

@app.post("/chat-vl")
async def chat_vl(
    file: Optional[UploadFile] = File(None),
    messages: str = Form(...),
    max_new_tokens: int = Form(256),
    stream: bool = Form(False),
):
    msgs = parse_messages(messages)
    need_video, need_image = analyze_media_requirements(msgs)

    if (need_video or need_image) and file is None:
        raise HTTPException(400, "messages 指定 video/image，但沒有上傳 file")

    temp_path = None
    video_path = None
    image_path = None

    if file:
        temp_path = await save_upload_tempfile(file)
        video_path, image_path = build_media_paths(temp_path, need_video, need_image)

    # 非串流模式：一次回傳 JSON
    if not stream:
        try:
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
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

    # 串流模式：StreamingResponse
    async def response_generator():
        try:
            for chunk in run_qwen_vl_streaming(
                msgs,
                video_path=video_path,
                image_path=image_path,
                max_new_tokens=max_new_tokens,
            ):
                yield chunk
        except Exception as e:
            yield f"[Error: {str(e)}]"
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

    return StreamingResponse(response_generator(), media_type="text/plain")


@app.get("/")
def root():
    return {
        "status": "ok",
        "endpoint": "/chat-vl",
        "note": "使用 form field stream=true 啟用串流",
    }