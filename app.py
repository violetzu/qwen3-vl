# app.py
import os
import tempfile
from typing import Optional

import torch
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"

app = FastAPI(title="Qwen3-VL-8B Vision-Language API")

# ---------- 啟動時載入模型（只載一次） ----------
print("Loading model and processor ...")
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    cache_dir="/workspace/models",
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(MODEL_NAME)
print("Model loaded.")

# ---------- 推理函式 ----------
def run_qwen_vl(media_path: str, media_type: str, prompt: str, max_new_tokens: int = 128) -> str:
    """
    media_type: 'video' 或 'image'
    media_path: 檔案在本機的路徑
    prompt: 使用者文字
    """
    if media_type not in {"video", "image"}:
        raise ValueError("media_type 必須是 'video' 或 'image'")

    # 根據 media_type 組 messages
    if media_type == "video":
        media_dict = {"type": "video", "video": media_path}
    else:
        media_dict = {"type": "image", "image": media_path}

    messages = [
        {
            "role": "user",
            "content": [
                media_dict,
                {"type": "text", "text": prompt},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )

    # 丟到模型所在裝置
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]
    output_texts = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return output_texts[0]


# ---------- API Endpoint ----------
@app.post("/chat-vl")
async def chat_vl(
    file: UploadFile = File(..., description="影片或圖片檔"),
    media_type: str = Form(..., description="media 類型：video 或 image"),
    prompt: str = Form("Describe this media.", description="文字提示"),
    max_new_tokens: int = Form(128, description="最大產生 token 數"),
):
    """
    multipart/form-data:
      - file: 上傳的影片或圖片
      - media_type: 'video' 或 'image'
      - prompt: 問題/描述指令
      - max_new_tokens: 選填
    回傳:
      { "text": "模型輸出的描述" }
    """
    media_type = media_type.lower()
    if media_type not in {"video", "image"}:
        raise HTTPException(status_code=400, detail="media_type 必須是 'video' 或 'image'")

    # 檢查副檔名（簡單防呆，可依需求放寬）
    filename = file.filename or ""
    ext = os.path.splitext(filename)[1].lower()
    if media_type == "video" and ext not in {".mp4", ".mov", ".avi", ".mkv"}:
        raise HTTPException(status_code=400, detail="請上傳影片檔 (mp4/mov/avi/mkv...)")
    if media_type == "image" and ext not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
        raise HTTPException(status_code=400, detail="請上傳圖片檔 (jpg/png/webp/bmp...)")

    # 儲存到暫存檔
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        # 執行推理
        text = run_qwen_vl(tmp_path, media_type, prompt, max_new_tokens)

        # 刪除暫存檔
        try:
            os.remove(tmp_path)
        except Exception:
            pass

        return JSONResponse({"text": text})

    except HTTPException:
        raise
    except Exception as e:
        # 出錯時也要確保暫存檔刪掉
        try:
            if "tmp_path" in locals() and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def root():
    return {"status": "ok", "message": "Qwen3-VL-8B API is running"}
