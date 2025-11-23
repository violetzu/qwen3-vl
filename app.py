# app.py
import os
import tempfile
from typing import Optional

import torch
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
MODEL_CACHE_DIR = "/workspace/models"

app = FastAPI(title="Qwen3-VL-8B Vision-Language API")

# ---------- 啟動時載入模型（只載一次） ----------
print("Loading model and processor ...")
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    cache_dir=MODEL_CACHE_DIR,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(MODEL_NAME, cache_dir=MODEL_CACHE_DIR)
print("Model loaded.")


# ---------- 推理函式 ----------
def run_qwen_vl(
    media_path: Optional[str],
    media_type: str,
    prompt: str,
    max_new_tokens: int = 128,
) -> str:
    """
    media_type: 'video' / 'image' / 'text'
    media_path: 檔案在本機的路徑（text 模式可以是 None）
    prompt: 使用者文字
    """
    if media_type not in {"video", "image", "text"}:
        raise ValueError("media_type 必須是 'video'、'image' 或 'text'")

    # 根據 media_type 組 messages
    if media_type == "text":
        # 純文字對話，不帶任何影音
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ],
            }
        ]
    else:
        if not media_path:
            raise ValueError("media_path 不可為空（video/image 模式必須有檔案路徑）")

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
    # 變成可選的，text 模式就可以不傳
    file: Optional[UploadFile] = File(
        None,
        description="影片或圖片檔（media_type=text 時可省略）",
    ),
    media_type: str = Form(..., description="media 類型：video、image 或 text"),
    prompt: str = Form("Describe this media.", description="文字提示"),
    max_new_tokens: int = Form(128, description="最大產生 token 數"),
):
    """
    multipart/form-data:
      - file: 上傳的影片或圖片（text 模式可不傳）
      - media_type: 'video' / 'image' / 'text'
      - prompt: 問題/描述指令
      - max_new_tokens: 選填
    回傳:
      { "text": "模型輸出的描述" }
    """
    media_type = media_type.lower()
    if media_type not in {"video", "image", "text"}:
        raise HTTPException(status_code=400, detail="media_type 必須是 'video'、'image' 或 'text'")

    # --- 純文字模式：不需要檔案 ---
    if media_type == "text":
        try:
            text = run_qwen_vl(
                media_path=None,
                media_type="text",
                prompt=prompt,
                max_new_tokens=max_new_tokens,
            )
            return JSONResponse({"text": text})
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # --- 以下是 video / image 模式，需要檔案 ---
    if file is None:
        raise HTTPException(
            status_code=400,
            detail="media_type 為 'video' 或 'image' 時必須上傳 file",
        )

    filename = file.filename or ""
    ext = os.path.splitext(filename)[1].lower()

    # 檢查副檔名（簡單防呆，可依需求放寬）
    if media_type == "video" and ext not in {".mp4", ".mov", ".avi", ".mkv"}:
        raise HTTPException(status_code=400, detail="請上傳影片檔 (mp4/mov/avi/mkv...)")
    if media_type == "image" and ext not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
        raise HTTPException(status_code=400, detail="請上傳圖片檔 (jpg/png/webp/bmp...)")

    # 儲存到暫存檔
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        # 執行推理
        text = run_qwen_vl(tmp_path, media_type, prompt, max_new_tokens)

        # 刪除暫存檔
        try:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass

        return JSONResponse({"text": text})

    except HTTPException:
        # 傳遞 FastAPI 自己的錯誤
        try:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        raise
    except Exception as e:
        # 出錯時也要確保暫存檔刪掉
        try:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=str(e))

# ---------- 全域擋瀏覽器 User-Agent ----------
@app.middleware("http")
async def block_browsers(request: Request, call_next):
    ua = request.headers.get("user-agent", "") or ""

    blocked_keywords = ["Mozilla", "Chrome", "Safari", "Firefox", "Edge", "Edg", "Opera"]

    if any(k in ua for k in blocked_keywords):
        return JSONResponse(
            status_code=403,
            content={"detail": "Browser access is not allowed."},
        )

    # 非瀏覽器就照正常流程往下跑
    response = await call_next(request)
    return response

@app.get("/")
def root():
    return {"status": "ok", "message": "Qwen3-VL-8B API is running"}
