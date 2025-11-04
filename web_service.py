#!/usr/bin/env python3
"""
DeepSeek-OCR Web Service - 增强版
基于 transformers 的稳定实现（替代 vLLM）
集成了 Find 和 Freeform 功能
"""
import os
import re
import tempfile
import shutil
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps
import torch
from transformers import AutoModel, AutoTokenizer
import uvicorn

# 全局变量
model = None
tokenizer = None
MODEL_PATH = '/data/model/DeepSeek-OCR'

@asynccontextmanager
async def lifespan(app: FastAPI):
    """模型加载生命周期"""
    global model, tokenizer
    
    print("="*50)
    print("🚀 DeepSeek-OCR 增强版启动中...")
    print("="*50)
    
    try:
        print(f"📦 正在加载模型: {MODEL_PATH}")
        
        # 加载 tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
        )
        
        # 加载模型
        model = AutoModel.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            use_safetensors=True,
            attn_implementation="eager",
            torch_dtype=torch.bfloat16,
        ).eval().to("cuda")
        
        # 设置 pad token
        if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token_id", None) is not None:
            tokenizer.pad_token = tokenizer.eos_token
        if getattr(model.config, "pad_token_id", None) is None and getattr(tokenizer, "pad_token_id", None) is not None:
            model.config.pad_token_id = tokenizer.pad_token_id
        
        print("✅ 模型加载成功！")
        print("="*50)
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        raise
    
    yield
    
    print("🛑 服务关闭中...")

# FastAPI 应用
app = FastAPI(
    title="DeepSeek-OCR API - 增强版",
    description="智能 OCR 识别服务 · Find & Freeform 支持",
    version="3.0.0",
    lifespan=lifespan
)

# CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def build_prompt(
    mode: str,
    custom_prompt: str = "",
    find_term: str = "",
    grounding: bool = False
) -> str:
    """构建提示词"""
    
    # 模式映射
    prompt_templates = {
        "document": "<image>\n<|grounding|>Convert the document to markdown.",
        "ocr": "<image>\n<|grounding|>OCR this image.",
        "free": "<image>\nFree OCR. Only output the raw text.",
        "figure": "<image>\nParse the figure.",
        "describe": "<image>\nDescribe this image in detail.",
        "find": "<image>\n<|grounding|>Locate <|ref|>{term}<|/ref|> in the image.",
        "freeform": "<image>\n{prompt}",
    }
    
    # 构建最终 prompt
    if mode == "find":
        term = find_term.strip() or "Total"
        prompt = prompt_templates["find"].replace("{term}", term)
    elif mode == "freeform":
        user_prompt = custom_prompt.strip() or "OCR this image."
        prompt = prompt_templates["freeform"].replace("{prompt}", user_prompt)
    else:
        prompt = prompt_templates.get(mode, prompt_templates["document"])
    
    return prompt

def clean_grounding_text(text: str) -> str:
    """移除 grounding 标记"""
    cleaned = re.sub(
        r"<\|ref\|>(.*?)<\|/ref\|>\s*<\|det\|>\s*\[.*?\]\s*<\|/det\|>",
        r"\1",
        text,
        flags=re.DOTALL,
    )
    cleaned = re.sub(r"<\|grounding\|>", "", cleaned)
    return cleaned.strip()

def parse_detections(text: str, image_width: int, image_height: int) -> List[Dict[str, Any]]:
    """解析边界框坐标"""
    boxes = []
    
    DET_BLOCK = re.compile(
        r"<\|ref\|>(?P<label>.*?)<\|/ref\|>\s*<\|det\|>\s*(?P<coords>\[.*?\])\s*<\|/det\|>",
        re.DOTALL,
    )
    
    for m in DET_BLOCK.finditer(text or ""):
        label = m.group("label").strip()
        coords_str = m.group("coords").strip()
        
        try:
            import ast
            parsed = ast.literal_eval(coords_str)
            
            # 标准化为列表的列表
            if isinstance(parsed, list) and len(parsed) == 4 and all(isinstance(n, (int, float)) for n in parsed):
                box_coords = [parsed]
            elif isinstance(parsed, list):
                box_coords = parsed
            else:
                continue
            
            # 处理每个 box
            for box in box_coords:
                if isinstance(box, (list, tuple)) and len(box) >= 4:
                    # 从 0-999 归一化坐标转换为实际像素坐标
                    x1 = int(float(box[0]) / 999 * image_width)
                    y1 = int(float(box[1]) / 999 * image_height)
                    x2 = int(float(box[2]) / 999 * image_width)
                    y2 = int(float(box[3]) / 999 * image_height)
                    boxes.append({"label": label, "box": [x1, y1, x2, y2]})
        except Exception as e:
            print(f"❌ 解析坐标失败: {e}")
            continue
    
    return boxes

@app.get("/", response_class=HTMLResponse)
async def root():
    """返回 Web UI"""
    ui_file_path = Path(__file__).parent / "ocr_ui_modern.html"
    
    if ui_file_path.exists():
        with open(ui_file_path, 'r', encoding='utf-8') as f:
            return HTMLResponse(content=f.read())
    
    return HTMLResponse(content="<h1>DeepSeek-OCR Web UI</h1><p>UI file not found</p>")

@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "model": MODEL_PATH,
        "engine": "transformers",
        "model_loaded": model is not None,
        "gpu_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
    }

@app.post("/ocr")
async def ocr_endpoint(
    file: UploadFile = File(...),
    prompt_type: str = Form("document"),
    find_term: str = Form(""),
    custom_prompt: str = Form(""),
    grounding: bool = Form(False)
):
    """OCR 识别接口 - 增强版支持 Find 和 Freeform"""
    
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="模型未加载")
    
    tmp_file = None
    output_dir = None
    
    try:
        # 读取上传的图片数据
        image_data = await file.read()
        
        # 保存到临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png', mode='wb') as tmp:
            tmp.write(image_data)
            tmp_file = tmp.name
        
        print(f"📸 临时文件已保存: {tmp_file}")
        
        # 读取图片获取尺寸
        try:
            with Image.open(tmp_file) as img:
                img = ImageOps.exif_transpose(img)
                img = img.convert('RGB')
                orig_w, orig_h = img.size
                print(f"📐 图片尺寸: {orig_w}x{orig_h}")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"图片加载失败: {str(e)}")
        
        # 构建 prompt
        prompt = build_prompt(prompt_type, custom_prompt, find_term, grounding)
        print(f"💬 提示词: {prompt[:100]}...")
        
        # 创建输出目录
        output_dir = tempfile.mkdtemp(prefix="ocr_")
        
        # 执行推理
        print(f"🚀 开始推理...")
        result = model.infer(
            tokenizer,
            prompt=prompt,
            image_file=tmp_file,
            output_path=output_dir,
            base_size=1024,
            image_size=640,
            crop_mode=True,
            save_results=False,
            test_compress=False,
            eval_mode=True,
        )
        
        print(f"✅ 推理完成，结果类型: {type(result)}")
        
        # 处理结果
        if isinstance(result, str):
            text = result.strip()
        elif isinstance(result, dict) and "text" in result:
            text = str(result["text"]).strip()
        else:
            text = str(result).strip()
        
        # 如果没有结果，检查输出文件
        if not text:
            result_file = os.path.join(output_dir, "result.mmd")
            if os.path.exists(result_file):
                with open(result_file, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
        
        if not text:
            text = "模型未返回结果"
        
        print(f"📝 结果长度: {len(text)} 字符")
        
        # 解析 grounding boxes
        boxes = []
        if "<|det|>" in text or "<|ref|>" in text:
            boxes = parse_detections(text, orig_w, orig_h)
            print(f"📦 找到 {len(boxes)} 个边界框")
        
        # 清理显示文本
        display_text = clean_grounding_text(text)
        
        if not display_text and boxes:
            display_text = ", ".join([b["label"] for b in boxes])
        
        return JSONResponse({
            "success": True,
            "text": display_text,
            "raw_text": text,
            "boxes": boxes,
            "image_dims": {"w": orig_w, "h": orig_h},
            "prompt_type": prompt_type,
            "metadata": {
                "mode": prompt_type,
                "grounding": grounding or (prompt_type in ["find", "document", "ocr"]),
                "has_boxes": len(boxes) > 0,
                "engine": "transformers"
            }
        })
        
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"❌ 错误详情:\n{error_detail}")
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)
        
    finally:
        # 清理临时文件
        if tmp_file and os.path.exists(tmp_file):
            try:
                os.remove(tmp_file)
                print(f"🗑️ 临时文件已删除: {tmp_file}")
            except Exception as e:
                print(f"⚠️ 删除临时文件失败: {e}")
        if output_dir and os.path.exists(output_dir):
            shutil.rmtree(output_dir, ignore_errors=True)
            print(f"🗑️ 输出目录已清理: {output_dir}")

if __name__ == "__main__":
    import sys
    
    port = 8001
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except:
            port = 8001
    
    print("\n" + "="*50)
    print("🚀 DeepSeek-OCR 增强版 Web 服务")
    print("="*50)
    print(f"📍 访问地址: http://0.0.0.0:{port}")
    print(f"📚 API 文档: http://0.0.0.0:{port}/docs")
    print(f"❤️ 健康检查: http://0.0.0.0:{port}/health")
    print("="*50 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
