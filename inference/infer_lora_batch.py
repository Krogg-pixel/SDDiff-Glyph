"""
SDDiff-Glyph Inference Script

This script generates decorative Chinese glyph images
using the trained SDDiff-Glyph LoRA model.

Base model: Stable Diffusion v1.5
LoRA: trained on the Huaniaozi dataset

Paper:
SDDiff-Glyph: Learning Decorative Semantics for the Generation
of Non-Rigid Folk Typography
"""

import os
import torch
from diffusers import StableDiffusionPipeline

# ===== 路径配置 =====
BASE_MODEL = "./pretrained/sd15"
LORA_DIR = "./outputs/lora_huaniao_struct_v0"   # ←结构约束 LoRA
OUT_DIR = "./outputs/infer_struct_50"

os.makedirs(OUT_DIR, exist_ok=True)

# =====================================================
# Prompt used for decorative glyph generation
# =====================================================
# The prompt describes floral and avian decorative
# semantics used in Huaniaozi typography.
# ===== Prompt =====
PROMPT = (
    "a traditional Chinese decorative floral-bird glyph, "
    "clear character silhouette, pure white background, colorful motifs"
)

NEG_PROMPT = (
    "gray background, dark background, low contrast, blurry, noisy, text watermark"
)

# ===== 加载模型 =====
pipe = StableDiffusionPipeline.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
).to("cuda")

pipe.load_lora_weights(LORA_DIR)

pipe.set_progress_bar_config(disable=False)

# ===== 固定 seed，保证可复现 =====
generator = torch.Generator("cuda").manual_seed(6688)
#manual_seed(1234)manual_seed(2026)

# ===== 批量生成 =====
NUM_IMAGES = 50

for i in range(NUM_IMAGES):
    image = pipe(
        PROMPT,
        negative_prompt=NEG_PROMPT,
        num_inference_steps=30,
        guidance_scale=7.5,
        generator=generator,
    ).images[0]

    save_path = os.path.join(OUT_DIR, f"struct_{i:03d}.png")
    image.save(save_path)
    print(f"saved: {save_path}")

print("✅ Done. 50 images generated.")
