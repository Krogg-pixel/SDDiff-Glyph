import os
import sys
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import lpips

from skimage.metrics import structural_similarity
from pytorch_fid import fid_score



REAL_DIR = "evaluation/GT"
GEN_DIR = "evaluation/doubao"  # 可改为 evaluation/baseline 或 evaluation/structure

# 真实/生成图像都统一为 512x512；若不统一，本脚本会强制 resize
FORCE_RESIZE = (512, 512)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================================================
# 工具函数
# =========================================================
def is_image_file(filename: str) -> bool:
    """只允许常见图像后缀，自动跳过子目录与无关文件。"""
    return filename.lower().endswith((".png", ".jpg", ".jpeg"))


def list_images(folder: str):
    """只列出 folder 下的图像文件（不包含子目录）。"""
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Folder not found: {folder}")

    files = []
    for f in os.listdir(folder):
        full = os.path.join(folder, f)
        if os.path.isfile(full) and is_image_file(f):
            files.append(f)
    return sorted(files)


def load_image_np(path: str, resize_hw=None) -> np.ndarray:
    """读取为 numpy (H,W,3), float32, range [0,255]."""
    img = Image.open(path).convert("RGB")
    if resize_hw is not None:
        img = img.resize((resize_hw[1], resize_hw[0]), Image.BICUBIC)
    arr = np.array(img).astype(np.float32)
    return arr


def load_image_tensor_for_lpips(path: str, resize_hw=None) -> torch.Tensor:
    """
    读取为 torch tensor: (1,3,H,W), float32, range [-1,1]
    LPIPS 标准输入。
    """
    img = Image.open(path).convert("RGB")
    if resize_hw is not None:
        img = img.resize((resize_hw[1], resize_hw[0]), Image.BICUBIC)

    arr = np.array(img).astype(np.float32) / 255.0
    ten = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    ten = ten * 2.0 - 1.0
    return ten


def validate_pairing(real_dir: str, gen_dir: str):
    real_files = set(list_images(real_dir))
    gen_files = set(list_images(gen_dir))

    missing = sorted(list(real_files - gen_files))
    extra = sorted(list(gen_files - real_files))

    if missing:
        print("\n[ERROR] Generated folder is missing files that exist in real folder:")
        for m in missing[:30]:
            print("  -", m)
        raise RuntimeError("File pairing check failed.")

    if extra:
        print("\n[WARN] Generated folder has extra files not in real folder (ignored):")
        for e in extra[:30]:
            print("  -", e)

    paired = sorted(list(real_files))
    if len(paired) == 0:
        raise RuntimeError("No valid images found in real folder.")
    return paired


# =========================================================
# 指标计算
# =========================================================
def compute_ssim_metric(real_dir: str, gen_dir: str, filenames):
    ssim_list = []

    for name in tqdm(filenames, desc="SSIM"):
        real_path = os.path.join(real_dir, name)
        gen_path = os.path.join(gen_dir, name)

        real_img = load_image_np(real_path, resize_hw=FORCE_RESIZE)
        gen_img = load_image_np(gen_path, resize_hw=FORCE_RESIZE)

        ssim = structural_similarity(real_img, gen_img, channel_axis=-1, data_range=255)

        ssim_list.append(ssim)

    return float(np.mean(ssim_list))


def compute_lpips_metric(real_dir: str, gen_dir: str, filenames):
    loss_fn = lpips.LPIPS(net="alex").to(DEVICE)
    lpips_list = []

    for name in tqdm(filenames, desc="LPIPS"):
        real_path = os.path.join(real_dir, name)
        gen_path = os.path.join(gen_dir, name)

        real_t = load_image_tensor_for_lpips(real_path, resize_hw=FORCE_RESIZE).to(DEVICE)
        gen_t = load_image_tensor_for_lpips(gen_path, resize_hw=FORCE_RESIZE).to(DEVICE)

        with torch.no_grad():
            dist = loss_fn(real_t, gen_t)

        lpips_list.append(dist.item())

    return float(np.mean(lpips_list))


def compute_fid_metric(real_dir: str, gen_dir: str):
    fid = fid_score.calculate_fid_given_paths(
        [real_dir, gen_dir],
        batch_size=50,
        device=DEVICE,
        dims=2048,
    )
    return float(fid)


# =========================================================
# 主程序
# =========================================================
if __name__ == "__main__":
    print(f"Evaluating: {GEN_DIR}")
    print(f"Real dir   : {REAL_DIR}")
    print(f"Device     : {DEVICE}")

    filenames = validate_pairing(REAL_DIR, GEN_DIR)
    print(f"Paired images: {len(filenames)}")

    # SSIM
    ssim = compute_ssim_metric(REAL_DIR, GEN_DIR, filenames)

    # LPIPS
    lpips_value = compute_lpips_metric(REAL_DIR, GEN_DIR, filenames)

    # FID
    fid_value = compute_fid_metric(REAL_DIR, GEN_DIR)

    print("\n===== Quantitative Results =====")
    print(f"SSIM  : {ssim:.4f}  (higher is better)")
    print(f"LPIPS : {lpips_value:.4f}  (lower is better)")
    print(f"FID   : {fid_value:.4f}  (lower is better)")
