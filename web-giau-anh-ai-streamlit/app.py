"""Streamlit interface for the ORIGINAL HidingNet / RevealNet architecture.

Copy HidingUNet.py and RevealNet.py from the original project to models/.
The original .pth checkpoints can be placed in checkPoint/ for local runs,
or configured as GitHub Release URLs in Streamlit Cloud secrets.
No user image is intentionally written to disk.
"""
from __future__ import annotations

import io
import os
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image, UnidentifiedImageError

from models.HidingUNet import UnetGenerator
from models.RevealNet import RevealNet

ROOT = Path(__file__).resolve().parent
SIZE = 256
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = 20_000_000

st.set_page_config(page_title="Steganography", page_icon="🔐", layout="centered")
torch.set_num_threads(min(2, os.cpu_count() or 1))


def setting(name: str) -> str:
    value = os.environ.get(name, "")
    if value:
        return value
    try:
        return str(st.secrets.get(name, ""))
    except Exception:
        return ""


def checkpoint_file(kind: str) -> Path:
    
    local_dir = ROOT / "checkPoint"
    if local_dir.is_dir():
        exact = local_dir / ("hnet.pth" if kind == "HNET" else "rnet.pth")
        if exact.is_file():
            return exact
        candidates = sorted(local_dir.glob("netH*.pth" if kind == "HNET" else "netR*.pth"))
        if candidates:
            return candidates[0]

    url = setting(f"{kind}_URL").strip()
    if not url.startswith("https://"):
        raise FileNotFoundError(
            f"Chưa có {kind} checkpoint. Hãy tải file .pth lên GitHub Release "
            f"và đặt {kind}_URL trong Streamlit Secrets."
        )

    
    model_dir = Path(tempfile.gettempdir()) / "deep_stego_models"
    model_dir.mkdir(parents=True, exist_ok=True)
    target = model_dir / ("hnet.pth" if kind == "HNET" else "rnet.pth")
    if target.is_file() and target.stat().st_size > 100_000:
        return target

    temporary = target.with_suffix(".download")
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "DeepStegoStreamlit/1.0"})
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as f:
            downloaded = 0
            while chunk := response.read(1024 * 1024):
                downloaded += len(chunk)
                if downloaded > 500 * 1024 * 1024:
                    raise ValueError("Checkpoint lớn bất thường (>500 MB).")
                f.write(chunk)
        if temporary.stat().st_size < 100_000:
            raise ValueError("Checkpoint tải về không hợp lệ hoặc quá nhỏ.")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def load_state(model: nn.Module, kind: str) -> nn.Module:
    path = checkpoint_file(kind)
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError(f"{kind}: checkpoint không phải state_dict.")
    
    if state and all(str(k).startswith("module.") for k in state):
        state = {k[len("module."):]: v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


@st.cache_resource(show_spinner="Đang tải mô hình HidingNet…")
def get_hnet() -> nn.Module:
    model = UnetGenerator(input_nc=6, output_nc=3, num_downs=7, output_function=nn.Sigmoid)
    return load_state(model, "HNET")


@st.cache_resource(show_spinner="Đang tải mô hình RevealNet…")
def get_rnet() -> nn.Module:
    model = RevealNet(output_function=nn.Sigmoid())
    return load_state(model, "RNET")


def open_image(uploaded) -> Image.Image:
    data = uploaded.getvalue()
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError("File ảnh lớn hơn 10 MB. Hãy chọn ảnh nhỏ hơn.")
    try:
        with Image.open(io.BytesIO(data)) as image:
            return image.convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("Không đọc được ảnh. Hãy dùng PNG, JPG hoặc WebP hợp lệ.") from exc


def make_tensor(image: Image.Image) -> torch.Tensor:
    
    image = image.resize((SIZE, SIZE), resample=Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.uint8).copy()
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous().float().div_(255.0)
    return tensor.unsqueeze(0)


def tensor_png(tensor: torch.Tensor) -> bytes:
    array = (tensor.detach().clamp(0, 1).squeeze(0).mul(255.0)
             .round().to(torch.uint8).permute(1, 2, 0).cpu().numpy())
    image = Image.fromarray(array, mode="RGB")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


@torch.inference_mode()
def hide(cover: Image.Image, secret: Image.Image) -> bytes:
    x = torch.cat([make_tensor(cover), make_tensor(secret)], dim=1)
    return tensor_png(get_hnet()(x))


@torch.inference_mode()
def reveal(container: Image.Image) -> bytes:
    return tensor_png(get_rnet()(make_tensor(container)))



st.title("Steganography")
st.caption("HCMUS - Group 7 - PHY00010")
st.info(
    "Đầu ra 256×256. Nên tải Container dưới dạng PNG để đảm bảo khả năng "
    "khôi phục. Website không lưu dữ liệu vào database."
)

hide_tab, reveal_tab = st.tabs(["🔐 Giấu ảnh", "🔎 Khôi phục ảnh"])

with hide_tab:
    st.subheader("Cover + Secret → Container")
    cover_file = st.file_uploader(
        "Ảnh Cover", type=["png", "jpg", "jpeg", "webp"], key="cover_file"
    )
    secret_file = st.file_uploader(
        "Ảnh Secret", type=["png", "jpg", "jpeg", "webp"], key="secret_file"
    )
    if st.button("Giấu ảnh", disabled=not (cover_file and secret_file)):
        try:
            with st.spinner("Đang xử lý bằng HidingNet…"):
                st.session_state["container_png"] = hide(
                    open_image(cover_file), open_image(secret_file)
                )
            st.success("Đã tạo ảnh Container.")
        except Exception as exc:
            st.error(f"Không thể giấu ảnh: {exc}")
    if "container_png" in st.session_state:
        st.image(st.session_state["container_png"], caption="Ảnh Container 256×256")
        st.download_button(
            "Tải Container.png", st.session_state["container_png"],
            "container.png", "image/png", key="download_container"
        )

with reveal_tab:
    st.subheader("Container → Revealed")
    container_file = st.file_uploader(
        "Ảnh Container", type=["png", "jpg", "jpeg", "webp"], key="container_file"
    )
    if st.button("Khôi phục ảnh", disabled=not container_file):
        try:
            with st.spinner("Đang xử lý bằng RevealNet…"):
                st.session_state["revealed_png"] = reveal(open_image(container_file))
            st.success("Đã khôi phục ảnh.")
        except Exception as exc:
            st.error(f"Không thể khôi phục ảnh: {exc}")
    if "revealed_png" in st.session_state:
        st.image(st.session_state["revealed_png"], caption="Ảnh Revealed 256×256")
        st.download_button(
            "Tải Revealed.png", st.session_state["revealed_png"],
            "revealed.png", "image/png", key="download_revealed"
        )

st.divider()
if st.button("Xóa kết quả đang hiển thị khỏi phiên này"):
    st.session_state.pop("container_png", None)
    st.session_state.pop("revealed_png", None)
    st.rerun()
