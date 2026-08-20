# ============================================================
# app.py – Raon-VisionEncoder + zvec + PyWebView
# ============================================================
import os
import sys
import gc
import json
import time
import logging
import threading
import traceback
import tkinter as tk
from tkinter import filedialog
from pathlib import Path

import warnings
warnings.filterwarnings("ignore", message="Palette images")

import numpy as np
import requests
import torch
import torch.nn.functional as F
import webview
from PIL import Image
from tqdm import tqdm

# ── 현재 프로젝트 경로를 sys.path에 추가 ──────────────────────
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

# raon_vision_encoder 패키지가 프로젝트 안에 있어야 함
RAON_PKG_DIR = APP_DIR / "raon_vision_encoder"

# ── 모델 파일 다운로드 경로 ───────────────────────────────────
_local_app_data = os.environ.get("LOCALAPPDATA", "").strip()
if not _local_app_data:
    if os.name == "nt":
        _local_app_data = os.path.join(os.path.expanduser("~"), "AppData", "Local")
    else:
        _local_app_data = os.path.join(os.path.expanduser("~"), ".local", "share")
LOCAL_APP_DATA = os.path.join(_local_app_data, "RaonVisionEncoder")
CONFIG_URL = (
    "https://huggingface.co/KRAFTON/Raon-VisionEncoder"
    "/resolve/main/config.json"
)
WEIGHT_URL = (
    "https://huggingface.co/KRAFTON/Raon-VisionEncoder"
    "/resolve/main/model.safetensors"
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".gif"}

# ── 로깅 설정 ────────────────────────────────────────────────
os.makedirs(LOCAL_APP_DATA, exist_ok=True)
LOG_FILE = os.path.join(LOCAL_APP_DATA, "app.log")

logger = logging.getLogger("RaonVisionSearch")
logger.setLevel(logging.DEBUG)

_fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
_fh.setLevel(logging.DEBUG)
_fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(funcName)s - %(message)s"))

_ch = logging.StreamHandler()
_ch.setLevel(logging.INFO)
_ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

logger.addHandler(_fh)
logger.addHandler(_ch)

logger.info("=== RaonVisionSearch 앱 시작 ===")
logger.info("APP_DIR: %s", APP_DIR)
logger.info("LOCAL_APP_DATA: %s", LOCAL_APP_DATA)
logger.info("LOG_FILE: %s", LOG_FILE)

# ── 만료된 HF 토큰 파일 제거 (401 방지) ────────────────────
_hf_token_path = os.path.join(
    os.path.expanduser("~"), ".cache", "huggingface", "token"
)
if os.path.isfile(_hf_token_path):
    try:
        os.remove(_hf_token_path)
        logger.info("[HF] 만료된 토큰 파일 제거: %s", _hf_token_path)
    except Exception as e:
        logger.warning("[HF] 토큰 파일 제거 실패: %s", e)


# ============================================================
#  zvec – 경량 numpy 벡터 DB
# ============================================================
class ZVec:
    """초경량 벡터 저장소 (cosine similarity 기반)."""

    def __init__(self, dim: int = 1152):
        self.dim = dim
        self.ids: list[str] = []
        self.meta: list[dict] = []
        self.vectors: np.ndarray | None = None  # (N, dim)

    # ── 추가 ──────────────────────────────────────────────────
    def add(self, vec_id: str, vector: np.ndarray, metadata: dict | None = None):
        v = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        if self.vectors is None:
            self.vectors = v
        else:
            self.vectors = np.vstack([self.vectors, v])
        self.ids.append(vec_id)
        self.meta.append(metadata or {})

    # ── 검색 (cosine) ────────────────────────────────────────
    def search(self, query: np.ndarray, top_k: int = 10):
        if self.vectors is None or len(self.ids) == 0:
            return []
        q = np.asarray(query, dtype=np.float32).reshape(1, -1)
        q_norm = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-9)
        db_norm = self.vectors / (
            np.linalg.norm(self.vectors, axis=1, keepdims=True) + 1e-9
        )
        scores = (q_norm @ db_norm.T).squeeze(0)  # (N,)
        top_idx = np.argsort(scores)[::-1][:top_k]
        results = []
        for i in top_idx:
            results.append(
                {"id": self.ids[i], "score": float(scores[i]), **self.meta[i]}
            )
        return results

    # ── 저장 / 로드 ──────────────────────────────────────────
    def save(self, path: str):
        data = {
            "dim": self.dim,
            "ids": self.ids,
            "meta": self.meta,
        }
        np.savez_compressed(path, vectors=self.vectors, **{})
        json_path = path.replace(".npz", ".json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    def load(self, path: str):
        npz = np.load(path)
        self.vectors = npz["vectors"]
        self.dim = self.vectors.shape[1]
        json_path = path.replace(".npz", ".json")
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.ids = data["ids"]
        self.meta = data["meta"]

    def __len__(self):
        return len(self.ids)


# ============================================================
#  모델 다운로드 & 로드
# ============================================================
def _download_file(url: str, dest: str, desc: str = ""):
    """파일 다운로드 (진행률 표시)."""
    print(f"[다운로드] {desc or url}")
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=desc or os.path.basename(dest)
    ) as bar:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            f.write(chunk)
            bar.update(len(chunk))
    print(f"[완료] {dest}")


def ensure_model_files() -> str:
    """config.json / model.safetensors 존재 확인 → 없으면 다운로드.
    반환값: 모델이 저장된 로컬 디렉토리 경로.
    """
    os.makedirs(LOCAL_APP_DATA, exist_ok=True)

    config_path = os.path.join(LOCAL_APP_DATA, "config.json")
    weight_path = os.path.join(LOCAL_APP_DATA, "model.safetensors")

    # ── config.json ───────────────────────────────────────────
    if os.path.isfile(config_path):
        print(f"[체크] config.json 이미 존재: {config_path}")
    else:
        print(f"[체크] config.json 없음 → 다운로드 시작")
        _download_file(CONFIG_URL, config_path, "config.json")

    # ── model.safetensors ─────────────────────────────────────
    if os.path.isfile(weight_path):
        print(f"[체크] model.safetensors 이미 존재: {weight_path}")
    else:
        print(f"[체크] model.safetensors 없음 → 다운로드 시작")
        _download_file(WEIGHT_URL, weight_path, "model.safetensors")

    return LOCAL_APP_DATA


def load_raon_model(model_dir: str):
    """RaonVEModel + Processor 로드 (float16, 메모리 최적화)."""
    import gc
    from configuration_raonve import RaonVEConfig
    from modeling_raonve import RaonVEModel, RaonVEProcessor

    logger.info("[모델] config 로드 중…")
    config = RaonVEConfig.from_pretrained(model_dir)

    logger.info("[모델] RaonVEModel 생성 중…")
    model = RaonVEModel(config)

    # safetensors 가중치 로드 (float16 변환)
    weight_path = os.path.join(model_dir, "model.safetensors")
    if os.path.isfile(weight_path):
        from safetensors.torch import load_file
        logger.info("[모델] 가중치 로드 중: %s", weight_path)
        state = load_file(weight_path, device="cpu")
        state = {k: v.to(torch.float16) for k, v in state.items()}
        model.load_state_dict(state, strict=False)
        del state
        gc.collect()
        logger.info("[모델] 가중치 로드 완료 (float16)")

    model.eval()
    model = model.half()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    gc.collect()
    logger.info("[모델] 디바이스: %s, dtype: float16", device)

    logger.info("[프로세서] 로드 중…")
    processor = RaonVEProcessor.from_pretrained(model_dir)

    gc.collect()
    return model, processor, device

# ============================================================
#  PyWebView ↔ Python 브리지 (Api)
# ============================================================
class Api:
    def __init__(self):
        self.model = None
        self.processor = None
        self.device = None
        self.zvec_db = ZVec(dim=1152)
        self.indexed_count = 0
        self.total_images = 0
        self.progress_msg = ""
        self._indexing = False
        self._downloading = False
        self._download_pct = 0.0
        self._download_msg = ""
        self._model_ready = False
        logger.info("[Api] 초기화 완료")

    # ── 모델 준비 체크 (모든 이벤트 진입점) ──────────────────
    def _check_ready(self):
        if self._downloading:
            return {"ok": False, "msg": "모델 다운로드 중입니다. 완료 후 이용하세요."}
        if not self._model_ready:
            return {"ok": False, "msg": "모델이 준비되지 않았습니다. 먼저 다운로드하세요."}
        return None

    # ── 초기화 (백그라운드 스레드) ───────────────────────────
    def init_model(self):
        logger.info("[이벤트] 모델 다운로드/초기화 요청")
        if self._downloading:
            return {"ok": False, "msg": "이미 다운로드 중입니다."}
        if self._model_ready:
            return {"ok": True, "msg": "모델 이미 로드됨"}
        self._downloading = True
        self._download_pct = 0.0
        self._download_msg = "다운로드 준비 중..."
        threading.Thread(target=self._init_worker, daemon=True).start()
        logger.info("[다운로드] 백그라운드 스레드 시작")
        return {"ok": True, "msg": "다운로드 시작"}

    def _init_worker(self):
        try:
            model_dir = self._ensure_model_files_with_progress()
            logger.info("[모델] 모델 디렉토리: %s", model_dir)
            self._download_msg = "모델 가중치 로드 중..."
            self._download_pct = 99.0
            self.model, self.processor, self.device = load_raon_model(model_dir)
            logger.info("[모델] 로드 완료, 디바이스: %s", self.device)
            self._model_ready = True
            self._download_pct = 100.0
            self._download_msg = "모델 로드 완료!"
        except Exception as e:
            logger.error("[모델 초기화 오류] %s", e, exc_info=True)
            self._download_msg = f"오류: {e}"
        finally:
            self._downloading = False

    def _ensure_model_files_with_progress(self):
        os.makedirs(LOCAL_APP_DATA, exist_ok=True)
        config_path = os.path.join(LOCAL_APP_DATA, "config.json")
        weight_path = os.path.join(LOCAL_APP_DATA, "model.safetensors")

        if os.path.isfile(config_path):
            logger.info("[체크] config.json 존재: %s", config_path)
        else:
            logger.info("[체크] config.json 없음 → 다운로드")
            self._download_msg = "config.json 다운로드 중..."
            self._download_file_with_progress(CONFIG_URL, config_path, "config.json")

        if os.path.isfile(weight_path):
            logger.info("[체크] model.safetensors 존재: %s", weight_path)
        else:
            logger.info("[체크] model.safetensors 없음 → 다운로드")
            self._download_msg = "model.safetensors 다운로드 중 (4.5GB)..."
            self._download_pct = 0.0
            self._download_file_with_progress(WEIGHT_URL, weight_path, "model.safetensors")

        return LOCAL_APP_DATA

    def _download_file_with_progress(self, url, dest, desc):
        logger.info("[다운로드] %s 시작", desc)
        resp = requests.get(url, stream=True, timeout=60)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        downloaded = 0
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    self._download_pct = round(downloaded / total * 100, 1)
                mb_dl = downloaded // (1 << 20)
                mb_total = total // (1 << 20)
                self._download_msg = f"{desc} {self._download_pct}% ({mb_dl}MB / {mb_total}MB)"
        logger.info("[다운로드] %s 완료", desc)

    # ── 다운로드 진행률 조회 (JS 폴링용) ─────────────────────
    def get_download_progress(self):
        return {
            "downloading": self._downloading,
            "pct": self._download_pct,
            "msg": self._download_msg,
            "ready": self._model_ready,
        }

    # ── 폴더 선택 ─────────────────────────────────────────────
    def select_folder(self):
        logger.info("[이벤트] 폴더 선택 버튼 클릭")
        block = self._check_ready()
        if block:
            logger.warning("[차단] %s", block["msg"])
            return block
        try:
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            folder = filedialog.askdirectory(title="이미지 폴더 선택")
            root.destroy()
            logger.info("[폴더 다이얼로그] 반환값: '%s'", folder)
            if not folder:
                logger.warning("[폴더 다이얼로그] 폴더 미선택")
                return {"ok": False, "msg": "폴더가 선택되지 않았습니다."}
            logger.info("[폴더 선택] %s", folder)
            return {"ok": True, "path": folder}
        except Exception as e:
            logger.error("[폴더 선택 오류] %s", e, exc_info=True)
            return {"ok": False, "msg": str(e)}

    # ── 폴더 내 이미지 스캔 ───────────────────────────────────
    def scan_images(self, folder_path: str):
        logger.info("[이벤트] 이미지 스캔: %s", folder_path)
        block = self._check_ready()
        if block:
            return block
        images = []
        for root, _dirs, files in os.walk(folder_path):
            for fname in files:
                if Path(fname).suffix.lower() in IMAGE_EXTS:
                    images.append(os.path.join(root, fname))
        logger.info("[스캔] 발견: %d개", len(images))
        return {"ok": True, "count": len(images), "images": images}

    # ── 벡터화 + zvec 인덱싱 (백그라운드 스레드) ─────────────
    def start_indexing(self, folder_path: str):
        logger.info("[이벤트] 인덱싱 요청: %s", folder_path)
        block = self._check_ready()
        if block:
            return block
        if self._indexing:
            logger.warning("[인덱싱] 이미 진행 중")
            return {"ok": False, "msg": "이미 인덱싱 중입니다."}
        self._indexing = True
        threading.Thread(
            target=self._index_worker, args=(folder_path,), daemon=True
        ).start()
        logger.info("[인덱싱] 백그라운드 스레드 시작")
        return {"ok": True, "msg": "인덱싱 시작"}

    def _index_worker(self, folder_path: str):
        logger.info("[인덱싱 워커] 시작: %s", folder_path)
        try:
            images = []
            for root, _dirs, files in os.walk(folder_path):
                for fname in files:
                    if Path(fname).suffix.lower() in IMAGE_EXTS:
                        images.append(os.path.join(root, fname))

            self.total_images = len(images)
            self.indexed_count = 0
            self.zvec_db = ZVec(dim=1152)
            logger.info("[인덱싱 워커] 총 이미지: %d개", self.total_images)

            batch_size = 1
            for start in range(0, len(images), batch_size):
                batch_paths = images[start : start + batch_size]
                pil_imgs = []
                valid_paths = []
                for p in batch_paths:
                    try:
                        img = Image.open(p)
                        if img.mode in ("RGBA", "LA", "P"):
                            img = img.convert("RGBA")
                            bg = Image.new("RGB", img.size, (255, 255, 255))
                            bg.paste(img, mask=img.split()[-1])
                            img = bg
                        else:
                            img = img.convert("RGB")
                        pil_imgs.append(img)
                        valid_paths.append(p)
                    except Exception as img_err:
                        logger.warning("[인덱싱] 로드 실패: %s (%s)", p, img_err)
                        continue

                if not pil_imgs:
                    continue

                inputs = self.processor(images=pil_imgs, max_num_patches=256)
                pixel_values = inputs["pixel_values"].to(self.device)
                pixel_mask = inputs["pixel_attention_mask"].to(self.device)
                spatial_shapes = inputs["spatial_shapes"].to(self.device)

                with torch.inference_mode():
                    feats = self.model.encode_image(
                        pixel_values,
                        pixel_attention_mask=pixel_mask,
                        spatial_shapes=spatial_shapes,
                    )  # [B, 1152]

                feats_np = feats.float().cpu().numpy()
                del pixel_values, pixel_mask, spatial_shapes, feats
                gc.collect()
                for i, p in enumerate(valid_paths):
                    self.zvec_db.add(
                        vec_id=p,
                        vector=feats_np[i],
                        metadata={"path": p, "name": Path(p).name},
                    )
                    self.indexed_count += 1

                self.progress_msg = (
                    f"{self.indexed_count}/{self.total_images} 인덱싱 완료"
                )
                logger.info("[인덱싱] %s", self.progress_msg)

            save_path = os.path.join(LOCAL_APP_DATA, "zvec_index.npz")
            self.zvec_db.save(save_path)
            self.progress_msg = (
                f"인덱싱 완료! 총 {self.indexed_count}개 이미지. "
                f"저장: {save_path}"
            )
            logger.info("[인덱싱] %s", self.progress_msg)

        except Exception as e:
            logger.error("[인덱싱 오류] %s", e, exc_info=True)
            self.progress_msg = f"인덱싱 오류: {e}"
        finally:
            self._indexing = False
            logger.info("[인덱싱 워커] 종료")

    # ── 진행률 조회 ───────────────────────────────────────────
    def get_progress(self):
        logger.debug(
            "[진행률] %d/%d indexing=%s",
            self.indexed_count, self.total_images, self._indexing,
        )
        return {
            "indexing": self._indexing,
            "current": self.indexed_count,
            "total": self.total_images,
            "msg": self.progress_msg,
        }

    # ── 자연어 검색 ───────────────────────────────────────────
    def search(self, query: str, top_k: int = 20):
        logger.info("[이벤트] 검색: '%s' (top_k=%d)", query, top_k)
        block = self._check_ready()
        if block:
            return block
        if len(self.zvec_db) == 0:
            logger.error("[검색] 인덱스 비어있음")
            return {"ok": False, "msg": "인덱스가 비어 있습니다. 먼저 인덱싱하세요."}

        try:
            inputs = self.processor(text=[query])
            input_ids = inputs["input_ids"].to(self.device)
            logger.debug("[검색] input_ids shape: %s", input_ids.shape)

            with torch.inference_mode():
                text_feat = self.model.encode_text(input_ids)  # [1, 1152]

            text_np = text_feat.cpu().numpy().squeeze(0)
            logger.debug("[검색] text_feat shape: %s", text_np.shape)
            results = self.zvec_db.search(text_np, top_k=top_k)
            logger.info("[검색] 결과 %d개", len(results))
            return {"ok": True, "results": results}
        except Exception as e:
            logger.error("[검색 오류] %s", e, exc_info=True)
            return {"ok": False, "msg": str(e)}

    # ── zvec 인덱스 로드 ──────────────────────────────────────
    def load_index(self):
        logger.info("[이벤트] 저장된 인덱스 로드 요청")
        block = self._check_ready()
        if block:
            return block
        npz_path = os.path.join(LOCAL_APP_DATA, "zvec_index.npz")
        json_path = npz_path.replace(".npz", ".json")
        logger.debug("[인덱스] npz: %s", npz_path)
        logger.debug("[인덱스] json: %s", json_path)
        if os.path.isfile(npz_path) and os.path.isfile(json_path):
            self.zvec_db.load(npz_path)
            logger.info("[인덱스] 로드 완료: %d개", len(self.zvec_db))
            return {"ok": True, "msg": f"인덱스 로드 완료 ({len(self.zvec_db)}개)"}
        logger.warning("[인덱스] 저장된 파일 없음")
        return {"ok": False, "msg": "저장된 인덱스가 없습니다."}

# ============================================================
#  HTML / JS – PyWebView 렌더링
# ============================================================
HTML_PAGE = r"""
<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8"/>
<title>XY Zvec - Raon Vision Search</title>
<style>
  * { margin:0; padding:0; box-sizing:border-box; }
  body {
    font-family: 'Segoe UI', 'Malgun Gothic', sans-serif;
    background: #0f1117; color: #e0e0e0;
    display:flex; flex-direction:column; height:100vh;
  }
  header {
    background:#1a1d27; padding:16px 24px;
    display:flex; align-items:center; gap:16px;
    border-bottom:1px solid #2a2d3a;
  }
  header h1 { font-size:20px; color:#7eb8ff; }
  .toolbar {
    padding:12px 24px; display:flex; gap:12px;
    align-items:center; flex-wrap:wrap;
    background:#14161f;
  }
  button {
    background:#2563eb; color:#fff; border:none;
    padding:10px 20px; border-radius:8px; cursor:pointer;
    font-size:14px; transition:background .2s;
  }
  button:hover { background:#1d4ed8; }
  button:disabled { background:#3a3d4a; cursor:not-allowed; }
  .search-box {
    flex:1; display:flex; gap:8px; min-width:280px;
  }
  .search-box input {
    flex:1; padding:10px 14px; border-radius:8px;
    border:1px solid #3a3d4a; background:#1e2130;
    color:#fff; font-size:14px; outline:none;
  }
  .search-box input:focus { border-color:#2563eb; }
  #status {
    padding:8px 24px; font-size:13px; color:#9ca3af;
    background:#14161f; border-bottom:1px solid #1e2130;
  }
  .progress-bar {
    height:4px; background:#1e2130; border-radius:2px;
    margin:6px 24px; overflow:hidden;
  }
  .progress-bar .fill {
    height:100%; background:#2563eb; width:0%;
    transition:width .3s;
  }
  #gallery {
    flex:1; overflow-y:auto; padding:16px 24px;
    display:grid;
    grid-template-columns:repeat(auto-fill, minmax(200px,1fr));
    gap:14px; align-content:start;
  }
  .card {
    background:#1a1d27; border-radius:10px; overflow:hidden;
    border:1px solid #2a2d3a; transition:transform .15s;
  }
  .card:hover { transform:translateY(-3px); }
  .card img {
    width:100%; height:160px; object-fit:cover; display:block;
  }
  .card .info {
    padding:8px 10px; font-size:12px; color:#9ca3af;
    white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
  }
  .card .score {
    color:#7eb8ff; font-weight:600;
  }
  .empty-msg {
    grid-column:1/-1; text-align:center;
    color:#555; padding:60px 0; font-size:15px;
  }
</style>
</head>
<body>

<header>
  <h1>🔍 XY Zvec - Raon Vision Search</h1>
  <span style="font-size:13px;color:#666;">
    자연어로 이미지를 검색하세요
  </span>
</header>

<div class="toolbar">
  <button id="btnDownload" onclick="startDownload()"
          style="display:none;background:#16a34a;">⬇️ 모델 다운로드</button>
  <button id="btnFolder" onclick="selectFolder()" disabled>📁 폴더 선택</button>
  <button id="btnIndex" onclick="startIndexing()" disabled>⚙️ 인덱싱 시작</button>
  <button id="btnLoadIdx" onclick="loadIndex()" disabled>📂 저장된 인덱스 로드</button>
  <div class="search-box">
    <input id="searchInput" type="text" disabled
           placeholder="자연어로 검색… 예: 바다 위 일몰, 빨간 자동차"
           onkeydown="if(event.key==='Enter')doSearch()"/>
    <button id="btnSearch" onclick="doSearch()" disabled>검색</button>
  </div>
</div>

<div id="status">모델 로딩 중… 잠시 기다려 주세요.</div>
<div class="progress-bar"><div class="fill" id="pbar"></div></div>

<div id="gallery">
  <div class="empty-msg">모델 로딩 후 폴더를 선택하고 인덱싱하세요.</div>
</div>

<script>
let selectedFolder = "";
let pollTimer = null;
let dlTimer = null;
let apiReady = false;
let modelReady = false;

// ── pywebview.api 준비 대기 ───────────────────────────────
function waitForApi(callback) {
  if (window.pywebview && window.pywebview.api) {
    apiReady = true;
    console.log("[JS] pywebview.api 준비 완료");
    callback();
  } else {
    setTimeout(function() { waitForApi(callback); }, 200);
  }
}

// ── 버튼 활성화/비활성화 ──────────────────────────────────
function setButtonsEnabled(enabled) {
  document.getElementById("btnFolder").disabled = !enabled;
  document.getElementById("btnIndex").disabled = !enabled;
  document.getElementById("btnLoadIdx").disabled = !enabled;
  document.getElementById("btnSearch").disabled = !enabled;
  document.getElementById("searchInput").disabled = !enabled;
}

// ── 다운로드 폴링 ─────────────────────────────────────────
function startDownloadPolling() {
  dlTimer = setInterval(async function() {
    try {
      const d = await pywebview.api.get_download_progress();
      console.log("[JS] download:", d.pct, "%", d.msg);
      setStatus("⬇️ " + d.msg);
      document.getElementById("pbar").style.width = d.pct + "%";

      if (!d.downloading && d.ready) {
        clearInterval(dlTimer);
        dlTimer = null;
        modelReady = true;
        setButtonsEnabled(true);
        document.getElementById("btnIndex").disabled = true;
        setStatus("✅ 모델 로드 완료! 폴더를 선택하세요.");
        document.getElementById("pbar").style.width = "100%";
      } else if (!d.downloading && !d.ready) {
        clearInterval(dlTimer);
        dlTimer = null;
        setStatus("❌ " + d.msg);
        document.getElementById("btnDownload").disabled = false;
      }
    } catch(e) {
      console.error("[JS] download poll 예외:", e);
    }
  }, 500);
}

// ── 앱 시작 ───────────────────────────────────────────────
window.addEventListener("DOMContentLoaded", function() {
  console.log("[JS] DOMContentLoaded");
  setStatus("⏳ pywebview API 준비 대기 중...");
  setButtonsEnabled(false);

  waitForApi(async function() {
    setStatus("⏳ 모델 자동 로딩 중...");
    try {
      console.log("[JS] 자동 init_model 호출");
      const res = await pywebview.api.init_model();
      console.log("[JS] init_model 응답:", JSON.stringify(res));
      if (res.ok) {
        startDownloadPolling();
      } else {
        setStatus("⚠️ " + res.msg);
        document.getElementById("btnDownload").style.display = "inline-block";
        document.getElementById("btnDownload").disabled = false;
      }
    } catch(e) {
      console.error("[JS] 자동 초기화 예외:", e);
      setStatus("❌ 초기화 오류: " + e);
      document.getElementById("btnDownload").style.display = "inline-block";
      document.getElementById("btnDownload").disabled = false;
    }
  });
});

// ── 다운로드 버튼 ─────────────────────────────────────────
async function startDownload() {
  console.log("[JS] 다운로드 버튼 클릭");
  try {
    document.getElementById("btnDownload").disabled = true;
    setButtonsEnabled(false);
    const res = await pywebview.api.init_model();
    console.log("[JS] init_model 응답:", JSON.stringify(res));
    if (res.ok) {
      startDownloadPolling();
    } else {
      setStatus("⚠️ " + res.msg);
      document.getElementById("btnDownload").disabled = false;
    }
  } catch(e) {
    console.error("[JS] startDownload 예외:", e);
    setStatus("❌ 다운로드 시작 오류: " + e);
    document.getElementById("btnDownload").disabled = false;
  }
}

// ── 폴더 선택 ─────────────────────────────────────────────
async function selectFolder() {
  console.log("[JS] selectFolder 버튼 클릭");
  try {
    if (!modelReady) { setStatus("⚠️ 모델 다운로드 후 이용하세요."); return; }
    const res = await pywebview.api.select_folder();
    console.log("[JS] select_folder 응답:", JSON.stringify(res));
    if (!res.ok) { setStatus("⚠️ " + res.msg); return; }
    selectedFolder = res.path;
    console.log("[JS] 선택 폴더:", selectedFolder);
    setStatus("📁 선택된 폴더: " + selectedFolder);
    document.getElementById("btnIndex").disabled = false;

    const scan = await pywebview.api.scan_images(selectedFolder);
    console.log("[JS] scan_images 응답:", scan.count, "개");
    setStatus("📁 " + selectedFolder + "  →  이미지 " + scan.count + "개 발견");
  } catch(e) {
    console.error("[JS] selectFolder 예외:", e);
    setStatus("❌ 폴더 선택 오류: " + e);
  }
}

// ── 인덱싱 시작 ───────────────────────────────────────────
async function startIndexing() {
  console.log("[JS] startIndexing 클릭, 폴더:", selectedFolder);
  try {
    if (!modelReady) { setStatus("⚠️ 모델 다운로드 후 이용하세요."); return; }
    if (!selectedFolder) { setStatus("⚠️ 먼저 폴더를 선택하세요."); return; }
    document.getElementById("btnIndex").disabled = true;
    await pywebview.api.start_indexing(selectedFolder);
    setStatus("⚙️ 인덱싱 시작…");
    pollTimer = setInterval(pollProgress, 500);
  } catch(e) {
    console.error("[JS] startIndexing 예외:", e);
    setStatus("❌ 인덱싱 시작 오류: " + e);
  }
}

async function pollProgress() {
  try {
    const p = await pywebview.api.get_progress();
    console.log("[JS] pollProgress:", p.current, "/", p.total);
    setStatus("⚙️ " + p.msg);
    const pct = p.total > 0 ? Math.round(p.current / p.total * 100) : 0;
    document.getElementById("pbar").style.width = pct + "%";
    if (!p.indexing) {
      clearInterval(pollTimer);
      document.getElementById("btnIndex").disabled = false;
      setStatus("✅ " + p.msg);
    }
  } catch(e) {
    console.error("[JS] pollProgress 예외:", e);
  }
}

// ── 저장된 인덱스 로드 ────────────────────────────────────
async function loadIndex() {
  console.log("[JS] loadIndex 클릭");
  try {
    if (!modelReady) { setStatus("⚠️ 모델 다운로드 후 이용하세요."); return; }
    const res = await pywebview.api.load_index();
    console.log("[JS] load_index 응답:", JSON.stringify(res));
    setStatus(res.ok ? "✅ " + res.msg : "⚠️ " + res.msg);
  } catch(e) {
    console.error("[JS] loadIndex 예외:", e);
    setStatus("❌ 인덱스 로드 오류: " + e);
  }
}

// ── 검색 ──────────────────────────────────────────────────
async function doSearch() {
  const q = document.getElementById("searchInput").value.trim();
  console.log("[JS] doSearch:", q);
  if (!q) return;
  try {
    if (!modelReady) { setStatus("⚠️ 모델 다운로드 후 이용하세요."); return; }
    setStatus("🔍 검색 중: " + q);
    const res = await pywebview.api.search(q, 20);
    console.log("[JS] search 응답:", res.ok, res.results ? res.results.length : 0, "개");
    if (!res.ok) { setStatus("❌ " + res.msg); return; }
    renderResults(res.results);
    setStatus("🔍 \"" + q + "\" → " + res.results.length + "개 결과");
  } catch(e) {
    console.error("[JS] doSearch 예외:", e);
    setStatus("❌ 검색 오류: " + e);
  }
}

function renderResults(results) {
  const g = document.getElementById("gallery");
  if (!results || results.length === 0) {
    g.innerHTML = '<div class="empty-msg">검색 결과가 없습니다.</div>';
    return;
  }
  let html = "";
  for (const r of results) {
    const score = (r.score * 100).toFixed(1);
    html += `
      <div class="card">
        <img src="local-file://${r.path.replace(/\\/g,'/')}"
             onerror="this.src='data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 width=%22200%22 height=%22160%22><rect fill=%22%231a1d27%22 width=%22200%22 height=%22160%22/><text x=%2250%25%22 y=%2250%25%22 fill=%22%23555%22 text-anchor=%22middle%22>No Image</text></svg>'"/>
        <div class="info">
          <span class="score">${score}%</span> · ${r.name}
        </div>
      </div>`;
  }
  g.innerHTML = html;
}

function setStatus(msg) {
  document.getElementById("status").textContent = msg;
}
</script>
</body>
</html>
"""


# ============================================================
#  로컬 파일 서빙 (PyWebView이 file:// 이미지를 표시할 수 있게)
# ============================================================
def local_file_handler(path: str):
    """local-file:// 스킴으로 들어온 경로를 실제 파일로 반환."""
    real = path.replace("local-file://", "")
    if os.name == "nt":
        real = real.lstrip("/")
    if os.path.isfile(real):
        return real
    return None


# ============================================================
#  메인
# ============================================================
def main():
    logger.info("[메인] PyWebView 창 생성 시작")
    api = Api()
    window = webview.create_window(
        "XY Zvec - Raon Vision Search",
        html=HTML_PAGE,
        js_api=api,
        width=1200,
        height=800,
        min_size=(900, 600),
    )
    logger.info("[메인] PyWebView 시작 (debug=True → F12 콘솔 확인 가능)")
    webview.start(debug=True)


if __name__ == "__main__":
    main()