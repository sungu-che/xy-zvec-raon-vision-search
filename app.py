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

import io
import base64
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

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".gif", ".pdf"}
PDF_EXTS = {".pdf"}

# ── 번역 모델 정의 ──────────────────────────────────────────
import locale

TRANSLATION_MODEL_FILES = [
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "source.spm",
    "special_tokens_map.json",
    "target.spm",
    "tokenizer_config.json",
    "vocab.json",
    "vocab.spm",
    "added_tokens.json",
]

TRANSLATION_LANGS = {
    "kor": {"name": "한국어",           "model": "kor-eng"},
    "fra": {"name": "Français",        "model": "fra-eng"},
    "deu": {"name": "Deutsch",          "model": "deu-eng"},
    "ita": {"name": "Italiano",         "model": "ita-eng"},
    "nld": {"name": "Nederlands",       "model": "nld-eng"},
    "rus": {"name": "Русский",          "model": "rus-eng"},
    "ara": {"name": "العربية",          "model": "ara-eng"},
    "zho": {"name": "中文",             "model": "zho-eng"},
    "ell": {"name": "Ελληνικά",         "model": "ell-eng"},
    "tur": {"name": "Türkçe",           "model": "tur-eng"},
    "spa": {"name": "Español",          "model": "spa-eng"},
    "cat": {"name": "Català",           "model": "cat-eng"},
    "eng": {"name": "English (원본)",   "model": None},
}

TRANSLATION_DOWNLOAD_URLS = {}
for _code, _info in TRANSLATION_LANGS.items():
    if _info["model"] is not None:
        _base = f"https://huggingface.co/Helsinki-NLP/opus-mt_tiny_{_info['model']}/resolve/main"
        TRANSLATION_DOWNLOAD_URLS[_info["model"]] = {
            fname: f"{_base}/{fname}" for fname in TRANSLATION_MODEL_FILES
        }

def _detect_pc_language() -> str:
    """PC 기본 언어를 ISO 639-1 코드로 반환."""
    try:
        lang_code = locale.getlocale()[0]
        if lang_code:
            short = lang_code.split("_")[0].lower()
            if short in TRANSLATION_LANGS:
                return short
    except Exception:
        pass
    if os.name == "nt":
        try:
            import ctypes
            lang_id = ctypes.windll.kernel32.GetUserDefaultUILanguage()
            _map = {
                0x0412: "kor", 0x040C: "fra", 0x0407: "deu",
                0x0410: "ita", 0x0413: "nld", 0x0419: "rus",
                0x0401: "ara", 0x0804: "zho", 0x0408: "ell",
                0x041F: "tur", 0x0C0A: "spa", 0x040A: "spa",
                0x0403: "cat", 0x0409: "eng",
            }
            if lang_id in _map:
                return _map[lang_id]
        except Exception:
            pass
    return "eng"

PC_LANGUAGE = _detect_pc_language()

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


def _detect_accelerator():
    """사용 가능한 GPU 가속기를 감지 (CUDA / ROCm / CPU)."""
    if torch.cuda.is_available():
        if hasattr(torch.version, "hip") and torch.version.hip is not None:
            return torch.device("cuda"), "ROCm (AMD GPU)"
        return torch.device("cuda"), "CUDA (NVIDIA GPU)"
    return torch.device("cpu"), "CPU"


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
    device, accel_name = _detect_accelerator()
    model = model.to(device)
    gc.collect()
    logger.info("[모델] 디바이스: %s, 가속기: %s, dtype: float16", device, accel_name)

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
        self.recent_indexed: list[dict] = []
        self.translator_model = None
        self.translator_tokenizer = None
        self.selected_lang = PC_LANGUAGE
        self.last_indexed_folder = ""
        self._index_path = os.path.join(LOCAL_APP_DATA, "zvec_index.npz")
        self._index_meta_path = os.path.join(LOCAL_APP_DATA, "zvec_index.json")
        self._index_state_path = os.path.join(LOCAL_APP_DATA, "index_state.json")
        logger.info("[Api] 초기화 완료 (PC 언어: %s)", PC_LANGUAGE)

    # ── 인덱스 상태 저장 (폴더 경로 포함) ─────────────────────
    def _save_index_state(self, folder: str):
        try:
            state = {
                "folder": folder,
                "count": len(self.zvec_db),
                "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            with open(self._index_state_path, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
            self.last_indexed_folder = folder
            logger.info("[상태] 인덱스 상태 저장: %s", state)
        except Exception as e:
            logger.error("[상태] 저장 실패: %s", e)

    # ── 인덱스 상태 복원 ──────────────────────────────────────
    def _try_auto_load_index(self):
        try:
            if not (os.path.isfile(self._index_path) and os.path.isfile(self._index_meta_path)):
                logger.info("[자동로드] 저장된 인덱스 없음")
                return {"loaded": False, "msg": "저장된 인덱스 없음"}

            self.zvec_db.load(self._index_path)

            folder = ""
            if os.path.isfile(self._index_state_path):
                with open(self._index_state_path, "r", encoding="utf-8") as f:
                    state = json.load(f)
                folder = state.get("folder", "")

            self.last_indexed_folder = folder
            self.indexed_count = len(self.zvec_db)
            self.total_images = len(self.zvec_db)
            self.progress_msg = (
                f"이전 인덱스 자동 로드 완료: {len(self.zvec_db)}개 이미지"
                f" (폴더: {folder})"
            )

            # recent_indexed 복원 (메타에서 상위 30개)
            self.recent_indexed = []
            for i in range(max(0, len(self.zvec_db) - 30), len(self.zvec_db)):
                meta = self.zvec_db.meta[i]
                p = meta.get("path", "")
                self.recent_indexed.append({
                    "path": p,
                    "name": meta.get("name", Path(p).name if p else ""),
                })
            # thumb_b64는 메모리 절약을 위해 lazy 생성 안 함 (요청 시 생성)

            logger.info("[자동로드] 완료: %d개, 폴더: %s", len(self.zvec_db), folder)
            return {"loaded": True, "count": len(self.zvec_db), "folder": folder}
        except Exception as e:
            logger.error("[자동로드] 실패: %s", e, exc_info=True)
            return {"loaded": False, "msg": str(e)}

    # ── 언어 리스트 조회 (JS용) ──────────────────────────────
    def get_language_list(self):
        langs = []
        for code, info in TRANSLATION_LANGS.items():
            langs.append({"code": code, "name": info["name"]})
        return {"ok": True, "langs": langs, "default": PC_LANGUAGE}

    # ── 번역 모델 다운로드 + 로드 ────────────────────────────
    def load_translator(self, lang_code: str):
        logger.info("[이벤트] 번역 모델 로드 요청: %s", lang_code)
        block = self._check_ready()
        if block:
            return block

        if lang_code == "eng":
            self.selected_lang = "eng"
            self.translator_model = None
            self.translator_tokenizer = None
            logger.info("[번역] 영어 선택 → 번역 불필요")
            return {"ok": True, "msg": "영어 선택됨 (번역 불필요)"}

        info = TRANSLATION_LANGS.get(lang_code)
        if not info or info["model"] is None:
            return {"ok": False, "msg": f"지원하지 않는 언어: {lang_code}"}

        model_pair = info["model"]
        local_dir = os.path.join(LOCAL_APP_DATA, "translation", model_pair)
        os.makedirs(local_dir, exist_ok=True)

        missing = []
        for fname in TRANSLATION_MODEL_FILES:
            fpath = os.path.join(local_dir, fname)
            if not os.path.isfile(fpath):
                url = TRANSLATION_DOWNLOAD_URLS[model_pair][fname]
                missing.append((fname, url, fpath))

        if missing:
            logger.info("[번역] 다운로드 필요 파일: %d개", len(missing))
            threading.Thread(
                target=self._download_translator_files,
                args=(missing, local_dir, model_pair, lang_code),
                daemon=True,
            ).start()
            return {"ok": True, "msg": f"번역 모델 다운로드 시작 ({len(missing)}개 파일)"}

        return self._load_translator_model(local_dir, lang_code, model_pair)

    def _download_translator_files(self, missing, local_dir, model_pair, lang_code):
        try:
            for fname, url, fpath in missing:
                self._download_msg = f"번역 모델 다운로드: {fname}"
                self._download_file_with_progress(url, fpath, fname)
            logger.info("[번역] 다운로드 완료")
            self._load_translator_model(local_dir, lang_code, model_pair)
        except Exception as e:
            logger.error("[번역] 다운로드 실패: %s", e, exc_info=True)
            self._download_msg = f"번역 모델 다운로드 실패: {e}"

    def _load_translator_model(self, local_dir, lang_code, model_pair):
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            logger.info("[번역] 모델 로드 중: %s", local_dir)
            self.translator_tokenizer = AutoTokenizer.from_pretrained(local_dir)
            self.translator_model = AutoModelForSeq2SeqLM.from_pretrained(local_dir)
            self.translator_model.eval()
            self.translator_model = self.translator_model.to(self.device)
            self.selected_lang = lang_code
            logger.info("[번역] 모델 로드 완료: %s", model_pair)
            return {"ok": True, "msg": f"번역 모델 로드 완료 ({model_pair})"}
        except Exception as e:
            logger.error("[번역] 모델 로드 실패: %s", e, exc_info=True)
            return {"ok": False, "msg": str(e)}

    # ── 텍스트에 해당 언어가 포함되어 있는지 감지 ────────────
    def _has_target_lang(self, text: str, lang_code: str) -> bool:
        if lang_code == "kor":
            for ch in text:
                if '\uac00' <= ch <= '\ud7a3' or '\u1100' <= ch <= '\u11ff':
                    return True
            return False
        elif lang_code == "zho":
            for ch in text:
                if '\u4e00' <= ch <= '\u9fff':
                    return True
            return False
        elif lang_code == "ara":
            for ch in text:
                if '\u0600' <= ch <= '\u06ff':
                    return True
            return False
        elif lang_code == "ell":
            for ch in text:
                if '\u0370' <= ch <= '\u03ff':
                    return True
            return False
        elif lang_code == "rus":
            for ch in text:
                if '\u0400' <= ch <= '\u04ff':
                    return True
            return False
        # 라틴 문자 기반 언어 (fra, deu, ita, nld, tur, spa, cat)
        # 영어와 구분이 어려우므로 번역 모델이 있으면 항상 번역 시도
        return True

    # ── 번역 실행 ─────────────────────────────────────────────
    def _translate_to_english(self, text: str) -> str:
        if self.translator_model is None or self.translator_tokenizer is None:
            return text
        try:
            inputs = self.translator_tokenizer(
                text, return_tensors="pt", truncation=True, max_length=128
            ).to(self.device)
            with torch.inference_mode():
                outputs = self.translator_model.generate(
                    **inputs, max_length=128, num_beams=4
                )
            result = self.translator_tokenizer.decode(
                outputs[0], skip_special_tokens=True
            ).strip()
            return result
        except Exception as e:
            logger.warning("[번역] 실패: %s", e)
            return text

    # ── 앱 시작 시 번역 모델 자동 로드 ────────────────────────
    def _auto_load_translator(self, lang_code: str):
        """앱 시작 시 PC 기본 언어 번역 모델 자동 로드."""
        try:
            info = TRANSLATION_LANGS.get(lang_code)
            if not info or info["model"] is None:
                return
            model_pair = info["model"]
            local_dir = os.path.join(LOCAL_APP_DATA, "translation", model_pair)
            os.makedirs(local_dir, exist_ok=True)

            missing = []
            for fname in TRANSLATION_MODEL_FILES:
                fpath = os.path.join(local_dir, fname)
                if not os.path.isfile(fpath):
                    url = TRANSLATION_DOWNLOAD_URLS[model_pair][fname]
                    missing.append((fname, url, fpath))

            if missing:
                logger.info("[번역] 자동 다운로드: %d개 파일", len(missing))
                for fname, url, fpath in missing:
                    self._download_msg = f"번역 모델 다운로드: {fname}"
                    self._download_file_with_progress(url, fpath, fname)

            self._load_translator_model(local_dir, lang_code, model_pair)
        except Exception as e:
            logger.error("[번역] 자동 로드 실패: %s", e, exc_info=True)

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
            self._download_pct = 90.0
            self.model, self.processor, self.device = load_raon_model(model_dir)
            logger.info("[모델] 로드 완료, 디바이스: %s", self.device)

            # 모델 로드 후 이전 인덱스 자동 복원
            self._download_msg = "이전 인덱스 확인 중..."
            self._download_pct = 93.0
            auto = self._try_auto_load_index()

            # PC 기본 언어 번역 모델 자동 로드
            if PC_LANGUAGE != "eng":
                self._download_msg = f"번역 모델 로드 중 ({PC_LANGUAGE})..."
                self._download_pct = 96.0
                self._auto_load_translator(PC_LANGUAGE)

            if auto.get("loaded"):
                self._download_msg = (
                    f"모델 로드 완료! 이전 인덱스 {auto['count']}개 복원됨"
                )
            else:
                self._download_msg = "모델 로드 완료!"

            self._model_ready = True
            self._download_pct = 100.0
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
                        if Path(p).suffix.lower() in PDF_EXTS:
                            pdf_imgs = self._pdf_to_images(p)
                            for page_idx, pi in enumerate(pdf_imgs):
                                pil_imgs.append(pi)
                                valid_paths.append((p, page_idx + 1))
                        else:
                            img = Image.open(p)
                            if img.mode in ("RGBA", "LA", "P"):
                                img = img.convert("RGBA")
                                bg = Image.new("RGB", img.size, (255, 255, 255))
                                bg.paste(img, mask=img.split()[-1])
                                img = bg
                            else:
                                img = img.convert("RGB")
                            pil_imgs.append(img)
                            valid_paths.append((p, None))
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

                feats_np = feats.cpu().numpy()
                for i, (p, page_num) in enumerate(valid_paths):
                    vec_id = f"{p}#p{page_num}" if page_num else p
                    display_name = f"{Path(p).name} (p.{page_num})" if page_num else Path(p).name
                    self.zvec_db.add(
                        vec_id=vec_id,
                        vector=feats_np[i],
                        metadata={"path": p, "name": display_name, "page": page_num},
                    )
                    self.indexed_count += 1
                    self.recent_indexed.append(
                        {"path": p, "name": display_name, "thumb_b64": self._make_thumb_b64(p)}
                    )
                    if len(self.recent_indexed) > 30:
                        self.recent_indexed = self.recent_indexed[-30:]

                self.progress_msg = (
                    f"{self.indexed_count}/{self.total_images} 인덱싱 완료"
                )
                logger.info("[인덱싱] %s", self.progress_msg)

            self.zvec_db.save(self._index_path)
            self._save_index_state(folder_path)
            self.progress_msg = (
                f"인덱싱 완료! 총 {self.indexed_count}개 이미지. "
                f"저장: {self._index_path}"
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
        # thumb_b64가 없는 항목은 lazy 생성
        recent_out = []
        for r in reversed(self.recent_indexed):
            item = dict(r)
            if not item.get("thumb_b64") and item.get("path"):
                item["thumb_b64"] = self._make_thumb_b64(item["path"])
                r["thumb_b64"] = item["thumb_b64"]  # 캐시
            recent_out.append(item)
        return {
            "indexing": self._indexing,
            "current": self.indexed_count,
            "total": self.total_images,
            "msg": self.progress_msg,
            "recent": recent_out,
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
            translated = query
            if self.selected_lang != "eng" and self._has_target_lang(query, self.selected_lang):
                translated = self._translate_to_english(query)
                logger.info("[번역] '%s' → '%s'", query, translated)

            inputs = self.processor(text=[translated])
            input_ids = inputs["input_ids"].to(self.device)
            logger.debug("[검색] input_ids shape: %s", input_ids.shape)
            with torch.inference_mode():
                text_feat = self.model.encode_text(input_ids)  # [1, 1152]
            text_np = text_feat.cpu().numpy().squeeze(0)
            logger.debug("[검색] text_feat shape: %s", text_np.shape)
            results = self.zvec_db.search(text_np, top_k=top_k)
            for r in results:
                r["thumb_b64"] = self._make_thumb_b64(r.get("path", ""))
            logger.info("[검색] 결과 %d개", len(results))
            return {"ok": True, "results": results, "translated": translated}
        except Exception as e:
            logger.error("[검색 오류] %s", e, exc_info=True)
            return {"ok": False, "msg": str(e)}

    # ── PDF → 이미지 변환 ────────────────────────────────────
    def _pdf_to_images(self, pdf_path: str, max_pages: int = 10, dpi: int = 150) -> list:
        """PDF 각 페이지를 PIL Image 리스트로 변환."""
        try:
            import fitz  # pymupdf
            doc = fitz.open(pdf_path)
            images = []
            page_count = min(len(doc), max_pages)
            for i in range(page_count):
                page = doc[i]
                pix = page.get_pixmap(dpi=dpi)
                img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                images.append(img)
            doc.close()
            logger.info("[PDF] %s → %d페이지 변환", Path(pdf_path).name, page_count)
            return images
        except Exception as e:
            logger.warning("[PDF] 변환 실패: %s (%s)", pdf_path, e)
            return []

    # ── 썸네일 base64 생성 (크로스 플랫폼) ────────────────────
    def _make_thumb_b64(self, path: str, size: int = 220) -> str:
        try:
            if Path(path).suffix.lower() in PDF_EXTS:
                imgs = self._pdf_to_images(path, max_pages=1, dpi=100)
                if not imgs:
                    return ""
                img = imgs[0]
            else:
                img = Image.open(path)
            img.thumbnail((size, size), Image.LANCZOS)
            if img.mode != "RGB":
                img = img.convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=80)
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            return f"data:image/jpeg;base64,{b64}"
        except Exception as e:
            logger.warning("[썸네일] 생성 실패: %s (%s)", path, e)
            return ""

    # ── 인덱스 상태 조회 (JS용) ───────────────────────────────
    def get_index_state(self):
        return {
            "has_index": len(self.zvec_db) > 0,
            "count": len(self.zvec_db),
            "folder": self.last_indexed_folder,
            "index_path": self._index_path,
        }

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
  .card img {
    cursor:zoom-in;
  }
  #imgModal {
    display:none;
    position:fixed;
    top:0; left:0;
    width:100vw; height:100vh;
    background:rgba(0,0,0,0.85);
    z-index:9999;
    align-items:center;
    justify-content:center;
    cursor:pointer;
  }
  #imgModalInner {
    position:relative;
    max-width:90vw;
    max-height:90vh;
    text-align:center;
  }
  #imgModalClose {
    position:absolute;
    top:-44px; right:0;
    background:#ef4444;
    color:#fff;
    border:none;
    border-radius:8px;
    padding:8px 18px;
    cursor:pointer;
    font-size:14px;
    z-index:10000;
  }
  #imgModalClose:hover {
    background:#dc2626;
  }
  #imgModalImg {
    max-width:90vw;
    max-height:82vh;
    object-fit:contain;
    border-radius:8px;
    display:block;
    margin:0 auto;
  }
  #imgModalInfo {
    color:#e0e0e0;
    padding:10px 0 4px 0;
    font-size:13px;
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
  <select id="langSelect" onchange="changeLanguage()" disabled
          style="padding:10px 14px;border-radius:8px;border:1px solid #3a3d4a;
                 background:#1e2130;color:#fff;font-size:14px;outline:none;
                 cursor:pointer;">
  </select>
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

<!-- 이미지 팝업 모달 -->
<div id="imgModal" onclick="closeImageModal()">
  <div id="imgModalInner" onclick="event.stopPropagation()">
    <button id="imgModalClose" onclick="closeImageModal()">✕ 닫기</button>
    <img id="imgModalImg" src=""/>
    <div id="imgModalInfo"></div>
  </div>
</div>

<script>
let selectedFolder = "";
let pollTimer = null;
let dlTimer = null;
let apiReady = false;
let modelReady = false;

// ── pywebview.api 준비 대기 ───────────────────────────────
function waitForApi(callback) {
  if (window.pywebview && window.pywebview.api && typeof window.pywebview.api.init_model === 'function') {
    apiReady = true;
    console.log("[JS] pywebview.api 준비 완료 (메서드 확인됨)");
    callback();
  } else {
    setTimeout(function() { waitForApi(callback); }, 300);
  }
}

// ── 버튼 활성화/비활성화 ──────────────────────────────────
function setButtonsEnabled(enabled) {
  document.getElementById("btnFolder").disabled = !enabled;
  document.getElementById("btnIndex").disabled = !enabled;
  document.getElementById("btnLoadIdx").disabled = !enabled;
  document.getElementById("btnSearch").disabled = !enabled;
  document.getElementById("searchInput").disabled = !enabled;
  document.getElementById("langSelect").disabled = !enabled;
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
        document.getElementById("pbar").style.width = "100%";

        // 이전 인덱스 상태 확인
        try {
          const st = await pywebview.api.get_index_state();
          console.log("[JS] index_state:", JSON.stringify(st));
          if (st.has_index && st.count > 0) {
            selectedFolder = st.folder;
            setStatus("✅ 모델 로드 완료! 이전 인덱스 " + st.count + "개 복원됨. 바로 검색 가능합니다.");
            document.getElementById("btnIndex").disabled = false;
            // 복원된 recent 표시
            const p = await pywebview.api.get_progress();
            if (p.recent && p.recent.length > 0) {
              renderRecent(p.recent);
            }
          } else {
            setStatus("✅ 모델 로드 완료! 폴더를 선택하세요.");
            document.getElementById("btnIndex").disabled = true;
          }
        } catch(e) {
          console.error("[JS] index_state 조회 예외:", e);
          setStatus("✅ 모델 로드 완료! 폴더를 선택하세요.");
          document.getElementById("btnIndex").disabled = true;
        }
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

    // 언어 선택 드롭다운 초기화
    try {
      const langRes = await pywebview.api.get_language_list();
      console.log("[JS] 언어 리스트:", JSON.stringify(langRes));
      if (langRes.ok) {
        const sel = document.getElementById("langSelect");
        sel.innerHTML = "";
        for (const l of langRes.langs) {
          const opt = document.createElement("option");
          opt.value = l.code;
          opt.textContent = l.name;
          if (l.code === langRes.default) {
            opt.selected = true;
          }
          sel.appendChild(opt);
        }
      }
    } catch(e) {
      console.error("[JS] 언어 리스트 로드 예외:", e);
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

    if (p.recent && p.recent.length > 0) {
      renderRecent(p.recent);
    }

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
    if (res.results.length === 0) {
      const p = await pywebview.api.get_progress();
      if (p.recent && p.recent.length > 0) {
        setStatus("🔍 \"" + q + "\" → 검색 결과 없음. 최근 인덱싱된 이미지를 표시합니다.");
        renderRecent(p.recent);
      } else {
        renderResults([]);
        setStatus("🔍 \"" + q + "\" → 검색 결과 없음");
      }
      return;
    }
    renderResults(res.results);
    const transInfo = res.translated && res.translated !== q
      ? " (번역: " + res.translated + ")"
      : "";
    setStatus("🔍 \"" + q + "\"" + transInfo + " → " + res.results.length + "개 결과");
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
    const imgSrc = r.thumb_b64 || "";
    const safeName = (r.name || "").replace(/'/g, "\\'");
    html += `
      <div class="card">
        <img src="${imgSrc}"
             onclick="showImageModal(this.src, '${safeName}', '${score}%')"
             onerror="this.style.display='none'"/>
        <div class="info">
          <span class="score">${score}%</span> · ${r.name}
        </div>
      </div>`;
  }
  g.innerHTML = html;
}

function renderRecent(items) {
  const g = document.getElementById("gallery");
  if (!items || items.length === 0) return;
  let html = '<div style="grid-column:1/-1;color:#7eb8ff;font-size:13px;padding:4px 0;">📌 인덱싱 완료된 이미지 (최신순)</div>';
  for (const r of items) {
    const imgSrc = r.thumb_b64 || "";
    const safeName = (r.name || "").replace(/'/g, "\\'");
    html += `
      <div class="card">
        <img src="${imgSrc}"
             onclick="showImageModal(this.src, '${safeName}', '')"
             onerror="this.style.display='none'"/>
        <div class="info">✅ ${r.name}</div>
      </div>`;
  }
  g.innerHTML = html;
}

async function changeLanguage() {
  const sel = document.getElementById("langSelect");
  const langCode = sel.value;
  console.log("[JS] 언어 변경:", langCode);
  try {
    setStatus("🌐 번역 모델 로딩 중: " + sel.options[sel.selectedIndex].text);
    const res = await pywebview.api.load_translator(langCode);
    console.log("[JS] load_translator 응답:", JSON.stringify(res));
    if (res.ok) {
      setStatus("✅ " + res.msg);
    } else {
      setStatus("⚠️ " + res.msg);
    }
  } catch(e) {
    console.error("[JS] changeLanguage 예외:", e);
    setStatus("❌ 언어 변경 오류: " + e);
  }
}

// ── 이미지 팝업 모달 ──────────────────────────────────────
function showImageModal(src, name, score) {
  if (!src) return;
  const modal = document.getElementById("imgModal");
  const modalImg = document.getElementById("imgModalImg");
  const modalInfo = document.getElementById("imgModalInfo");
  modalImg.src = src;
  modalInfo.textContent = name + (score ? " · " + score : "");
  modal.style.display = "flex";
  document.body.style.overflow = "hidden";
}

function closeImageModal() {
  const modal = document.getElementById("imgModal");
  modal.style.display = "none";
  document.getElementById("imgModalImg").src = "";
  document.body.style.overflow = "";
}

document.addEventListener("keydown", function(e) {
  if (e.key === "Escape") {
    closeImageModal();
  }
});

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
    """local-file:// 스킴으로 들어온 경로를 실제 파일로 반환.
    Windows / macOS / Linux 모두 지원.
    """
    import urllib.parse

    real = path.replace("local-file://", "")
    real = urllib.parse.unquote(real)

    if os.name == "nt":
        # Windows: C:/Users/... 또는 /C:/Users/...
        real = real.lstrip("/")
        if len(real) >= 2 and real[1] == ":":
            pass  # C:/Users/... 형태 유지
    else:
        # macOS/Linux: /Users/... 또는 /home/...
        if not real.startswith("/"):
            real = "/" + real

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
        "Raon Vision Search",
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