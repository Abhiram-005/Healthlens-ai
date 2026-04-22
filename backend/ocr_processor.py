"""
HealthLens AI — OCR Processor
Multi-strategy text extraction. Windows Tesseract path auto-detected.
"""

import os
import logging
import platform
import numpy as np
from pathlib import Path
from typing import List, Tuple

import cv2
from PIL import Image, ImageEnhance, ImageFilter

# ── Tesseract ─────────────────────────────────────────────────────────────────
try:
    import pytesseract
    if platform.system() == "Windows":
        _un = os.getenv("USERNAME", "")
        _win_paths = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            rf"C:\Users\{_un}\AppData\Local\Programs\Tesseract-OCR\tesseract.exe",
            r"C:\Tesseract-OCR\tesseract.exe",
        ]
        for _p in _win_paths:
            if os.path.isfile(_p):
                pytesseract.pytesseract.tesseract_cmd = _p
                break
    pytesseract.get_tesseract_version()
    TESSERACT_AVAILABLE = True
    logging.getLogger(__name__).info("Tesseract OCR found and ready.")
except Exception as _e:
    TESSERACT_AVAILABLE = False
    logging.getLogger(__name__).warning(f"Tesseract not available: {_e}")

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

try:
    import pdf2image
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False

from models import OCRResult, ConfidenceLevel

logger = logging.getLogger(__name__)

TESS_CONFIGS = [
    "--oem 3 --psm 6 -l eng",
    "--oem 3 --psm 4 -l eng",
    "--oem 3 --psm 3 -l eng",
    "--oem 3 --psm 11 -l eng",
]
MIN_MEANINGFUL_TEXT_LEN = 80


class OCRProcessor:
    def process(self, file_path: str) -> OCRResult:
        ext = Path(file_path).suffix.lower()
        if ext == ".pdf":
            return self._process_pdf(file_path)
        return self._process_image_file(file_path)

    def _process_pdf(self, file_path: str) -> OCRResult:
        warnings: List[str] = []
        if PDFPLUMBER_AVAILABLE:
            direct_text, page_count = self._extract_pdf_direct(file_path)
            if len(direct_text.strip()) >= MIN_MEANINGFUL_TEXT_LEN:
                return OCRResult(raw_text=direct_text, method="direct_extraction",
                                 confidence=ConfidenceLevel.HIGH, confidence_score=95.0,
                                 page_count=page_count, warnings=[])
            warnings.append("Direct PDF text sparse; falling back to OCR.")
        if PDF2IMAGE_AVAILABLE and TESSERACT_AVAILABLE:
            return self._ocr_pdf_images(file_path, warnings)
        if not TESSERACT_AVAILABLE:
            warnings.append("Tesseract not installed. Install from https://github.com/UB-Mannheim/tesseract/wiki")
        return OCRResult(raw_text="", method="failed", confidence=ConfidenceLevel.LOW,
                         confidence_score=0.0, page_count=0, warnings=warnings)

    def _extract_pdf_direct(self, file_path: str) -> Tuple[str, int]:
        pages_text: List[str] = []
        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    chunks: List[str] = []
                    text = page.extract_text()
                    if text:
                        chunks.append(text.strip())
                    for table in page.extract_tables():
                        for row in table:
                            if row:
                                fmt = " | ".join((str(c).strip() if c else "") for c in row)
                                if fmt.strip(" |"):
                                    chunks.append(fmt)
                    if chunks:
                        pages_text.append("\n".join(chunks))
        except Exception as e:
            logger.warning(f"pdfplumber: {e}")
        return "\n\n--- PAGE BREAK ---\n\n".join(pages_text), len(pages_text)

    def _ocr_pdf_images(self, file_path: str, warnings: List[str]) -> OCRResult:
        page_texts: List[str] = []
        try:
            images = pdf2image.convert_from_path(file_path, dpi=300, fmt="png", thread_count=2)
            for i, pil_img in enumerate(images):
                result = self._ocr_numpy(np.array(pil_img))
                page_texts.append(result["text"])
            combined = "\n\n--- PAGE BREAK ---\n\n".join(page_texts)
            avg_conf = 75.0 if all(len(t.strip()) > 20 for t in page_texts) else 55.0
            return OCRResult(raw_text=combined, method="pdf_ocr",
                             confidence=self._conf_label(avg_conf), confidence_score=avg_conf,
                             page_count=len(images), warnings=warnings)
        except Exception as e:
            logger.error(f"pdf2image: {e}")
            return OCRResult(raw_text="", method="failed", confidence=ConfidenceLevel.LOW,
                             confidence_score=0.0, page_count=0, warnings=warnings + [str(e)])

    def _process_image_file(self, file_path: str) -> OCRResult:
        if not TESSERACT_AVAILABLE:
            return OCRResult(raw_text="", method="failed", confidence=ConfidenceLevel.LOW,
                             confidence_score=0.0, warnings=[
                                 "Tesseract OCR not installed. "
                                 "Download from https://github.com/UB-Mannheim/tesseract/wiki "
                                 "Install to C:\\Program Files\\Tesseract-OCR\\ then restart server."
                             ])
        try:
            pil_img = Image.open(file_path).convert("RGB")
            w, h = pil_img.size
            if w < 1200:
                pil_img = pil_img.resize((int(w * 1200 / w), int(h * 1200 / w)), Image.LANCZOS)
            img_np = np.array(pil_img)
        except Exception as e:
            return OCRResult(raw_text="", method="failed", confidence=ConfidenceLevel.LOW,
                             confidence_score=0.0, warnings=[f"Cannot open image: {e}"])

        result = self._ocr_numpy(img_np)
        warn = [] if result["text"].strip() else [
            "OCR produced no text. Check Tesseract installation and image quality."
        ]
        return OCRResult(raw_text=result["text"], method="image_ocr",
                         confidence=self._conf_label(result["confidence"]),
                         confidence_score=result["confidence"], page_count=1, warnings=warn)

    def _ocr_numpy(self, img_np: np.ndarray) -> dict:
        if not TESSERACT_AVAILABLE:
            return {"text": "", "confidence": 0.0}
        candidates: List[Tuple[str, float]] = []
        for fn in [self._full_preprocess, self._apply_clahe, self._pil_sharpen, self._otsu_threshold]:
            try:
                candidates += self._run_tesseract_configs(fn(img_np))
            except Exception:
                pass
        if not candidates:
            return {"text": "", "confidence": 0.0}
        best_text, best_conf = max(candidates, key=lambda x: (len(x[0].strip()) > 30) * 1000 + x[1])
        return {"text": best_text, "confidence": best_conf}

    def _run_tesseract_configs(self, img: np.ndarray) -> List[Tuple[str, float]]:
        results = []
        for config in TESS_CONFIGS:
            try:
                text = pytesseract.image_to_string(img, config=config)
                data = pytesseract.image_to_data(img, config=config, output_type=pytesseract.Output.DICT)
                confs = [int(c) for c in data.get("conf", []) if str(c).lstrip("-").isdigit() and int(c) != -1]
                conf = float(sum(confs) / len(confs)) if confs else 0.0
                results.append((text, conf))
            except Exception:
                pass
        return results

    def _to_gray(self, img):
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img.copy()

    def _upscale(self, gray, min_width=1600):
        h, w = gray.shape[:2]
        if w < min_width:
            gray = cv2.resize(gray, None, fx=min_width/w, fy=min_width/w, interpolation=cv2.INTER_CUBIC)
        return gray

    def _full_preprocess(self, img):
        gray = self._upscale(self._to_gray(img))
        gray = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10)
        return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1)))

    def _apply_clahe(self, img):
        gray = self._upscale(self._to_gray(img))
        enhanced = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

    def _pil_sharpen(self, img):
        pil = Image.fromarray(img).convert("L")
        pil = pil.resize((pil.width * 2, pil.height * 2), Image.LANCZOS)
        pil = pil.filter(ImageFilter.SHARPEN).filter(ImageFilter.SHARPEN)
        return np.array(ImageEnhance.Contrast(pil).enhance(2.0))

    def _otsu_threshold(self, img):
        gray = self._upscale(self._to_gray(img))
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

    def _conf_label(self, score: float) -> ConfidenceLevel:
        if score >= 75: return ConfidenceLevel.HIGH
        if score >= 45: return ConfidenceLevel.MEDIUM
        return ConfidenceLevel.LOW
