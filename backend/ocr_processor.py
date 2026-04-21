"""
HealthLens AI — OCR Processor (Workflow Node)
Multi-strategy document text extraction with noise reduction.

Strategy order:
  1. Direct PDF text extraction via pdfplumber (best quality for digital PDFs)
  2. PDF-to-image conversion → OCR (for scanned PDFs)
  3. Image preprocessing pipeline → OCR (for photos)

Image preprocessing pipeline:
  grayscale → upscale → denoise → deskew → adaptive threshold → sharpen → OCR
  We try multiple enhancement variants and pick the highest-confidence result.
"""

import os
import logging
import numpy as np
from pathlib import Path
from typing import List, Tuple

import cv2
from PIL import Image, ImageEnhance, ImageFilter

try:
    import pytesseract
    import platform
    if platform.system() == "Windows":
        import os as _os
        # Try common install locations on Windows
        _win_paths = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            r"C:\Users\{}\AppData\Local\Programs\Tesseract-OCR\tesseract.exe".format(_os.getenv("USERNAME", "")),
        ]
        for _p in _win_paths:
            if _os.path.exists(_p):
                pytesseract.pytesseract.tesseract_cmd = _p
                break
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

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

# ─── Tesseract Configuration Variants ─────────────────────────────────────────
# psm 3 = fully automatic (default)   psm 4 = single column
# psm 6 = single uniform block        psm 11 = sparse text
# oem 3 = LSTM + legacy
TESS_CONFIGS = [
    "--oem 3 --psm 6 -l eng",   # Best for lab reports (uniform block)
    "--oem 3 --psm 4 -l eng",   # Good for single-column documents
    "--oem 3 --psm 3 -l eng",   # Auto - fallback
    "--oem 3 --psm 11 -l eng",  # Sparse text - last resort
]

MIN_MEANINGFUL_TEXT_LEN = 80   # Below this → fall back to OCR


class OCRProcessor:
    """
    Handles all file types (PDF, JPG, PNG, TIFF, BMP, WEBP).
    Returns OCRResult with raw text and confidence metadata.
    """

    def process(self, file_path: str) -> OCRResult:
        ext = Path(file_path).suffix.lower()
        if ext == ".pdf":
            return self._process_pdf(file_path)
        else:
            return self._process_image_file(file_path)

    # ──────────────────────────────────────────────────────────────────────────
    # PDF Handling
    # ──────────────────────────────────────────────────────────────────────────

    def _process_pdf(self, file_path: str) -> OCRResult:
        warnings: List[str] = []

        # ── Strategy 1: Direct text extraction ──
        if PDFPLUMBER_AVAILABLE:
            direct_text, page_count = self._extract_pdf_direct(file_path)
            if len(direct_text.strip()) >= MIN_MEANINGFUL_TEXT_LEN:
                return OCRResult(
                    raw_text=direct_text,
                    method="direct_extraction",
                    confidence=ConfidenceLevel.HIGH,
                    confidence_score=95.0,
                    page_count=page_count,
                    warnings=[],
                )
            else:
                warnings.append("Direct PDF text extraction yielded sparse text; falling back to OCR.")

        # ── Strategy 2: PDF → images → OCR ──
        if PDF2IMAGE_AVAILABLE:
            return self._ocr_pdf_images(file_path, warnings)

        return OCRResult(
            raw_text="",
            method="failed",
            confidence=ConfidenceLevel.LOW,
            confidence_score=0.0,
            page_count=0,
            warnings=["Neither pdfplumber nor pdf2image is available."],
        )

    def _extract_pdf_direct(self, file_path: str) -> Tuple[str, int]:
        """Extract text and tables from a digital PDF."""
        pages_text: List[str] = []
        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    chunks: List[str] = []

                    # Plain text
                    text = page.extract_text()
                    if text:
                        chunks.append(text.strip())

                    # Tables → formatted rows
                    for table in page.extract_tables():
                        for row in table:
                            if row:
                                formatted = " | ".join(
                                    (str(cell).strip() if cell else "") for cell in row
                                )
                                if formatted.strip(" |"):
                                    chunks.append(formatted)

                    if chunks:
                        pages_text.append("\n".join(chunks))

        except Exception as e:
            logger.warning(f"pdfplumber error: {e}")

        return "\n\n--- PAGE BREAK ---\n\n".join(pages_text), len(pages_text)

    def _ocr_pdf_images(self, file_path: str, warnings: List[str]) -> OCRResult:
        """Convert each PDF page to an image, then OCR."""
        page_texts: List[str] = []
        try:
            images = pdf2image.convert_from_path(
                file_path, dpi=300, fmt="png", thread_count=2
            )
            for i, pil_img in enumerate(images):
                img_np = np.array(pil_img)
                result = self._ocr_numpy(img_np, f"page_{i+1}")
                page_texts.append(result["text"])

            combined = "\n\n--- PAGE BREAK ---\n\n".join(page_texts)
            avg_conf = self._avg_conf(page_texts)
            return OCRResult(
                raw_text=combined,
                method="pdf_ocr",
                confidence=self._conf_label(avg_conf),
                confidence_score=avg_conf,
                page_count=len(images),
                warnings=warnings,
            )
        except Exception as e:
            logger.error(f"pdf2image OCR error: {e}")
            return OCRResult(
                raw_text="",
                method="failed",
                confidence=ConfidenceLevel.LOW,
                confidence_score=0.0,
                page_count=0,
                warnings=warnings + [str(e)],
            )

    # ──────────────────────────────────────────────────────────────────────────
    # Image Handling
    # ──────────────────────────────────────────────────────────────────────────

    def _process_image_file(self, file_path: str) -> OCRResult:
        try:
            pil_img = Image.open(file_path).convert("RGB")
            img_np = np.array(pil_img)
        except Exception as e:
            return OCRResult(
                raw_text="",
                method="failed",
                confidence=ConfidenceLevel.LOW,
                confidence_score=0.0,
                warnings=[f"Cannot open image: {e}"],
            )

        result = self._ocr_numpy(img_np, "image")
        return OCRResult(
            raw_text=result["text"],
            method="image_ocr",
            confidence=self._conf_label(result["confidence"]),
            confidence_score=result["confidence"],
            page_count=1,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Core OCR Engine
    # ──────────────────────────────────────────────────────────────────────────

    def _ocr_numpy(self, img_np: np.ndarray, label: str = "") -> dict:
        """
        Try multiple preprocessing variants.
        Return the text + confidence of the best result.
        """
        if not TESSERACT_AVAILABLE:
            return {"text": "", "confidence": 0.0}

        candidates: List[Tuple[str, float]] = []

        # Variant 1: Full preprocessing pipeline
        preprocessed = self._full_preprocess(img_np)
        candidates += self._run_tesseract_configs(preprocessed)

        # Variant 2: CLAHE (contrast-limited adaptive histogram equalisation)
        clahe_img = self._apply_clahe(img_np)
        candidates += self._run_tesseract_configs(clahe_img)

        # Variant 3: PIL sharpening + contrast boost
        sharp_img = self._pil_sharpen(img_np)
        candidates += self._run_tesseract_configs(sharp_img)

        # Variant 4: Otsu global thresholding (good for clean scans)
        otsu_img = self._otsu_threshold(img_np)
        candidates += self._run_tesseract_configs(otsu_img)

        if not candidates:
            return {"text": "", "confidence": 0.0}

        # Pick highest-confidence result that also has substantial text
        # (avoids picking a high-confidence empty result)
        best_text, best_conf = max(
            candidates,
            key=lambda x: (len(x[0].strip()) > 30) * 1000 + x[1]
        )
        return {"text": best_text, "confidence": best_conf}

    def _run_tesseract_configs(self, img: np.ndarray) -> List[Tuple[str, float]]:
        """Run all Tesseract config variants on a preprocessed image."""
        results = []
        for config in TESS_CONFIGS:
            try:
                text = pytesseract.image_to_string(img, config=config)
                data = pytesseract.image_to_data(
                    img, config=config, output_type=pytesseract.Output.DICT
                )
                conf = self._tess_confidence(data)
                results.append((text, conf))
            except Exception:
                pass
        return results

    def _tess_confidence(self, data: dict) -> float:
        confs = [int(c) for c in data.get("conf", []) if str(c).lstrip("-").isdigit() and int(c) != -1]
        return float(sum(confs) / len(confs)) if confs else 0.0

    # ──────────────────────────────────────────────────────────────────────────
    # Image Preprocessing Variants
    # ──────────────────────────────────────────────────────────────────────────

    def _full_preprocess(self, img: np.ndarray) -> np.ndarray:
        """Main pipeline: grayscale → upscale → denoise → deskew → adaptive threshold."""
        gray = self._to_gray(img)
        gray = self._upscale(gray)
        gray = self._fast_denoise(gray)
        gray = self._deskew(gray)
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 31, 10,
        )
        # Close small gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    def _apply_clahe(self, img: np.ndarray) -> np.ndarray:
        gray = self._to_gray(img)
        gray = self._upscale(gray)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

    def _pil_sharpen(self, img: np.ndarray) -> np.ndarray:
        pil = Image.fromarray(img).convert("L")
        pil = pil.resize((pil.width * 2, pil.height * 2), Image.LANCZOS)
        pil = pil.filter(ImageFilter.SHARPEN)
        pil = pil.filter(ImageFilter.SHARPEN)
        pil = ImageEnhance.Contrast(pil).enhance(2.0)
        return np.array(pil)

    def _otsu_threshold(self, img: np.ndarray) -> np.ndarray:
        gray = self._to_gray(img)
        gray = self._upscale(gray)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

    # ──────────────────────────────────────────────────────────────────────────
    # Low-level Helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _to_gray(self, img: np.ndarray) -> np.ndarray:
        if len(img.shape) == 3:
            return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return img.copy()

    def _upscale(self, gray: np.ndarray, min_width: int = 1400) -> np.ndarray:
        h, w = gray.shape[:2]
        if w < min_width:
            scale = min_width / w
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        return gray

    def _fast_denoise(self, gray: np.ndarray) -> np.ndarray:
        return cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)

    def _deskew(self, gray: np.ndarray) -> np.ndarray:
        """Correct small rotation (±15°) caused by scanning or camera angle."""
        try:
            # Use Hough line detection to estimate skew
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)
            if lines is None:
                return gray

            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x2 != x1:
                    angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                    if -15 < angle < 15:
                        angles.append(angle)

            if not angles:
                return gray

            median_angle = float(np.median(angles))
            if abs(median_angle) < 0.3:
                return gray  # Skip near-zero rotation

            h, w = gray.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
            return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        except Exception:
            return gray

    def _conf_label(self, score: float) -> ConfidenceLevel:
        if score >= 80:
            return ConfidenceLevel.HIGH
        elif score >= 60:
            return ConfidenceLevel.MEDIUM
        return ConfidenceLevel.LOW

    def _avg_conf(self, texts: List[str]) -> float:
        # Simple proxy: more text generally means better OCR
        if not texts:
            return 0.0
        non_empty = [t for t in texts if len(t.strip()) > 20]
        return 75.0 if len(non_empty) == len(texts) else 55.0
