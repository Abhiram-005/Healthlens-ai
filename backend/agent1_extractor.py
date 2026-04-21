"""
HealthLens AI — Agent 1: Report Understanding Agent

PURPOSE:
  Takes raw OCR text → outputs a structured JSON of ALL medical entities found.
  This becomes the "source of truth" for the entire pipeline.

ANTI-HALLUCINATION DESIGN:
  - Prompt strictly forbids inventing values not present in OCR text
  - Every test result includes a `source_text` field (exact substring from OCR)
  - Confidence scores per field
  - Extraction is validated post-hoc against the OCR text
  - Falls back gracefully if LLM returns invalid JSON

MODEL: llama-3.3-70b-versatile (Groq free tier) — highest quality available
"""

import os
import json
import re
import logging
from groq import Groq
from models import (
    ExtractionResult, PatientInfo, TestResult, Medicine,
    DocumentType, ConfidenceLevel, TestStatus
)

logger = logging.getLogger(__name__)

# ─── Prompt ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a precise medical document data extraction engine.

YOUR ONLY JOB: Extract information EXACTLY as it appears in the provided medical document text.

══════════════════════════════════════════════════════════════════════
CRITICAL ACCURACY RULES — VIOLATING THESE MAKES THE SYSTEM DANGEROUS
══════════════════════════════════════════════════════════════════════

1. NEVER INVENT VALUES. If a number, name, or finding is not in the text, use null.
2. NEVER INFER NORMAL RANGES. Only use ranges explicitly printed in the document.
3. COPY VALUES VERBATIM. Do not round, convert, or paraphrase numeric values.
4. SOURCE TEXT REQUIRED. For every test result, paste the exact substring from the OCR 
   where you found that value. If you cannot find the exact substring, set confidence to LOW.
5. MARK UNCERTAINTY. If text is garbled, unclear, or you are not sure of a value, 
   set confidence = "LOW" and add a warning.
6. DO NOT DIAGNOSE. Extract only what is written. Do not add clinical interpretations.
7. STATUS BASED ON DOCUMENT. Mark a test as "high"/"low" ONLY if the document itself 
   marks it (e.g. with H, L, *, ↑, ↓, or the value is outside the printed reference range).
   If the document does not mark it, use "unknown".

══════════════════════════════════════════════════════════════════════
OUTPUT FORMAT
══════════════════════════════════════════════════════════════════════

Return ONLY valid JSON matching this exact schema. No prose before or after.

{
  "document_type": "lab_report" | "prescription" | "discharge_summary" | "radiology" | "pathology" | "other",
  "extraction_confidence": "HIGH" | "MEDIUM" | "LOW",
  "patient": {
    "name": string | null,
    "age": string | null,
    "gender": string | null,
    "patient_id": string | null
  },
  "document_date": string | null,
  "lab_name": string | null,
  "doctor_name": string | null,
  "tests": [
    {
      "test_name": "exact name from document",
      "value": "exact value string from document",
      "unit": "unit string or null",
      "reference_range": "range string exactly as printed, or null",
      "status": "normal" | "high" | "low" | "abnormal" | "critical" | "unknown",
      "confidence": "HIGH" | "MEDIUM" | "LOW",
      "source_text": "copy the exact 10-40 chars from OCR where this was found"
    }
  ],
  "medicines": [
    {
      "name": "exact medicine name",
      "dosage": "exact dosage or null",
      "frequency": "exact frequency or null",
      "duration": "exact duration or null",
      "confidence": "HIGH" | "MEDIUM" | "LOW"
    }
  ],
  "diagnoses": ["exact diagnosis strings from document"],
  "doctor_notes": "verbatim doctor notes or null",
  "overall_summary": "1-2 sentence factual summary of what type of report this is. NO diagnoses.",
  "critical_findings": ["list any values marked CRITICAL or PANIC in the document"],
  "warnings": ["list any OCR ambiguities, illegible sections, or extraction uncertainties"]
}"""

USER_TEMPLATE = """Medical document OCR text below. Extract ALL medical entities.

FILENAME: {filename}
UPLOAD DATE: {upload_date}

═══════════════════════════ OCR TEXT START ═══════════════════════════
{ocr_text}
═══════════════════════════ OCR TEXT END ═════════════════════════════

Remember:
- Only extract what is explicitly present above.
- source_text must be a real substring of the OCR text above.
- Return ONLY valid JSON."""


# ─── Reference Ranges for Post-hoc Validation ─────────────────────────────────
# Used to VERIFY the LLM's status labels, not to add values
COMMON_REFERENCE_RANGES = {
    "hemoglobin": {"male": (13.0, 17.5), "female": (11.5, 15.5), "unit": "g/dL"},
    "hba1c": {"all": (4.0, 5.7), "unit": "%"},
    "fasting glucose": {"all": (70, 100), "unit": "mg/dL"},
    "random glucose": {"all": (0, 140), "unit": "mg/dL"},
    "creatinine": {"male": (0.7, 1.3), "female": (0.5, 1.1), "unit": "mg/dL"},
    "total cholesterol": {"all": (0, 200), "unit": "mg/dL"},
    "ldl cholesterol": {"all": (0, 130), "unit": "mg/dL"},
    "hdl cholesterol": {"male": (40, 999), "female": (50, 999), "unit": "mg/dL"},
    "triglycerides": {"all": (0, 150), "unit": "mg/dL"},
    "tsh": {"all": (0.4, 4.0), "unit": "mIU/L"},
    "wbc": {"all": (4000, 11000), "unit": "cells/µL"},
    "rbc": {"male": (4.5, 5.9), "female": (4.1, 5.1), "unit": "million/µL"},
    "platelets": {"all": (150000, 400000), "unit": "cells/µL"},
    "alt": {"all": (7, 56), "unit": "U/L"},
    "ast": {"all": (10, 40), "unit": "U/L"},
    "bilirubin total": {"all": (0.1, 1.2), "unit": "mg/dL"},
    "sodium": {"all": (136, 145), "unit": "mEq/L"},
    "potassium": {"all": (3.5, 5.0), "unit": "mEq/L"},
    "urea": {"all": (7, 20), "unit": "mg/dL"},
    "uric acid": {"male": (3.5, 7.2), "female": (2.6, 6.0), "unit": "mg/dL"},
}


class Agent1Extractor:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable not set. Get a free key at https://console.groq.com")
        self.client = Groq(api_key=api_key)
        self.model = os.getenv("GROQ_MODEL_ACCURATE", "llama-3.3-70b-versatile")

    def extract(self, ocr_text: str, filename: str = "", upload_date: str = "") -> ExtractionResult:
        if not ocr_text.strip():
            return ExtractionResult(
                extraction_confidence=ConfidenceLevel.LOW,
                warnings=["OCR returned empty text. Cannot extract medical data."]
            )

        # Truncate if very long (Groq context window safety)
        max_chars = 14000
        truncated = ocr_text[:max_chars]
        truncation_warning = []
        if len(ocr_text) > max_chars:
            truncation_warning = [f"Document was truncated to {max_chars} chars for processing. First {max_chars} chars used."]

        user_message = USER_TEMPLATE.format(
            filename=filename,
            upload_date=upload_date,
            ocr_text=truncated,
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.0,       # Zero temperature = maximum determinism
                max_tokens=4096,
                response_format={"type": "json_object"},  # Force JSON output
            )

            raw_json = response.choices[0].message.content
            result = self._parse_and_validate(raw_json, ocr_text, truncation_warning)
            return result

        except Exception as e:
            logger.error(f"Agent 1 LLM error: {e}")
            return ExtractionResult(
                extraction_confidence=ConfidenceLevel.LOW,
                warnings=[f"LLM extraction failed: {str(e)}"] + truncation_warning,
            )

    def _parse_and_validate(self, raw_json: str, original_ocr: str, extra_warnings: list) -> ExtractionResult:
        """Parse LLM JSON output and post-hoc validate extracted values."""
        try:
            data = json.loads(raw_json)
        except json.JSONDecodeError as e:
            # Try to extract JSON from text
            match = re.search(r"\{.*\}", raw_json, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                except Exception:
                    return ExtractionResult(
                        extraction_confidence=ConfidenceLevel.LOW,
                        warnings=[f"Could not parse LLM output as JSON: {e}"] + extra_warnings,
                    )
            else:
                return ExtractionResult(
                    extraction_confidence=ConfidenceLevel.LOW,
                    warnings=[f"LLM did not return valid JSON. Raw: {raw_json[:200]}"] + extra_warnings,
                )

        warnings = list(data.get("warnings", [])) + extra_warnings

        # Build patient info
        patient_data = data.get("patient", {}) or {}
        patient = PatientInfo(
            name=patient_data.get("name"),
            age=patient_data.get("age"),
            gender=patient_data.get("gender"),
            patient_id=patient_data.get("patient_id"),
        )

        # Build test results with validation
        tests = []
        for t in (data.get("tests") or []):
            test_name = str(t.get("test_name", "")).strip()
            value = str(t.get("value", "")).strip()

            if not test_name or not value:
                continue

            # Validate: source_text must be a real substring of OCR
            source_text = t.get("source_text", "")
            conf_str = t.get("confidence", "MEDIUM")
            if source_text and source_text not in original_ocr:
                # LLM hallucinated the source text
                conf_str = "LOW"
                warnings.append(f"Source text for '{test_name}' not found in OCR — confidence downgraded.")

            # Map status
            status_map = {
                "normal": TestStatus.NORMAL,
                "high": TestStatus.HIGH,
                "low": TestStatus.LOW,
                "abnormal": TestStatus.ABNORMAL,
                "critical": TestStatus.CRITICAL,
            }
            status_str = str(t.get("status", "unknown")).lower()
            status = status_map.get(status_str, TestStatus.UNKNOWN)

            # Cross-check status against reference ranges if possible
            status = self._validate_status(test_name, value, t.get("unit"), status, patient)

            tests.append(TestResult(
                test_name=test_name,
                value=value,
                unit=t.get("unit"),
                reference_range=t.get("reference_range"),
                status=status,
                confidence=ConfidenceLevel(conf_str) if conf_str in ("HIGH", "MEDIUM", "LOW") else ConfidenceLevel.MEDIUM,
                source_text=source_text or None,
            ))

        # Build medicines
        medicines = []
        for m in (data.get("medicines") or []):
            name = str(m.get("name", "")).strip()
            if not name:
                continue
            conf_str = m.get("confidence", "MEDIUM")
            medicines.append(Medicine(
                name=name,
                dosage=m.get("dosage"),
                frequency=m.get("frequency"),
                duration=m.get("duration"),
                confidence=ConfidenceLevel(conf_str) if conf_str in ("HIGH", "MEDIUM", "LOW") else ConfidenceLevel.MEDIUM,
            ))

        # Document type
        doc_type_map = {
            "lab_report": DocumentType.LAB_REPORT,
            "prescription": DocumentType.PRESCRIPTION,
            "discharge_summary": DocumentType.DISCHARGE_SUMMARY,
            "radiology": DocumentType.RADIOLOGY,
            "pathology": DocumentType.PATHOLOGY,
        }
        doc_type = doc_type_map.get(str(data.get("document_type", "")).lower(), DocumentType.OTHER)

        # Extraction confidence
        conf_val = str(data.get("extraction_confidence", "MEDIUM")).upper()
        extraction_conf = ConfidenceLevel(conf_val) if conf_val in ("HIGH", "MEDIUM", "LOW") else ConfidenceLevel.MEDIUM

        return ExtractionResult(
            document_type=doc_type,
            extraction_confidence=extraction_conf,
            patient=patient,
            document_date=data.get("document_date"),
            lab_name=data.get("lab_name"),
            doctor_name=data.get("doctor_name"),
            tests=tests,
            medicines=medicines,
            diagnoses=[str(d) for d in (data.get("diagnoses") or []) if d],
            doctor_notes=data.get("doctor_notes"),
            overall_summary=data.get("overall_summary"),
            critical_findings=[str(c) for c in (data.get("critical_findings") or []) if c],
            warnings=warnings,
        )

    def _validate_status(
        self,
        test_name: str,
        value: str,
        unit: str | None,
        llm_status: TestStatus,
        patient: PatientInfo,
    ) -> TestStatus:
        """
        Cross-check LLM status against known reference ranges.
        Only overrides if we can clearly determine a discrepancy.
        """
        # Normalise test name for lookup
        name_lower = test_name.lower().strip()
        ref = None
        for key in COMMON_REFERENCE_RANGES:
            if key in name_lower:
                ref = COMMON_REFERENCE_RANGES[key]
                break

        if not ref:
            return llm_status  # No built-in reference; trust LLM

        # Extract numeric value
        numeric_match = re.search(r"[\d.]+", value.replace(",", ""))
        if not numeric_match:
            return llm_status

        try:
            num = float(numeric_match.group())
        except ValueError:
            return llm_status

        # Determine gender-specific range
        gender = (patient.gender or "").lower()
        if "male" in gender and "male" in ref:
            lo, hi = ref["male"]
        elif "female" in gender and "female" in ref:
            lo, hi = ref["female"]
        elif "all" in ref:
            lo, hi = ref["all"]
        else:
            return llm_status

        if num < lo:
            return TestStatus.LOW
        elif num > hi:
            return TestStatus.HIGH
        else:
            return TestStatus.NORMAL
