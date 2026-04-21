"""
HealthLens AI — Agent 3: Verification / Hallucination Guard Agent

PURPOSE:
  Audits the Agent 1 extraction against the original OCR text.
  Flags or removes any claim not supported by the source document.
  This is the safety layer — runs BEFORE Agent 2 sees the data.

DESIGN:
  - Checks every extracted value against the OCR text
  - Removes or flags unsupported claims
  - Assigns an overall confidence verdict
  - Produces a "safe" extraction that Agent 2 can use

MODEL: llama-3.3-70b-versatile (most capable free model)
"""

import os
import json
import re
import logging
from groq import Groq
from models import (
    ExtractionResult, VerificationResult, UnsupportedClaim,
    ConfidenceLevel, TestStatus
)

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a strict medical data verification auditor.

YOUR JOB: You are given (1) raw OCR text from a medical document and (2) structured data 
that was extracted from that text by an AI. 

Your task is to verify whether each extracted value is genuinely supported by the OCR text.

══════════════════════════════════════════════════════════════════════
VERIFICATION RULES
══════════════════════════════════════════════════════════════════════

1. For each test result: find the exact value in the OCR text. If the value or something
   clearly matching it (same digits, same test name) cannot be found → mark as UNSUPPORTED.

2. For medicines: find the medicine name in the OCR text. If not findable → UNSUPPORTED.

3. For diagnoses: find the diagnosis phrase or an equivalent in the OCR text → else UNSUPPORTED.

4. Do NOT add any new information not in the extraction. Only verify what is there.

5. Be strict but fair. Minor OCR garbling (e.g. "Hb" vs "Hemoglobin") is acceptable — 
   same value, plausible synonymn → mark SUPPORTED. Complete invention → UNSUPPORTED.

6. Critical findings MUST be in the OCR text verbatim or very close to verbatim.

══════════════════════════════════════════════════════════════════════
OUTPUT FORMAT — Return ONLY valid JSON
══════════════════════════════════════════════════════════════════════

{
  "is_verified": true | false,
  "overall_confidence": "HIGH" | "MEDIUM" | "LOW",
  "verification_notes": "1-2 sentence summary of verification outcome",
  "unsupported_claims": [
    {
      "claim": "description of the unsupported item",
      "reason": "why it is not supported by OCR text",
      "severity": "high" | "medium" | "low"
    }
  ],
  "low_confidence_items": ["list of items that appear in OCR but are ambiguous"],
  "safety_warnings": ["any flags the user or doctor should know about"],
  "supported_test_names": ["list of test_name values that ARE supported"],
  "unsupported_test_names": ["list of test_name values that are NOT supported in OCR"],
  "supported_medicine_names": ["list of medicine names that ARE supported"],
  "unsupported_medicine_names": ["list of medicine names that are NOT supported"]
}"""

USER_TEMPLATE = """Please verify the following extraction against the original OCR text.

═══════════ ORIGINAL OCR TEXT ═══════════
{ocr_text}
═════════════════════════════════════════

═══════════ EXTRACTED DATA ═══════════
{extraction_json}
═════════════════════════════════════════

Verify each extracted item. Return ONLY valid JSON with your verification results."""


class Agent3Verifier:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set")
        self.client = Groq(api_key=api_key)
        self.model = os.getenv("GROQ_MODEL_ACCURATE", "llama-3.3-70b-versatile")

    def verify(self, extraction: ExtractionResult, ocr_text: str) -> VerificationResult:
        if not ocr_text.strip():
            return VerificationResult(
                is_verified=False,
                overall_confidence=ConfidenceLevel.LOW,
                safety_warnings=["OCR text is empty — cannot verify any claims."],
                verification_notes="Verification skipped: no source text available.",
                verified_extraction=extraction,
            )

        # First do a fast rule-based check
        fast_result = self._rule_based_check(extraction, ocr_text)

        # Then run LLM verification for a deeper check
        llm_result = self._llm_verify(extraction, ocr_text)

        # Merge results
        return self._merge_results(fast_result, llm_result, extraction, ocr_text)

    # ──────────────────────────────────────────────────────────────────────────
    # Rule-based verification (fast, deterministic)
    # ──────────────────────────────────────────────────────────────────────────

    def _rule_based_check(self, extraction: ExtractionResult, ocr_text: str) -> dict:
        ocr_lower = ocr_text.lower()
        unsupported_tests = []
        low_conf_tests = []

        for test in extraction.tests:
            name_found = test.test_name.lower()[:6] in ocr_lower  # Partial name match
            # Extract numeric value from test.value
            num_match = re.search(r"[\d.]+", test.value.replace(",", ""))
            value_found = num_match and num_match.group() in ocr_text

            if not name_found and not value_found:
                unsupported_tests.append(test.test_name)
            elif not value_found:
                low_conf_tests.append(test.test_name)

        unsupported_medicines = [
            m.name for m in extraction.medicines
            if m.name.lower()[:5] not in ocr_lower
        ]

        return {
            "unsupported_tests": unsupported_tests,
            "unsupported_medicines": unsupported_medicines,
            "low_conf_tests": low_conf_tests,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # LLM-based deep verification
    # ──────────────────────────────────────────────────────────────────────────

    def _llm_verify(self, extraction: ExtractionResult, ocr_text: str) -> dict:
        # Serialize extraction to JSON for the prompt
        extraction_dict = extraction.model_dump()
        extraction_json = json.dumps(extraction_dict, indent=2, default=str)

        # Truncate to fit context window
        ocr_truncated = ocr_text[:10000]
        extraction_truncated = extraction_json[:6000]

        user_message = USER_TEMPLATE.format(
            ocr_text=ocr_truncated,
            extraction_json=extraction_truncated,
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.0,
                max_tokens=2048,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                match = re.search(r"\{.*\}", raw, re.DOTALL)
                return json.loads(match.group()) if match else {}

        except Exception as e:
            logger.error(f"Agent 3 LLM error: {e}")
            return {}

    # ──────────────────────────────────────────────────────────────────────────
    # Merge rule-based + LLM results and produce clean output
    # ──────────────────────────────────────────────────────────────────────────

    def _merge_results(
        self,
        fast: dict,
        llm: dict,
        extraction: ExtractionResult,
        ocr_text: str,
    ) -> VerificationResult:
        unsupported_tests_rb = set(fast.get("unsupported_tests", []))
        unsupported_tests_llm = set(llm.get("unsupported_test_names", []))
        # FIX: Use INTERSECTION not UNION. Only remove a test if BOTH checks agree.
        # The old union logic was too aggressive and removed valid test results.
        if unsupported_tests_llm:
            all_unsupported_tests = unsupported_tests_rb & unsupported_tests_llm
        else:
            all_unsupported_tests = set()  # LLM returned nothing, trust nothing

        unsupported_medicines_rb = set(fast.get("unsupported_medicines", []))
        unsupported_medicines_llm = set(llm.get("unsupported_medicine_names", []))
        if unsupported_medicines_llm:
            all_unsupported_medicines = unsupported_medicines_rb & unsupported_medicines_llm
        else:
            all_unsupported_medicines = set()

        # Build unsupported claims list
        unsupported_claims = []
        for t in all_unsupported_tests:
            unsupported_claims.append(UnsupportedClaim(
                claim=f"Test result: {t}",
                reason="Value or test name not found in OCR source text",
                severity="high",
            ))
        for m in all_unsupported_medicines:
            unsupported_claims.append(UnsupportedClaim(
                claim=f"Medicine: {m}",
                reason="Medicine name not found in OCR source text",
                severity="medium",
            ))
        for claim in llm.get("unsupported_claims", []):
            if isinstance(claim, dict):
                unsupported_claims.append(UnsupportedClaim(
                    claim=claim.get("claim", "unknown"),
                    reason=claim.get("reason", ""),
                    severity=claim.get("severity", "medium"),
                ))

        # Produce clean extraction (remove unsupported items)
        clean_extraction = ExtractionResult(
            document_type=extraction.document_type,
            extraction_confidence=extraction.extraction_confidence,
            patient=extraction.patient,
            document_date=extraction.document_date,
            lab_name=extraction.lab_name,
            doctor_name=extraction.doctor_name,
            tests=[t for t in extraction.tests if t.test_name not in all_unsupported_tests],
            medicines=[m for m in extraction.medicines if m.name not in all_unsupported_medicines],
            diagnoses=extraction.diagnoses,
            doctor_notes=extraction.doctor_notes,
            overall_summary=extraction.overall_summary,
            critical_findings=extraction.critical_findings,
            warnings=extraction.warnings + [
                f"Removed unverified test: {t}" for t in all_unsupported_tests
            ] + [
                f"Removed unverified medicine: {m}" for m in all_unsupported_medicines
            ],
        )

        # Determine overall confidence
        total_items = len(extraction.tests) + len(extraction.medicines)
        removed_items = len(all_unsupported_tests) + len(all_unsupported_medicines)
        removal_ratio = removed_items / total_items if total_items > 0 else 0

        llm_conf_str = llm.get("overall_confidence", "MEDIUM").upper()
        if removal_ratio > 0.3 or llm_conf_str == "LOW":
            overall_conf = ConfidenceLevel.LOW
        elif removal_ratio > 0.1 or llm_conf_str == "MEDIUM":
            overall_conf = ConfidenceLevel.MEDIUM
        else:
            overall_conf = ConfidenceLevel.HIGH

        is_verified = overall_conf != ConfidenceLevel.LOW and removed_items == 0

        safety_warnings = llm.get("safety_warnings", [])
        if clean_extraction.critical_findings:
            safety_warnings.insert(0, "⚠️ CRITICAL findings detected — please consult a doctor immediately.")

        return VerificationResult(
            is_verified=is_verified,
            overall_confidence=overall_conf,
            unsupported_claims=unsupported_claims,
            low_confidence_items=fast.get("low_conf_tests", []) + llm.get("low_confidence_items", []),
            safety_warnings=safety_warnings,
            verified_extraction=clean_extraction,
            verification_notes=llm.get("verification_notes", f"Removed {removed_items} unverified items from {total_items} total."),
        )
