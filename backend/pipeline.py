"""
HealthLens AI — Pipeline Orchestrator

FULL PIPELINE:
  Upload → OCR Node → Agent 1 (Extract) → Comparison Node → Agent 3 (Verify) → Agent 2 (Recommend)

Each step is isolated. Errors in one step produce graceful degradation, not crashes.
"""

import os
import time
import logging
from datetime import datetime

from models import (
    HealthLensReport, OCRResult, ExtractionResult,
    ComparisonResult, VerificationResult, RecommendationResult,
    ConfidenceLevel, MetricTrend
)
from ocr_processor import OCRProcessor
from agent1_extractor import Agent1Extractor
from agent2_recommender import Agent2Recommender
from agent3_verifier import Agent3Verifier

logger = logging.getLogger(__name__)


class HealthLensPipeline:
    def __init__(self):
        self.ocr = OCRProcessor()
        self.agent1 = Agent1Extractor()
        self.agent2 = Agent2Recommender()
        self.agent3 = Agent3Verifier()

    async def run(
        self,
        file_path: str,
        filename: str,
        file_id: str,
        previous_report: dict | None = None,
    ) -> dict:
        """
        Execute the full HealthLens pipeline.
        Returns a dict (serialisable) matching HealthLensReport schema.
        """
        start_time = time.time()
        pipeline_errors: list[str] = []
        upload_time = datetime.now().isoformat()

        report = HealthLensReport(
            file_id=file_id,
            filename=filename,
            upload_time=upload_time,
        )

        # ── Step 1: OCR ───────────────────────────────────────────────────────
        logger.info(f"[Pipeline] Step 1: OCR — {filename}")
        try:
            ocr_result: OCRResult = self.ocr.process(file_path)
            report.ocr = ocr_result
            if ocr_result.confidence == ConfidenceLevel.LOW:
                pipeline_errors.append("OCR quality is low. Results may be less accurate.")
        except Exception as e:
            logger.error(f"OCR failed: {e}")
            ocr_result = OCRResult(raw_text="", method="failed", confidence=ConfidenceLevel.LOW)
            report.ocr = ocr_result
            pipeline_errors.append(f"OCR failed: {e}")
            report.pipeline_errors = pipeline_errors
            report.pipeline_success = False
            report.processing_time_seconds = time.time() - start_time
            return report.model_dump()

        # ── Step 2: Agent 1 — Extract ─────────────────────────────────────────
        logger.info("[Pipeline] Step 2: Agent 1 Extraction")
        try:
            extraction: ExtractionResult = self.agent1.extract(
                ocr_text=ocr_result.raw_text,
                filename=filename,
                upload_date=upload_time[:10],
            )
            report.extraction = extraction
        except Exception as e:
            logger.error(f"Agent 1 failed: {e}")
            extraction = ExtractionResult(warnings=[f"Extraction failed: {e}"])
            report.extraction = extraction
            pipeline_errors.append(f"Agent 1 failed: {e}")

        # ── Step 3: Comparison Node ───────────────────────────────────────────
        logger.info("[Pipeline] Step 3: Comparison")
        try:
            comparison = self._compare(extraction, previous_report)
            report.comparison = comparison
        except Exception as e:
            logger.error(f"Comparison failed: {e}")
            comparison = ComparisonResult()
            report.comparison = comparison
            pipeline_errors.append(f"Comparison failed: {e}")

        # ── Step 4: Agent 3 — Verify ──────────────────────────────────────────
        logger.info("[Pipeline] Step 4: Agent 3 Verification")
        try:
            verification: VerificationResult = self.agent3.verify(extraction, ocr_result.raw_text)
            report.verification = verification
            # Use verified extraction for Agent 2
            safe_extraction = verification.verified_extraction or extraction
        except Exception as e:
            logger.error(f"Agent 3 failed: {e}")
            verification = VerificationResult(
                is_verified=False,
                overall_confidence=ConfidenceLevel.LOW,
                verification_notes=f"Verification failed: {e}",
                verified_extraction=extraction,
            )
            report.verification = verification
            safe_extraction = extraction
            pipeline_errors.append(f"Agent 3 failed: {e}")

        # ── Step 5: Agent 2 — Recommend ───────────────────────────────────────
        logger.info("[Pipeline] Step 5: Agent 2 Recommendation")
        try:
            recommendation: RecommendationResult = self.agent2.recommend(safe_extraction, comparison)
            report.recommendation = recommendation
        except Exception as e:
            logger.error(f"Agent 2 failed: {e}")
            report.recommendation = RecommendationResult(
                plain_summary="Could not generate summary. Please consult your doctor.",
                disclaimer="Always consult a qualified doctor.",
            )
            pipeline_errors.append(f"Agent 2 failed: {e}")

        # ── Finalise ──────────────────────────────────────────────────────────
        report.pipeline_errors = pipeline_errors
        report.pipeline_success = len(pipeline_errors) == 0
        report.processing_time_seconds = round(time.time() - start_time, 2)

        logger.info(f"[Pipeline] Complete in {report.processing_time_seconds}s — success={report.pipeline_success}")
        return report.model_dump()

    async def chat(self, report: dict, question: str) -> str:
        """Answer a follow-up question about an existing report."""
        # Reconstruct verified extraction
        try:
            verification_data = report.get("verification") or {}
            verified_extraction_data = verification_data.get("verified_extraction") or report.get("extraction")
            if verified_extraction_data:
                extraction = ExtractionResult.model_validate(verified_extraction_data)
            else:
                from models import ExtractionResult
                extraction = ExtractionResult()
        except Exception as e:
            logger.error(f"Chat context error: {e}")
            from models import ExtractionResult
            extraction = ExtractionResult()

        return self.agent2.chat(extraction, question)

    # ──────────────────────────────────────────────────────────────────────────
    # Comparison Node
    # ──────────────────────────────────────────────────────────────────────────

    def _compare(self, current: ExtractionResult, previous_report: dict | None) -> ComparisonResult:
        if not previous_report:
            return ComparisonResult(has_previous=False)

        prev_extraction_data = previous_report.get("extraction") or {}
        if not prev_extraction_data:
            return ComparisonResult(has_previous=False)

        try:
            prev_extraction = ExtractionResult.model_validate(prev_extraction_data)
        except Exception:
            return ComparisonResult(has_previous=False)

        # Map test names to values for quick lookup
        prev_tests = {t.test_name.lower(): t for t in prev_extraction.tests}
        curr_tests = {t.test_name.lower(): t for t in current.tests}

        trends: list[MetricTrend] = []
        new_tests = []
        resolved_issues = []
        persistent_issues = []

        for name, curr_test in curr_tests.items():
            if name in prev_tests:
                prev_test = prev_tests[name]
                trend = self._calc_trend(prev_test, curr_test)
                trends.append(MetricTrend(
                    test_name=curr_test.test_name,
                    previous_value=f"{prev_test.value} {prev_test.unit or ''}".strip(),
                    current_value=f"{curr_test.value} {curr_test.unit or ''}".strip(),
                    trend=trend,
                    change_description=self._trend_description(prev_test, curr_test, trend),
                ))
                # Track persistent issues
                from models import TestStatus
                if curr_test.status in (TestStatus.HIGH, TestStatus.LOW, TestStatus.ABNORMAL):
                    if prev_test.status in (TestStatus.HIGH, TestStatus.LOW, TestStatus.ABNORMAL):
                        persistent_issues.append(curr_test.test_name)
                    else:
                        pass  # New abnormality
                else:
                    if prev_test.status in (TestStatus.HIGH, TestStatus.LOW, TestStatus.ABNORMAL):
                        resolved_issues.append(curr_test.test_name)
            else:
                new_tests.append(curr_test.test_name)

        return ComparisonResult(
            has_previous=True,
            trends=trends,
            new_tests=new_tests,
            resolved_issues=resolved_issues,
            persistent_issues=persistent_issues,
        )

    def _calc_trend(self, prev, curr) -> str:
        import re
        from models import TestStatus

        prev_num = self._extract_number(prev.value)
        curr_num = self._extract_number(curr.value)

        if prev_num is None or curr_num is None:
            return "stable"  # Can't compare

        diff = curr_num - prev_num
        pct = abs(diff) / prev_num * 100 if prev_num != 0 else 0

        if pct < 5:
            return "stable"

        # For tests where lower is better (e.g. glucose, cholesterol when high)
        is_high_before = prev.status in (TestStatus.HIGH, TestStatus.ABNORMAL)
        is_normal_now = curr.status == TestStatus.NORMAL

        if diff < 0 and is_high_before and is_normal_now:
            return "improved"
        if diff > 0 and is_normal_now:
            return "worsened"
        if diff < 0:
            return "improved"
        return "worsened"

    def _extract_number(self, value: str) -> float | None:
        import re
        match = re.search(r"[\d.]+", str(value).replace(",", ""))
        try:
            return float(match.group()) if match else None
        except Exception:
            return None

    def _trend_description(self, prev, curr, trend: str) -> str:
        prev_val = f"{prev.value} {prev.unit or ''}".strip()
        curr_val = f"{curr.value} {curr.unit or ''}".strip()
        trend_word = {"improved": "improved ✅", "worsened": "worsened ⚠️", "stable": "stable", "new_finding": "new test"}.get(trend, trend)
        return f"{prev_val} → {curr_val} ({trend_word})"
