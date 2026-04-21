"""
HealthLens AI — Data Models
Strict typed models for each pipeline stage.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Literal
from datetime import datetime
from enum import Enum


# ─── Enums ────────────────────────────────────────────────────────────────────

class DocumentType(str, Enum):
    LAB_REPORT = "lab_report"
    PRESCRIPTION = "prescription"
    DISCHARGE_SUMMARY = "discharge_summary"
    RADIOLOGY = "radiology"
    PATHOLOGY = "pathology"
    OTHER = "other"

class ConfidenceLevel(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"

class TestStatus(str, Enum):
    NORMAL = "normal"
    HIGH = "high"
    LOW = "low"
    ABNORMAL = "abnormal"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


# ─── OCR Stage ────────────────────────────────────────────────────────────────

class OCRResult(BaseModel):
    raw_text: str
    method: str  # "direct_extraction" | "pdf_ocr" | "image_ocr"
    confidence: ConfidenceLevel
    confidence_score: float = 0.0
    page_count: int = 1
    warnings: List[str] = []


# ─── Agent 1 Output ───────────────────────────────────────────────────────────

class PatientInfo(BaseModel):
    name: Optional[str] = None
    age: Optional[str] = None
    gender: Optional[str] = None
    patient_id: Optional[str] = None

class TestResult(BaseModel):
    test_name: str
    value: str
    unit: Optional[str] = None
    reference_range: Optional[str] = None
    status: TestStatus = TestStatus.UNKNOWN
    confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM
    source_text: Optional[str] = None  # Exact substring from OCR for verification

class Medicine(BaseModel):
    name: str
    dosage: Optional[str] = None
    frequency: Optional[str] = None
    duration: Optional[str] = None
    confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM

class ExtractionResult(BaseModel):
    document_type: DocumentType = DocumentType.OTHER
    extraction_confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM
    patient: PatientInfo = PatientInfo()
    document_date: Optional[str] = None
    lab_name: Optional[str] = None
    doctor_name: Optional[str] = None
    tests: List[TestResult] = []
    medicines: List[Medicine] = []
    diagnoses: List[str] = []
    doctor_notes: Optional[str] = None
    overall_summary: Optional[str] = None
    critical_findings: List[str] = []
    warnings: List[str] = []


# ─── Comparison Node Output ───────────────────────────────────────────────────

class MetricTrend(BaseModel):
    test_name: str
    previous_value: str
    current_value: str
    trend: Literal["improved", "worsened", "stable", "new_finding"]
    change_description: str

class ComparisonResult(BaseModel):
    has_previous: bool = False
    trends: List[MetricTrend] = []
    new_tests: List[str] = []
    resolved_issues: List[str] = []
    persistent_issues: List[str] = []


# ─── Agent 3 Output (Verification) ────────────────────────────────────────────

class UnsupportedClaim(BaseModel):
    claim: str
    reason: str
    severity: Literal["high", "medium", "low"]

class VerificationResult(BaseModel):
    is_verified: bool
    overall_confidence: ConfidenceLevel
    unsupported_claims: List[UnsupportedClaim] = []
    low_confidence_items: List[str] = []
    safety_warnings: List[str] = []
    verified_extraction: Optional[ExtractionResult] = None
    verification_notes: str = ""


# ─── Agent 2 Output (Recommendation + Chat) ───────────────────────────────────

class SpecialistRecommendation(BaseModel):
    specialist_type: str
    reason: str
    urgency: Literal["urgent", "soon", "routine", "not_required"]

class InsightCard(BaseModel):
    title: str
    value: str
    status: TestStatus
    plain_explanation: str
    what_it_means: str

class RecommendationResult(BaseModel):
    plain_summary: str
    key_insights: List[InsightCard] = []
    specialist_recommendations: List[SpecialistRecommendation] = []
    lifestyle_notes: List[str] = []
    next_steps: List[str] = []
    disclaimer: str = (
        "This is an AI-generated analysis for informational purposes only. "
        "It is NOT a medical diagnosis. Always consult a qualified doctor before making health decisions."
    )
    chat_response: Optional[str] = None


# ─── Final Pipeline Output ────────────────────────────────────────────────────

class HealthLensReport(BaseModel):
    file_id: str
    filename: str
    upload_time: str = Field(default_factory=lambda: datetime.now().isoformat())
    
    # Pipeline stages
    ocr: Optional[OCRResult] = None
    extraction: Optional[ExtractionResult] = None
    comparison: Optional[ComparisonResult] = None
    verification: Optional[VerificationResult] = None
    recommendation: Optional[RecommendationResult] = None
    
    # Pipeline metadata
    pipeline_success: bool = True
    pipeline_errors: List[str] = []
    processing_time_seconds: float = 0.0
