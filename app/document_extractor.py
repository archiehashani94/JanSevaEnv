"""
document_extractor.py
Extracts structured fields from uploaded welfare/pension documents
(Aadhaar, bank passbook, job card, ration card, income/land certificates)
and maps them to question answers for the JanSevaEnv investigation flow.
"""

import re
import io
import logging
from typing import Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ─── Extracted document data ──────────────────────────────────────────────────

@dataclass
class DocumentFields:
    doc_type: str                        # declared by user
    raw_text: str = ""
    aadhaar_number: Optional[str] = None
    name: Optional[str] = None
    dob: Optional[str] = None
    gender: Optional[str] = None
    bank_account: Optional[str] = None
    ifsc_code: Optional[str] = None
    bank_name: Optional[str] = None
    last_transaction_date: Optional[str] = None
    last_transaction_amount: Optional[str] = None
    registration_number: Optional[str] = None   # scheme / beneficiary ID
    job_card_number: Optional[str] = None
    ration_card_number: Optional[str] = None
    fps_code: Optional[str] = None
    annual_income: Optional[str] = None
    land_survey_number: Optional[str] = None
    land_area: Optional[str] = None
    scheme_name: Optional[str] = None
    dbt_status: Optional[str] = None            # "enabled" / "disabled"
    kyc_status: Optional[str] = None            # "completed" / "pending"
    mobile_number: Optional[str] = None
    ekyc_done: Optional[bool] = None
    errors: list = field(default_factory=list)


# ─── Text extraction ───────────────────────────────────────────────────────────

def extract_text_from_bytes(file_bytes: bytes, filename: str) -> str:
    """Extract raw text from PDF, image, or plain-text file bytes."""
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""

    if ext == "pdf":
        return _extract_pdf_text(file_bytes)
    elif ext in ("png", "jpg", "jpeg", "webp", "bmp", "tiff", "tif"):
        return _extract_image_text(file_bytes)
    elif ext in ("txt", "text"):
        # plain text (for testing / copy-pasted content)
        return file_bytes.decode("utf-8", errors="replace")
    else:
        # Try PDF first, then image, then plain text
        try:
            text = _extract_pdf_text(file_bytes)
            if text.strip():
                return text
        except Exception:
            pass
        try:
            return _extract_image_text(file_bytes)
        except Exception:
            pass
        # Last resort: try UTF-8 decode
        try:
            return file_bytes.decode("utf-8", errors="replace")
        except Exception:
            raise ValueError("Could not extract text from file. Upload a PDF, image (PNG/JPG), or text file.")


def _extract_pdf_text(file_bytes: bytes) -> str:
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            pages = []
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages.append(text)
            return "\n".join(pages)
    except Exception as e:
        logger.warning(f"pdfplumber failed: {e}, trying PyMuPDF")
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            return "\n".join(page.get_text() for page in doc)
        except Exception as e2:
            raise ValueError(f"Could not extract text from PDF: {e2}")


def _extract_image_text(file_bytes: bytes) -> str:
    try:
        import pytesseract
        from PIL import Image
        img = Image.open(io.BytesIO(file_bytes))
        # Try common Tesseract locations on Windows
        import os, shutil
        tess_paths = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            shutil.which("tesseract"),
        ]
        for p in tess_paths:
            if p and os.path.exists(p):
                pytesseract.pytesseract.tesseract_cmd = p
                break
        text = pytesseract.image_to_string(img, lang="eng")
        return text
    except Exception as e:
        raise ValueError(
            f"Image OCR failed: {e}. "
            "Install Tesseract-OCR from https://github.com/UB-Mannheim/tesseract/wiki"
        )


# ─── Field parsing ─────────────────────────────────────────────────────────────

def parse_fields(text: str, doc_type: str) -> DocumentFields:
    """Run all regex patterns on extracted text and populate DocumentFields."""
    f = DocumentFields(doc_type=doc_type, raw_text=text)
    t = text  # preserve original case for some patterns

    # ── Aadhaar number (12 digits, often grouped as 4-4-4 or 4444 44444 4444)
    m = re.search(r"\b(\d{4}[\s\-]?\d{4}[\s\-]?\d{4})\b", t)
    if m:
        f.aadhaar_number = re.sub(r"[\s\-]", "", m.group(1))

    # ── Name (after "Name:" / "नाम:" / beneficiary labels)
    m = re.search(
        r"(?:name|beneficiary name|applicant name|नाम)\s*[:\-]\s*([A-Z][A-Za-z\s\.]{2,40})",
        t, re.IGNORECASE
    )
    if m:
        f.name = m.group(1).strip()

    # ── Date of birth
    m = re.search(
        r"(?:dob|date of birth|d\.o\.b|जन्म तिथि)\s*[:\-]?\s*"
        r"(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}|\d{2,4}[\/\-\.]\d{1,2}[\/\-\.]\d{1,2})",
        t, re.IGNORECASE
    )
    if m:
        f.dob = m.group(1).strip()

    # ── Gender
    m = re.search(r"\b(male|female|transgender)\b", t, re.IGNORECASE)
    if m:
        f.gender = m.group(1).title()

    # ── Bank account number (9–18 digits, often after "A/C", "Account No")
    m = re.search(
        r"(?:account\s*(?:no|number|num)|a\/c\s*(?:no|number|num)?)\s*[:\.\-]?\s*(\d{9,18})",
        t, re.IGNORECASE
    )
    if m:
        f.bank_account = m.group(1).strip()

    # ── IFSC code (11 chars: 4 alpha + 0 + 6 alphanumeric)
    m = re.search(r"\b([A-Z]{4}0[A-Z0-9]{6})\b", t)
    if m:
        f.ifsc_code = m.group(1)

    # ── Bank name (common Indian banks)
    banks = [
        "State Bank of India", "SBI", "Punjab National Bank", "PNB",
        "Bank of Baroda", "BOB", "Canara Bank", "Union Bank", "UCO Bank",
        "Bank of India", "Central Bank", "Indian Bank", "Syndicate Bank",
        "Allahabad Bank", "Indian Overseas Bank", "HDFC Bank", "ICICI Bank",
        "Axis Bank", "Kotak Mahindra", "YES Bank", "IDBI Bank",
        "Gramin Bank", "Grameen Bank", "Cooperative Bank",
    ]
    for bank in banks:
        if re.search(re.escape(bank), t, re.IGNORECASE):
            f.bank_name = bank
            break

    # ── Last transaction date
    m = re.search(
        r"(?:last\s+transaction|last\s+credit|last\s+debit|date)\s*[:\-]?\s*"
        r"(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})",
        t, re.IGNORECASE
    )
    if m:
        f.last_transaction_date = m.group(1)

    # ── Last transaction amount
    m = re.search(
        r"(?:amount|cr|dr|credit|debit)\s*[:\-]?\s*(?:rs\.?|₹|inr)?\s*([\d,]+(?:\.\d{2})?)",
        t, re.IGNORECASE
    )
    if m:
        f.last_transaction_amount = "₹" + m.group(1).replace(",", "")

    # ── Registration / beneficiary ID
    m = re.search(
        r"(?:registration\s*(?:no|number|id)|beneficiary\s*(?:id|code)|farmer\s*id|ref\s*(?:no|id))\s*[:\-]?\s*([A-Z0-9\/\-]{6,20})",
        t, re.IGNORECASE
    )
    if m:
        f.registration_number = m.group(1).strip()

    # ── Job card number (MGNREGA format: STATE/DISTRICT/.../JCNO)
    m = re.search(
        r"\b([A-Z]{2}[\-\/]\d{2,3}[\-\/]\d{3,6}[\-\/]\d{3,8})\b",
        t
    )
    if m:
        f.job_card_number = m.group(1)

    # ── Ration card number
    m = re.search(
        r"(?:ration\s*card\s*(?:no|number)|rc\s*(?:no|number))\s*[:\-]?\s*([A-Z0-9]{6,16})",
        t, re.IGNORECASE
    )
    if m:
        f.ration_card_number = m.group(1)

    # ── Fair Price Shop (FPS) code
    m = re.search(
        r"(?:fps|fair\s*price\s*shop)\s*(?:code|id|no)?\s*[:\-]?\s*([A-Z0-9]{4,12})",
        t, re.IGNORECASE
    )
    if m:
        f.fps_code = m.group(1)

    # ── Annual income
    m = re.search(
        r"(?:annual\s*income|yearly\s*income|income)\s*[:\-]?\s*(?:rs\.?|₹|inr)?\s*([\d,]+)",
        t, re.IGNORECASE
    )
    if m:
        raw = int(m.group(1).replace(",", ""))
        f.annual_income = str(raw)

    # ── Land survey / khasra number
    m = re.search(
        r"(?:survey\s*(?:no|number)|khasra\s*(?:no|number)|plot\s*(?:no|number))\s*[:\-]?\s*([A-Z0-9\/\-]{2,15})",
        t, re.IGNORECASE
    )
    if m:
        f.land_survey_number = m.group(1)

    # ── Land area
    m = re.search(
        r"([\d\.]+)\s*(?:hectare|bigha|acre|sq\s*ft|dismil)",
        t, re.IGNORECASE
    )
    if m:
        f.land_area = m.group(0).strip()

    # ── Mobile number
    m = re.search(r"\b([6-9]\d{9})\b", t)
    if m:
        f.mobile_number = m.group(1)

    # ── DBT status
    if re.search(r"dbt\s*(?:enabled|active|linked|yes)", t, re.IGNORECASE):
        f.dbt_status = "enabled"
    elif re.search(r"dbt\s*(?:disabled|inactive|not\s*linked|no)", t, re.IGNORECASE):
        f.dbt_status = "disabled"

    # ── KYC / eKYC
    if re.search(r"(?:ekyc|kyc)\s*(?:done|completed|verified|success)", t, re.IGNORECASE):
        f.kyc_status = "completed"
        f.ekyc_done = True
    elif re.search(r"(?:ekyc|kyc)\s*(?:pending|not\s*done|failed|incomplete)", t, re.IGNORECASE):
        f.kyc_status = "pending"
        f.ekyc_done = False

    # ── Scheme detection from text
    scheme_map = {
        "PM-KISAN": ["pm-kisan", "pm kisan", "pmkisan", "kisan samman"],
        "OAP": ["old age pension", "oap", "jeevan pramaan", "life certificate", "vriddhawastha"],
        "WAP": ["widow pension", "widow assistance", "wap", "vidhwa pension"],
        "DAP": ["disability pension", "dap", "divyang", "viklaang"],
        "MGNREGA": ["mgnrega", "mnrega", "nrega", "job card", "muster roll", "fto"],
        "NFSA-PDS": ["ration card", "nfsa", "pds", "fair price shop", "fps", "ration"],
    }
    for scheme, keywords in scheme_map.items():
        if any(k in t.lower() for k in keywords):
            f.scheme_name = scheme
            break

    return f


# ─── Field → Question answer mapping ──────────────────────────────────────────

def map_to_question_answers(fields: DocumentFields, scheme: Optional[str] = None) -> dict:
    """
    Returns dict:
    {
        "Q01": {
            "answer": "Yes",
            "confidence": 0.9,
            "source": "DBT status in document",
            "field": "dbt_status"
        },
        ...
    }
    Only questions that can be answered with reasonable confidence are included.
    """
    answers = {}
    detected_scheme = scheme or fields.scheme_name

    # Q01 — Is bank account linked for DBT?
    if fields.dbt_status == "enabled":
        answers["Q01"] = {
            "answer": "Yes",
            "confidence": 0.95,
            "source": "DBT status found as 'enabled' in document",
            "field": "dbt_status",
        }
    elif fields.dbt_status == "disabled":
        answers["Q01"] = {
            "answer": "No",
            "confidence": 0.95,
            "source": "DBT status found as 'disabled/not linked' in document",
            "field": "dbt_status",
        }
    elif fields.bank_account and fields.ifsc_code:
        # Bank details present → probably linked
        answers["Q01"] = {
            "answer": "Yes",
            "confidence": 0.7,
            "source": "Bank account and IFSC found in document (assumed linked)",
            "field": "bank_account",
        }

    # Q02 — Bank account number and IFSC correct?
    if fields.bank_account and fields.ifsc_code:
        answers["Q02"] = {
            "answer": "Yes",
            "confidence": 0.85,
            "source": f"Account: {fields.bank_account}, IFSC: {fields.ifsc_code} extracted from document",
            "field": "bank_account",
        }

    # Q03 — Bank account active?
    if fields.last_transaction_date:
        answers["Q03"] = {
            "answer": "Yes",
            "confidence": 0.8,
            "source": f"Last transaction on {fields.last_transaction_date} — account appears active",
            "field": "last_transaction_date",
        }

    # Q06 — Aadhaar linked/seeded?
    if fields.aadhaar_number:
        answers["Q06"] = {
            "answer": "Yes",
            "confidence": 0.8,
            "source": f"Aadhaar number {fields.aadhaar_number[:4]}XXXXXXXX found in document",
            "field": "aadhaar_number",
        }

    # Q07 — Name and DOB match?
    if fields.name and fields.dob:
        answers["Q07"] = {
            "answer": "Yes",
            "confidence": 0.75,
            "source": f"Name '{fields.name}' and DOB '{fields.dob}' present in document",
            "field": "name",
        }

    # Q08 — Aadhaar active/valid?
    if fields.aadhaar_number:
        answers["Q08"] = {
            "answer": "Yes",
            "confidence": 0.7,
            "source": "Aadhaar number present in official document (assumed active)",
            "field": "aadhaar_number",
        }

    # Q09 — eKYC completed?
    if fields.ekyc_done is True:
        answers["Q09"] = {
            "answer": "Yes",
            "confidence": 0.95,
            "source": "eKYC/KYC status explicitly marked as completed in document",
            "field": "kyc_status",
        }
    elif fields.ekyc_done is False:
        answers["Q09"] = {
            "answer": "No",
            "confidence": 0.95,
            "source": "eKYC/KYC status explicitly marked as pending/incomplete in document",
            "field": "kyc_status",
        }

    # Q11 — Income within eligibility limit?
    if fields.annual_income:
        income = int(fields.annual_income)
        # PM-KISAN: no formal income limit (land-based)
        # OAP/WAP/DAP: typically ≤ 1.5L per year for state schemes
        # MGNREGA: no income limit
        if detected_scheme in ("OAP", "WAP", "DAP"):
            threshold = 150000
        elif detected_scheme == "NFSA-PDS":
            threshold = 100000
        else:
            threshold = 200000
        answers["Q11"] = {
            "answer": "Yes" if income <= threshold else "No",
            "confidence": 0.9,
            "source": f"Annual income ₹{income:,} from document — threshold ₹{threshold:,}",
            "field": "annual_income",
        }

    # Q12 — Land records updated?
    if fields.land_survey_number or fields.land_area:
        answers["Q12"] = {
            "answer": "Yes",
            "confidence": 0.75,
            "source": f"Land survey/khasra details found in document ({fields.land_survey_number or fields.land_area})",
            "field": "land_survey_number",
        }

    # Q14 — Registered in scheme?
    if fields.registration_number or fields.job_card_number or fields.ration_card_number:
        answers["Q14"] = {
            "answer": "Yes",
            "confidence": 0.9,
            "source": "Registration/beneficiary ID found in document",
            "field": "registration_number",
        }

    # Q21 — Job card valid (MGNREGA)?
    if fields.job_card_number:
        answers["Q21"] = {
            "answer": "Yes",
            "confidence": 0.85,
            "source": f"Job card number {fields.job_card_number} found in document",
            "field": "job_card_number",
        }

    # Q22 — Registered at GP level for MGNREGA?
    if fields.job_card_number:
        answers["Q22"] = {
            "answer": "Yes",
            "confidence": 0.8,
            "source": "Job card found — implies GP-level registration",
            "field": "job_card_number",
        }

    # Q36 — Registration number traceable?
    reg = fields.registration_number or fields.job_card_number or fields.ration_card_number
    if reg:
        answers["Q36"] = {
            "answer": "Yes",
            "confidence": 0.85,
            "source": f"ID '{reg}' found in document",
            "field": "registration_number",
        }

    # Q44 — eKYC on PM-KISAN portal?
    if detected_scheme == "PM-KISAN" and fields.ekyc_done is True:
        answers["Q44"] = {
            "answer": "Yes",
            "confidence": 0.9,
            "source": "eKYC completed per document",
            "field": "kyc_status",
        }

    # Q47 — DBT enabled?
    if fields.dbt_status == "enabled":
        answers["Q47"] = {
            "answer": "Yes",
            "confidence": 0.95,
            "source": "DBT enabled in document",
            "field": "dbt_status",
        }
    elif fields.dbt_status == "disabled":
        answers["Q47"] = {
            "answer": "No",
            "confidence": 0.95,
            "source": "DBT disabled/not linked in document",
            "field": "dbt_status",
        }

    # Q48 — Bank passbook checked / last transaction?
    if fields.last_transaction_date or fields.last_transaction_amount:
        txn = fields.last_transaction_amount or ""
        dt = fields.last_transaction_date or ""
        answers["Q48"] = {
            "answer": "Yes",
            "confidence": 0.9,
            "source": f"Passbook shows last transaction {txn} on {dt}".strip(" on"),
            "field": "last_transaction_date",
        }

    # Q10 — Mobile linked?
    if fields.mobile_number:
        answers["Q10"] = {
            "answer": "Yes",
            "confidence": 0.8,
            "source": f"Mobile number {fields.mobile_number} found in document",
            "field": "mobile_number",
        }

    # NFSA-specific: ration card / FPS code
    if fields.ration_card_number:
        answers["Q31"] = {
            "answer": "Yes",
            "confidence": 0.85,
            "source": f"Ration card number {fields.ration_card_number} found",
            "field": "ration_card_number",
        }

    return answers


# ─── Public API ────────────────────────────────────────────────────────────────

def process_document(file_bytes: bytes, filename: str, doc_type: str, scheme: Optional[str] = None) -> dict:
    """
    Full pipeline: extract text → parse fields → map to answers.
    Returns a dict ready to be returned as JSON.
    """
    try:
        raw_text = extract_text_from_bytes(file_bytes, filename)
    except ValueError as e:
        return {
            "success": False,
            "error": str(e),
            "fields": {},
            "auto_answers": {},
            "doc_type": doc_type,
        }

    if not raw_text.strip():
        return {
            "success": False,
            "error": "No text could be extracted from the document. "
                     "Ensure the file is a clear scan or a text-based PDF.",
            "fields": {},
            "auto_answers": {},
            "doc_type": doc_type,
        }

    fields = parse_fields(raw_text, doc_type)
    auto_answers = map_to_question_answers(fields, scheme)

    # Build clean fields dict (only non-None values)
    clean_fields = {
        k: v for k, v in {
            "name": fields.name,
            "dob": fields.dob,
            "gender": fields.gender,
            "aadhaar_number": ("XXXX XXXX " + fields.aadhaar_number[-4:]) if fields.aadhaar_number else None,
            "mobile_number": fields.mobile_number,
            "bank_account": fields.bank_account,
            "ifsc_code": fields.ifsc_code,
            "bank_name": fields.bank_name,
            "last_transaction_date": fields.last_transaction_date,
            "last_transaction_amount": fields.last_transaction_amount,
            "registration_number": fields.registration_number,
            "job_card_number": fields.job_card_number,
            "ration_card_number": fields.ration_card_number,
            "fps_code": fields.fps_code,
            "annual_income": f"₹{int(fields.annual_income):,}" if fields.annual_income else None,
            "land_survey_number": fields.land_survey_number,
            "land_area": fields.land_area,
            "dbt_status": fields.dbt_status,
            "kyc_status": fields.kyc_status,
            "detected_scheme": fields.scheme_name,
        }.items()
        if v is not None
    }

    return {
        "success": True,
        "doc_type": doc_type,
        "detected_scheme": fields.scheme_name,
        "fields": clean_fields,
        "auto_answers": auto_answers,
        "questions_answered": len(auto_answers),
        "text_length": len(raw_text),
    }
