"""
policies.py
Scheme definitions and the diagnostic question bank.
Question text lives here; question IDs (Q01–Q50) are the contract with taxonomy.json.
"""

from pathlib import Path
import json
from typing import Optional

_TAXONOMY_PATH = Path(__file__).parent.parent / "data" / "taxonomy.json"
_qbank_cache: Optional[dict] = None


# ---------------------------------------------------------------------------
# Scheme metadata
# ---------------------------------------------------------------------------

SCHEMES: dict[str, dict] = {
    "PM-KISAN": {
        "full_name": "Pradhan Mantri Kisan Samman Nidhi",
        "description": "Income support of Rs 6,000/year to small and marginal farmers in 3 installments.",
        "ministry": "Ministry of Agriculture & Farmers Welfare",
        "beneficiary_type": "Farmer",
        "payment_frequency": "4-monthly installments",
        "helpline": "155261 / 011-24300606",
    },
    "MGNREGA": {
        "full_name": "Mahatma Gandhi National Rural Employment Guarantee Act",
        "description": "Guarantees 100 days of wage employment per year to rural households.",
        "ministry": "Ministry of Rural Development",
        "beneficiary_type": "Rural household",
        "payment_frequency": "Per work completion (FTO-based)",
        "helpline": "1800-111-555",
    },
    "OAP": {
        "full_name": "Old Age Pension (Indira Gandhi National Old Age Pension Scheme)",
        "description": "Monthly pension for citizens aged 60+ from BPL families.",
        "ministry": "Ministry of Rural Development",
        "beneficiary_type": "Senior citizen (60+, BPL)",
        "payment_frequency": "Monthly",
        "helpline": "1800-111-555",
    },
    "WAP": {
        "full_name": "Widow Pension (Indira Gandhi National Widow Pension Scheme)",
        "description": "Monthly pension for widows aged 40–59 from BPL families.",
        "ministry": "Ministry of Rural Development",
        "beneficiary_type": "Widow (40–59, BPL)",
        "payment_frequency": "Monthly",
        "helpline": "1800-111-555",
    },
    "DAP": {
        "full_name": "Disability Pension (Indira Gandhi National Disability Pension Scheme)",
        "description": "Monthly pension for severely disabled persons from BPL families.",
        "ministry": "Ministry of Rural Development",
        "beneficiary_type": "Disabled person (BPL)",
        "payment_frequency": "Monthly",
        "helpline": "1800-111-555",
    },
    "NFSA-PDS": {
        "full_name": "National Food Security Act – Public Distribution System",
        "description": "Subsidised food grains to eligible households through Fair Price Shops.",
        "ministry": "Ministry of Consumer Affairs, Food and Public Distribution",
        "beneficiary_type": "Priority/AAY household",
        "payment_frequency": "Monthly ration distribution",
        "helpline": "1967",
    },
    "EPFO": {
        "full_name": "Employees' Provident Fund Organisation",
        "description": "Provident fund and pension for organised sector employees.",
        "ministry": "Ministry of Labour & Employment",
        "beneficiary_type": "Organised sector employee / retiree",
        "payment_frequency": "Monthly (pension) / on withdrawal",
        "helpline": "1800-118-005",
    },
    "NPS": {
        "full_name": "National Pension System",
        "description": "Voluntary long-term retirement savings scheme.",
        "ministry": "Ministry of Finance / PFRDA",
        "beneficiary_type": "Subscriber (government / private)",
        "payment_frequency": "Monthly annuity post retirement",
        "helpline": "1800-222-080",
    },
    "PMJDY": {
        "full_name": "Pradhan Mantri Jan Dhan Yojana",
        "description": "Financial inclusion — bank accounts with overdraft, insurance, and RuPay card.",
        "ministry": "Ministry of Finance",
        "beneficiary_type": "Unbanked individuals",
        "payment_frequency": "DBT for various schemes",
        "helpline": "1800-11-0001",
    },
    "PMJJBY": {
        "full_name": "Pradhan Mantri Jeevan Jyoti Bima Yojana",
        "description": "Life insurance cover of Rs 2 lakh for death due to any reason.",
        "ministry": "Ministry of Finance",
        "beneficiary_type": "Bank account holders aged 18–50",
        "payment_frequency": "Annual premium",
        "helpline": "Respective bank helpline",
    },
    "PMSBY": {
        "full_name": "Pradhan Mantri Suraksha Bima Yojana",
        "description": "Accidental insurance cover of Rs 2 lakh at Rs 20/year.",
        "ministry": "Ministry of Finance",
        "beneficiary_type": "Bank account holders aged 18–70",
        "payment_frequency": "Annual premium",
        "helpline": "Respective bank helpline",
    },
    "PMJAY": {
        "full_name": "Pradhan Mantri Jan Arogya Yojana (Ayushman Bharat)",
        "description": "Health coverage of Rs 5 lakh/year for secondary and tertiary hospitalisation.",
        "ministry": "Ministry of Health & Family Welfare",
        "beneficiary_type": "BPL / SECC-listed families",
        "payment_frequency": "Cashless hospitalisation",
        "helpline": "14555",
    },
}


# ---------------------------------------------------------------------------
# Question bank
# ---------------------------------------------------------------------------

def get_question_bank() -> dict[str, str]:
    """
    Return the full question bank {Q_id: question_text}.
    Loaded from taxonomy.json to keep a single source of truth.
    """
    global _qbank_cache
    if _qbank_cache is None:
        with open(_TAXONOMY_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        _qbank_cache = data.get("question_bank", {})
    return _qbank_cache


def get_question_text(question_id: str) -> str:
    """Return the text for a question ID, or a fallback string."""
    return get_question_bank().get(question_id, f"[Unknown question: {question_id}]")


def get_scheme_questions(scheme: str) -> dict[str, str]:
    """
    Return a filtered question bank relevant to a specific scheme.
    Uses a curated mapping — keeps the question bank manageable per scheme.
    """
    scheme_question_map: dict[str, list[str]] = {
        "PM-KISAN": [
            "Q01", "Q02", "Q03", "Q04", "Q05", "Q06", "Q07", "Q08",
            "Q09", "Q11", "Q12", "Q14", "Q15", "Q16", "Q17", "Q20",
            "Q34", "Q35", "Q36", "Q38", "Q39", "Q41", "Q43", "Q44",
            "Q47", "Q48",
        ],
        "MGNREGA": [
            "Q01", "Q02", "Q03", "Q06", "Q07", "Q08", "Q09", "Q14",
            "Q17", "Q30", "Q31", "Q32", "Q33", "Q34", "Q35", "Q36",
            "Q38", "Q39", "Q42", "Q45", "Q46", "Q47", "Q48",
        ],
        "OAP": [
            "Q01", "Q02", "Q03", "Q06", "Q07", "Q09", "Q13", "Q16",
            "Q18", "Q19", "Q20", "Q21", "Q22", "Q23", "Q24", "Q25",
            "Q34", "Q35", "Q36", "Q39", "Q50",
        ],
        "WAP": [
            "Q01", "Q02", "Q03", "Q06", "Q07", "Q09", "Q16",
            "Q18", "Q19", "Q21", "Q22", "Q23", "Q24", "Q25",
            "Q34", "Q35", "Q36", "Q39", "Q50",
        ],
        "DAP": [
            "Q01", "Q02", "Q03", "Q06", "Q07", "Q09", "Q13", "Q16",
            "Q18", "Q19", "Q21", "Q22", "Q23", "Q24",
            "Q34", "Q35", "Q36", "Q39", "Q49", "Q50",
        ],
        "NFSA-PDS": [
            "Q06", "Q08", "Q14", "Q26", "Q27", "Q28", "Q29",
            "Q35", "Q36", "Q39",
        ],
        "EPFO": [
            "Q01", "Q02", "Q03", "Q06", "Q07", "Q09", "Q16",
            "Q19", "Q21", "Q22", "Q23", "Q24", "Q25",
            "Q34", "Q35", "Q36", "Q39",
        ],
        "NPS": [
            "Q01", "Q02", "Q03", "Q06", "Q07", "Q09", "Q16",
            "Q19", "Q21", "Q22", "Q23", "Q24", "Q25",
            "Q34", "Q35", "Q36", "Q39",
        ],
    }

    full_bank = get_question_bank()
    relevant_ids = scheme_question_map.get(scheme, list(full_bank.keys()))
    return {qid: full_bank[qid] for qid in relevant_ids if qid in full_bank}


def get_scheme_info(scheme: str) -> Optional[dict]:
    """Return scheme metadata or None."""
    return SCHEMES.get(scheme)


def get_all_scheme_codes() -> list[str]:
    """Return list of all scheme codes."""
    return list(SCHEMES.keys())
