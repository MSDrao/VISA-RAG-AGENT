"""
Structured keyword taxonomy used for:
- query-side metadata detection
- chunk-side metadata extraction
- retrieval/ranking boosts
"""

VISA_KEYWORD_GROUPS = {
    "H-1B": ["H-1B", "H-1B1", "H-4", "Concurrent H-1B", "H-1B Transfer", "H-1B Lottery", "Cap Exempt"],
    "F-1": ["F-1", "F-2", "SEVIS", "SEVP", "OPT", "STEM OPT", "CPT", "Day 1 CPT", "Cap Gap", "Grace Period"],
    "J-1": ["J-1", "J-2", "AT", "Academic Training", "DS-2019"],
    "B-1/B-2": ["B-1", "B-2", "B1/B2", "Visa Stamping", "Visa Renewal", "214(b)", "221(g)"],
    "O-1": ["O-1", "O-1A", "O-1B", "O-2", "O-3"],
    "L-1": ["L-1A", "L-1B", "L-2"],
    "Employment-based": ["EB-1A", "EB-1B", "EB-1C", "EB-2", "EB-2 NIW", "EB-3", "EB-4", "EB-5", "PERM", "Priority Date", "Visa Bulletin"],
    "Travel": ["Advance Parole", "Port of Entry", "Reentry", "Reentry Permit", "Automatic Visa Revalidation", "D/S", "Authorized Stay"],
}

PROCESS_KEYWORDS = [
    "Adjustment of Status",
    "Change of Status",
    "Extension of Status",
    "Consular Processing",
    "Premium Processing",
    "Biometrics",
    "Interview Waiver",
    "Visa Bulletin",
    "Priority Date",
    "Retrogression",
    "Unlawful Presence",
    "Out of Status",
    "Overstay",
    "Reinstatement",
    "RFE",
    "NOID",
    "NOIR",
    "RFIE",
    "Receipt Notice",
    "Approval Notice",
    "Denial Notice",
    "USCIS Case Status",
]

AGENCY_KEYWORDS = {
    "USCIS": ["USCIS", "USCIS Online Account", "Receipt Notice", "Approval Notice"],
    "CBP": ["CBP", "Port of Entry", "CBP Officer", "Secondary Inspection", "I-94"],
    "DOS": ["DOS", "Department of State", "Embassy", "Consulate", "CEAC", "Visa Bulletin"],
    "DHS": ["DHS", "ICE", "SEVP", "SEVIS"],
    "DOL": ["DOL", "Labor Certification", "Prevailing Wage", "ETA-9089", "ETA-9141", "ETA-9035", "LCA"],
    "NVC": ["NVC", "DS-260"],
    "IRS": ["IRS"],
    "SSA": ["SSA", "SSN"],
}

FORM_KEYWORDS = [
    "I-20", "DS-160", "DS-260", "DS-2019", "I-94", "I-797", "I-797A", "I-797B", "I-797C",
    "I-129", "I-130", "I-131", "I-134", "I-140", "I-485", "I-539", "I-765", "I-824",
    "I-864", "I-901", "I-983", "I-9", "ETA-9089", "ETA-9141", "ETA-9035", "G-28",
    "N-400", "N-600", "AR-11",
]

EMPLOYMENT_KEYWORDS = [
    "Employer Sponsor", "Sponsoring Employer", "Petitioner", "Beneficiary", "End Client",
    "Vendor", "Client Letter", "Employment Verification Letter", "Offer Letter", "Pay Stub",
    "W-2", "1099", "LCA", "Certified LCA", "Bench Pay", "Specialty Occupation", "SOC Code",
    "Wage Level", "Prevailing Wage Determination", "Work Authorization", "Unauthorized Employment",
    "Full-Time CPT", "Part-Time CPT", "Remote Work", "Multiple Employers", "Concurrent Employment",
]

FAMILY_KEYWORDS = [
    "Dependent Visa", "Spouse Visa", "Child Dependent", "H-4 EAD", "Derivative Beneficiary",
    "Principal Applicant", "Marriage Certificate", "Birth Certificate", "Affidavit of Support",
    "Sponsor Letter", "Household Member",
]

TRAVEL_KEYWORDS = [
    "Reentry", "Reentry Permit", "Travel Ban", "Visa Expiry", "Status Expiry", "Authorized Stay",
    "Admit Until Date", "Duration of Status", "D/S", "Multiple Entry", "Single Entry",
    "Port of Entry Stamp", "Secondary Inspection", "Travel History", "International Travel",
    "Automatic Visa Revalidation", "Abandonment of Application", "Departure Record", "Entry Record",
]


def canonicalize_keyword(value: str) -> str:
    return value.strip().lower().replace("/", " / ")
