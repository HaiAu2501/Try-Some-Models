import os
from typing import Dict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def read_file(file_name: str) -> str:
    return open(os.path.join(BASE_DIR, file_name), "r", encoding="utf-8").read()

PREPROCESS_PROMPTS: Dict[str, str] = {
    "cleaner": read_file("preprocess/cleaner.txt"),
    "transformer": read_file("preprocess/transformer.txt"),
    "extractor": read_file("preprocess/extractor.txt"),
    "reviewer": read_file("preprocess/reviewer.txt"),
}

ANALYZE_PROMPTS: Dict[str, str] = { 

}

DETECT_PROMPTS: Dict[str, str] = {
    
}

PROCESS_PROMPTS: Dict[str, str] = {

}

EVALUATE_PROMPTS: Dict[str, str] = {
    # Nhóm 1
    "econometrician": read_file("evaluate/1_econometrician.txt"),
    "empirical_economist": read_file("evaluate/1_empirical_economist.txt"),
    "normative_economist": read_file("evaluate/1_normative_economist.txt"),
    "macroeconomist": read_file("evaluate/1_macroeconomist.txt"),
    "microeconomist": read_file("evaluate/1_microeconomist.txt"),

    # Nhóm 2
    "behavioral_economist": read_file("evaluate/2_behavioral_economist.txt"),
    "socio_economist": read_file("evaluate/2_socio_economist.txt"),

    # Nhóm 3
    "corporate_management_expert": read_file("evaluate/3_corporate_management_expert.txt"),
    "financial_economist": read_file("evaluate/3_financial_economist.txt"),
    "international_economist": read_file("evaluate/3_international_economist.txt"),
    "logistics_and_supply_chain_expert": read_file("evaluate/3_logistics_and_supply_chain_expert.txt"),
    "trade_and_commerce_expert": read_file("evaluate/3_trade_and_commerce_expert.txt"),

    # Nhóm 4
    "digital_economy_and_innovation_expert": read_file("evaluate/4_digital_economy_and_innovation_expert.txt"),
    "environmental_economist": read_file("evaluate/4_environmental_economist.txt"),
    "public_policy_and_political_economy_expert": read_file("evaluate/4_public_policy_and_political_economy_expert.txt"),
}

CRITICIZE_PROMPTS: Dict[str, str] = {

}

STRATEGIZE_PROMPTS: Dict[str, str] = {
    
}

POSTPROCESS_PROMPTS: Dict[str, str] = {

}

PROMPTS: Dict[str, Dict[str, str]] = {
    "preprocess": PREPROCESS_PROMPTS,
    "analyze": ANALYZE_PROMPTS,
    "detect": DETECT_PROMPTS,
    "process": PROCESS_PROMPTS,
    "evaluate": EVALUATE_PROMPTS,
    "criticize": CRITICIZE_PROMPTS,
    "strategize": STRATEGIZE_PROMPTS,
    "postprocess": POSTPROCESS_PROMPTS
}