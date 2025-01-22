import os
from typing import Dict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PREPROCESS_PROMPTS: Dict[str, str] = {
    "cleaner": open(os.path.join(BASE_DIR, "preprocess/cleaner.txt"), "r", encoding="utf-8").read(),
    "transformer": open(os.path.join(BASE_DIR, "preprocess/transformer.txt"), "r", encoding="utf-8").read(),
    "extractor": open(os.path.join(BASE_DIR, "preprocess/extractor.txt"), "r", encoding="utf-8").read()
}

ANALYZE_PROMPTS: Dict[str, str] = { 

}

DETECT_PROMPTS: Dict[str, str] = {
    
}

PROCESS_PROMPTS: Dict[str, str] = {
    
}

EVALUATE_PROMPTS: Dict[str, str] = {
    
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