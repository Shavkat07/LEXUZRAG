import json
import os
os.environ["LEXUZ_MOCK_SEARCH"] = "0"

from lexuz.api.engine import RAGEngine
from lexuz.api.schemas import OptionsRequest

engine = RAGEngine()
res = engine.process_item(
    item_id="q_debug_1",
    question="Mehnat ta'tilining eng kam muddati qancha ?",
    item_language="uz-Latn",
    options=OptionsRequest(top_k=5, return_contexts=True, return_debug=True),
)

data = res.model_dump()
print("=== RESULT JSON ===")
print(json.dumps(data, indent=2, ensure_ascii=False))
