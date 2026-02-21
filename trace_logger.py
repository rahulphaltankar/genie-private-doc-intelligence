import json
import os
from datetime import datetime

LOG_FILE = "genie_trace_log.jsonl"

def log_trace(query, answer, grounding_score, decision, sources):
    trace = {
        "timestamp": datetime.utcnow().isoformat(),
        "query": query,
        "answer": answer,
        "grounding_score": grounding_score,
        "decision": decision,
        "sources": list(sources)
    }

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(trace) + "\n")
