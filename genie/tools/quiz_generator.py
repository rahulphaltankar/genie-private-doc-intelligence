# genie/quiz_generator.py
import random
from typing import List, Dict, Any
import logging
_logger = logging.getLogger(__name__)

def generate_mcqs_from_facts(facts: List[Dict[str,Any]], max_questions=10):
    """
    facts: list of {type,text,source}
    returns: list of MCQ dicts:
    {
      question: str,
      options: [str],
      answer: str,
      source: {...}
    }
    """
    # shuffle to vary selection, but deterministic in tests you can set seed
    random.shuffle(facts)

    # index facts by simple type buckets for distractor selection
    by_type = {}
    for f in facts:
        by_type.setdefault(f["type"], []).append(f)

    mcqs = []
    for f in facts:
        if len(mcqs) >= max_questions:
            break
        correct = f["text"]
        q = formulate_question_from_fact(f)
        # build distractors
        distractors = []
        # prefer same-type distractors
        pool = [x["text"] for x in by_type.get(f["type"], []) if x["text"] != f["text"]]
        # fallback to other facts
        if len(pool) < 3:
            pool += [x["text"] for x in facts if x["text"] != f["text"]]
        # take up to 3 distractors from pool
        pool = list(dict.fromkeys(pool))  # uniq
        if not pool:
            continue  # cannot create safe MCQ
        # deterministic selection
        distractors = pool[:3]
        options = [correct] + distractors
        # ensure 4 options
        while len(options) < 4:
            # padding by shuffling existing options (not ideal but safe)
            options.append(random.choice(options))
        random.shuffle(options)
        # final sanitize: ensure answer present
        if correct not in options:
            options[0] = correct
            random.shuffle(options)
        mcqs.append({
            "question": q,
            "options": options,
            "answer": correct,
            "source": f["source"]
        })
    return mcqs

def formulate_question_from_fact(fact: Dict[str,Any]) -> str:
    """
    Very conservative templates based on fact type
    """
    t = fact["type"]
    text = fact["text"]
    if t == "definition":
        # naive split: "<term> is defined as <def>"
        return f"According to the document, which statement correctly defines: \"{shorten(text,40)}\"?"
    if t == "equation":
        return f"Which expression is stated in the document? \"{shorten(text,40)}\""
    if t == "numeric":
        return f"Which numeric fact is stated in the document? \"{shorten(text,40)}\""
    # default
    return f"According to the document: {shorten(text,80)}"

def shorten(s, n=80):
    return (s[:n].rsplit(' ',1)[0] + '...') if len(s) > n else s
