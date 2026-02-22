import pytest

def test_quiz_generator():
    from quiz_generator import generate_mcqs_from_facts
    
    facts = [
        {"type": "numeric", "text": "10 epochs", "source": {"doc_id": "doc1"}},
        {"type": "numeric", "text": "20 epochs", "source": {"doc_id": "doc1"}},
        {"type": "numeric", "text": "30 epochs", "source": {"doc_id": "doc1"}},
        {"type": "numeric", "text": "40 epochs", "source": {"doc_id": "doc1"}},
    ]
    
    mcqs = generate_mcqs_from_facts(facts, max_questions=2)
    
    assert len(mcqs) == 2
    for mcq in mcqs:
        assert len(mcq["options"]) == 4
        assert mcq["answer"] in mcq["options"]
        assert all(opt in [f["text"] for f in facts] for opt in mcq["options"])
