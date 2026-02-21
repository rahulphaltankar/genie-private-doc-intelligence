import pytest
from quiz_generator import generate_quiz
from metadata_schema import ChunkMeta

def test_quiz_structure():
    mock_chunks = [
        ChunkMeta(chunk_id="1", doc_id="d1", filename="test.pdf", text="The capital of France is Paris. It has a population of 2 million.", page=1)
    ]
    # Note: This will attempt to call Mistral API, might need mocking in a real CI
    # For now, we test the logic assuming environment variable might be present or handled
    quiz = generate_quiz(mock_chunks, num_questions=1)
    
    assert "items" in quiz
    if len(quiz["items"]) > 0:
        item = quiz["items"][0]
        assert "question" in item
        assert "options" in item
        assert "answer" in item
        assert "source" in item
        assert item["source"]["filename"] == "test.pdf"

def test_empty_chunks():
    quiz = generate_quiz([], num_questions=1)
    assert quiz["items"] == []
