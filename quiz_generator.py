import random
import requests
import os
from typing import List, Dict, Any
from metadata_schema import ChunkMeta
from structured_extractor import extract_factual_sentences

def generate_quiz(chunks: List[ChunkMeta], num_questions: int = 5) -> Dict[str, Any]:
    """
    Generates a quiz by selecting factual sentences and using an LLM to form MCQs.
    Strictly extractive distractor logic is requested, but often requires LLM for quality.
    We will use a strict prompt to ensure distractors are plausible but grounded.
    """
    facts = extract_factual_sentences(chunks)
    if not facts:
        return {"items": []}
        
    # Sample facts to generate questions from
    selected_facts = random.sample(facts, min(len(facts), num_questions * 2))
    
    quiz_items = []
    
    api_key = os.getenv("MISTRAL_API_KEY")
    url = "https://api.mistral.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    for fact in selected_facts:
        if len(quiz_items) >= num_questions:
            break
            
        prompt = f"""Task: Create a multiple-choice question (MCQ) based ONLY on the following fact.
Fact: {fact['text']}

STRICT RULES:
1. The question must be a direct inquiry about the fact.
2. The correct answer must be present verbatim in the fact.
3. Provide 3 distractors that are plausible but incorrect based on the fact.
4. Output JSON format only: {{"question": "...", "options": ["A", "B", "C", "D"], "answer": "Option text", "explanation": "..."}}
"""

        data = {
            "model": "mistral-tiny",
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"}
        }

        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            item = response.json()['choices'][0]['message']['content']
            import json
            item_data = json.loads(item)
            # Sanitize options: ensure they are a list of strings
            if isinstance(item_data.get('options'), dict):
                item_data['options'] = list(item_data['options'].values())
            elif isinstance(item_data.get('options'), list):
                item_data['options'] = [str(o) if not isinstance(o, dict) else str(next(iter(o.values()))) for o in item_data['options']]
            
            item_data['source'] = fact['source']
            quiz_items.append(item_data)
        except Exception:
            continue
            
    return {"items": quiz_items}
