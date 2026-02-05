import pytest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config
from scripts.generate_data import settings

settings=config.Settings()
# Add project root to path
from src.semantic_engine import SemanticEngine

@pytest.fixture(scope="module")
def engine():
    print("Initializing SemanticEngine for testing...")
    return SemanticEngine(faq_json_path="data/faq.json",collection_name=settings.collection_name,persist_dir=settings.chroma_db_dir)

@pytest.mark.parametrize(
    "query,expected",
    [
        ("my customer count", "How many customers do I have?"),
        ("what is my product count ?", "How many products do I have?")
    ]
)
def test_search(engine, query, expected):
    search_result = engine.search(query=query)

    faq_question = search_result.original_question
    similarity = search_result.similarity

    print(f"\nquery ('{query}' matched with '{faq_question}'): {similarity:.4f}")

    assert faq_question == expected

def test_similarity_close_match(engine):
    text_a = "How many customer do I have"
    text_b = "My customer count"

    similarity = engine.semantic_similarity(text_a, text_b,semantic_boost=0.65,keyword_boost=0.98)
    print(f"\nSimilarity ('{text_a}' vs '{text_b}'): {similarity:.4f}")

    assert similarity > 0.95, "Similarity should be high for similar topics"

def test_similarity_distant_match(engine):

    text_a = "How may customer do I have"
    text_b = "How many products do I have"

    similarity = engine.semantic_similarity(text_a, text_b,semantic_boost=0.65,keyword_boost=0.98)
    print(f"\nSimilarity ('{text_a}' vs '{text_b}'): {similarity:.4f}")

    assert similarity < 0.60, "Similarity should be lower for different topics"