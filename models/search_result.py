from pydantic import  BaseModel

class SearchResult(BaseModel):
    answer: str
    topic: str
    matched_question: str
    original_question: str
    query_id: str
    distance: float
    similarity: float