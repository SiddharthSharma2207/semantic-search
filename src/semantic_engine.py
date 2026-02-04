import json

import chromadb
from chromadb.config import Settings
import spacy
from typing import List, Dict, Any, Optional
import numpy as np


class SemanticEngine:
    def __init__(self, faq_json_path: str, persist_dir: str = "chroma_db", collection_name: str = "faq_collection"):
        """
        Initialize the Semantic Engine with a local ChromaDB.
        """
        with open(faq_json_path, "r") as f:
            self.faq = json.loads(f.read())
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_collection(name=collection_name)

        # Load spaCy for keyword extraction
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Downloading spaCy model...")
            from spacy.cli import download
            download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")

    def extract_keywords(self, text: str) -> List[str]:
        """
        Extract key terms from the query for pre-filtering.
        Removes stop words and keeps nouns/verbs/adjs usually.
        """
        doc = self.nlp(text.lower())
        keywords = [
            token.text for token in doc
            if not token.is_stop and token.is_alpha and len(token.text) > 2
        ]
        return list(set(keywords))


    def search(self, query: str, n_results: int = 5) -> Optional[Dict[str, Any]]:
        """
        Perform the hybrid search:
        1. Extract Keywords
        2. Create Filter (OR condition)
        3. Vector Search with Filter
        """
        keywords = self.extract_keywords(query)
        print(f"extracted keywords: {keywords}")

        where_filter = None
        if keywords:
            # Construct Chroma $or filter
            # { "$or": [ {"document": {"$contains": "kw1"}}, ... ] }
            # Note: Chroma's $contains is on metadata or document content?
            # Chroma acts on metadata with 'where' and document content with 'where_document'.
            # We want to match text content usually, so we use 'where_document'.

            or_conditions = [{"$contains": kw} for kw in keywords]
            if len(or_conditions) > 1:
                where_filter = {"$or": or_conditions}
            else:
                where_filter = or_conditions[0]

        # Query
        # We try to use the filter. If it's too restrictive (no results),
        # we might want to fallback to pure vector search, but requirements say "Create search filter".
        # We will strictly follow "Perform a vector search with search filter".

        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where_document=where_filter
            )
        except Exception as e:
            print(f"Search failed with filter {where_filter}: {e}")
            # Fallback to searching without filters
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
            )

        if not results["ids"] or len(results["ids"][0]) == 0:
            return None

        # Return best match
        best_doc = results["documents"][0][0]
        best_meta = results["metadatas"][0][0]
        best_dist = results["distances"][0][0]
        query_id = best_meta["query_id"]

        # Note: Chroma default is cosine distance (lower is better, 0 is identical)
        # However, we often want similarity.
        # Distance range for cosine is 0 to 2.

        return {
            "answer": self.faq[query_id]["answer"],
            "topic": best_meta["topic"],
            "matched_question": best_doc,
            "original_question": self.faq[query_id]["question"],
            "query_id":query_id,
            "distance": best_dist,
            "similarity": 1.0 - best_dist  # Approximation for cosine distance
        }
