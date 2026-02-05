import json
from typing import List, Optional, Union

import chromadb
import numpy as np
import spacy
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from spacy.cli import download

from models.search_result import SearchResult


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
            download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")

    def extract_keywords(self, text: str) -> Union[List[str], None]:
        """
        Extract key terms from the query for pre-filtering.
        Removes stop words and keeps nouns/verbs/adjs usually.
        """
        try:
            doc = self.nlp(text.lower())
            keywords = [
                token.text for token in doc
                if not token.is_stop and token.is_alpha and len(token.text) > 2
            ]
            return list(set(keywords))
        except Exception as e:
            print(f"Error occurred while extracting keywords: {e} ")
            return None

    def semantic_similarity(self, text_a: str, text_b: str, keyword_boost: float = 0.95,
                            semantic_boost: float = 0.6) -> float:
        """
        Hybrid similarity: keyword overlap + semantic similarity.
        semantic_boost: Increasing it will increase weightage for semantic match
        keyword_boost: Increasing it will increase weightage for keyword match
        """

        # --- 1) Keyword Overlap Score ---
        # simple normalized overlap of non-stop words
        ka = set(self.extract_keywords(text_a) or [])
        kb = set(self.extract_keywords(text_b) or [])
        if not ka and not kb:
            keyword_score = 0.0
        else:
            intersect = len(ka.intersection(kb))
            union = len(ka.union(kb))
            keyword_score = intersect / union

        # --- 2) Semantic Cosine Similarity ---
        embedding_fn = DefaultEmbeddingFunction()

        emb_a = embedding_fn([text_a])[0]
        emb_b = embedding_fn([text_b])[0]
        va = np.array(emb_a, dtype=float)
        vb = np.array(emb_b, dtype=float)
        na, nb = np.linalg.norm(va), np.linalg.norm(vb)

        if na == 0 or nb == 0:
            semantic_score = 0.0
        else:
            semantic_score = float(np.dot(va, vb) / (na * nb))

        final_score = semantic_boost * semantic_score + keyword_boost * keyword_score

        # clip to [0,1]
        return max(0.0, min(1.0, final_score))

    def search(self, query: str, n_results: int = 5) -> Optional[SearchResult]:
        """
        Perform the hybrid search:
        1. Extract Keywords
        2. Create Filter (OR condition)
        3. Vector Search with Filter
        """
        try:
            keywords = self.extract_keywords(query)
            print(f"extracted keywords: {keywords}")

            where_filter = None
            if keywords:
                or_conditions = [{"$contains": kw} for kw in keywords]
                if len(or_conditions) > 1:
                    where_filter = {"$or": or_conditions}
                else:
                    where_filter = or_conditions[0]

            # Query
            # We try to use the filter. If it's too restrictive (no results),
            # we might want to fallback to pure vector search.

            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where_document=where_filter
            )

            if not results["ids"] or len(results["ids"][0]) == 0:
                print(f"Search failed with filter {where_filter}")
                # Fallback to searching without filters
                results = self.collection.query(
                    query_texts=[query],
                    n_results=n_results,
                )

            # Return best match
            best_doc = results["documents"][0][0]
            best_meta = results["metadatas"][0][0]
            best_dist = results["distances"][0][0]
            query_id = best_meta["query_id"]

            return SearchResult(answer=self.faq[query_id]["answer"],
                                topic=best_meta["topic"],
                                matched_question=best_doc,
                                original_question=self.faq[query_id]["question"],
                                query_id=query_id,
                                distance=best_dist,
                                similarity=1.0 - best_dist
                                )
        except Exception as e:
            print(f"Semantic Search failed: {e}")
