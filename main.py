import argparse
from src.semantic_engine import SemanticEngine
from config import Settings

settings=Settings()

search_engine=SemanticEngine(faq_json_path="data/faq.json",
    persist_dir=settings.chroma_db_dir,
                             collection_name=settings.collection_name,
                             )



if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Search FAQ section")

    parser.add_argument("--query", type=str, help="User query")

    args = parser.parse_args()

    print("Searching FAQ section....\n")
    result=search_engine.search(query=args.query,n_results=5)

    if result:
        print(f'Answer:\n{result.answer}')
    # text_1="How may customer do I have?"
    # text_2="How many products do I have"
    # text_3="My customer count"
    #
    # result= search_engine.semantic_similarity(text_1,text_2,semantic_boost=0.65,keyword_boost=0.98)
    # print(f"Similarity: {result}")
