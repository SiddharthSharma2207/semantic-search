import argparse
from src.semantic_engine import SemanticEngine
from config import Settings

settings=Settings()

search_engine=SemanticEngine(faq_json_path="data/faq.json",
    persist_dir=settings.chroma_db_dir,
                             collection_name=settings.collection_name)



if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Search FAQ section")

    parser.add_argument("--query", type=str, help="User query")

    args = parser.parse_args()

    print("Searching FAQ section....\n")
    result=search_engine.search(query=args.query,n_results=5)

    if result:
        print(f'Answer:\n{result["answer"]}')
