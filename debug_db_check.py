from src.common.config import config
from src.indexing.vector_store import VectorStoreWrapper
import os

def check_venture_doc():
    vector_store = VectorStoreWrapper(config)
    vector_store.initialize()
    
    # Target filename (partial match)
    target_part = "벤처기업협회"
    
    print(f"Querying ChromaDB for docs with source containing '{target_part}'...")
    
    # Chroma get() to fetch metadata/documents
    # We can't use 'contains' in 'where' clause easily in standard Chroma without specific implementation,
    # but we can fetch all and filter or try to match exact path if we knew it.
    # Let's try to fetch by a likely exact path or just list some.
    # Actually, let's use the 'get' without where and filter in python if the db is small enough (it's <200 docs).
    
    data = vector_store.vector_store.get()
    
    print(f"Total docs in DB: {len(data['ids'])}")
    
    found_count = 0
    for i, meta in enumerate(data['metadatas']):
        source = meta.get('source', '')
        if target_part in source:
            content = data['documents'][i]
            print(f"\n[FOUND MATCH {found_count+1}]")
            print(f"Source: {source}")
            print(f"Content Preview: {content[:100]}...")
            
            if "파일명:" in content and target_part in content:
                print(">>> SUCCESS: Filename is injected in the content! <<<")
            else:
                print(">>> WARNING: Filename NOT found in content. (Old chunk?) <<<")
            
            found_count += 1
            
    if found_count == 0:
        print("\n[ERROR] No document found for 벤처기업협회 in the DB.")

if __name__ == "__main__":
    check_venture_doc()
