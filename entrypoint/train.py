import os
import argparse
import sys

# 프로젝트 루트 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.common.config import ConfigLoader
from src.ingest.loader import get_loader
from src.chunking.splitter import TextSplitter
from src.indexing.vector_store import VectorStoreWrapper

def main():
    parser = argparse.ArgumentParser(description="RAG 챗봇: 데이터 적재(Ingestion) 파이프라인")
    parser.add_argument("--config", type=str, default="config/local.yaml", help="설정 파일 경로")
    parser.add_argument("--step", type=str, default="all", choices=["all", "load", "chunk", "embed"], help="실행할 파이프라인 단계")
    parser.add_argument("--limit", type=int, default=None, help="처리할 파일 개수 제한")
    args = parser.parse_args()

    # 1. 설정 로드
    print(f"설정 파일을 로드합니다: {args.config}...")
    config_loader = ConfigLoader(args.config)
    config = config_loader.config
    
    # 2. 컴포넌트 초기화
    splitter = TextSplitter(
        chunk_size=config.get("chunking", {}).get("chunk_size", 1000),
        chunk_overlap=config.get("chunking", {}).get("chunk_overlap", 200)
    )

    # 3. 데이터 로드
    raw_data_path = config.get("paths.raw_data", "data/files")
    if not os.path.exists(raw_data_path):
        os.makedirs(raw_data_path, exist_ok=True)
        print(f"데이터 디렉토리를 생성했습니다: {raw_data_path}")
    
    files = [os.path.join(raw_data_path, f) for f in os.listdir(raw_data_path) if os.path.isfile(os.path.join(raw_data_path, f))]
    if args.limit:
        files = files[:args.limit]
        print(f"파일 처리를 {args.limit}개로 제한합니다.")
    
    if not files:
        print(f"경로에 파일이 없습니다: {raw_data_path}. PDF나 HWP 파일을 넣어주세요.")
        return

    all_chunks = []
    
    for file_path in files:
        try:
            # 로드 (Load)
            loader = get_loader(file_path)
            docs = loader.load()
            print(f"문서 로드 완료 ({len(docs)}개): {os.path.basename(file_path)}")
            
            # 분할 (Chunk)
            chunks = splitter.split_documents(docs)
            print(f"  -> {len(chunks)}개의 청크 생성됨.")
            all_chunks.extend(chunks)
            
        except Exception as e:
            print(f"파일 처리 중 오류 발생 {file_path}: {e}")

    # 4. 저장 (Indexing)
    if all_chunks:
        print(f"총 {len(all_chunks)}개의 청크를 벡터 저장소(Vector DB)에 저장합니다...")
        vector_store = VectorStoreWrapper(config)
        vector_store.initialize(embedding_function=None)
        vector_store.add_documents(all_chunks)
        print("데이터 적재가 완료되었습니다.")
    else:
        print("저장할 청크가 없습니다.")

if __name__ == "__main__":
    main()
