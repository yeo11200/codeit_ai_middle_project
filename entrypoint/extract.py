import argparse
import sys
import os
import json
from tqdm import tqdm

# 프로젝트 루트 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ingest.loader import get_loader
from src.ingest.metadata import MetadataExtractor

def main():
    parser = argparse.ArgumentParser(description="RAG 챗봇: 메타데이터 추출")
    parser.add_argument("--file", type=str, help="처리할 단일 파일 경로")
    parser.add_argument("--dir", type=str, default="data/files", help="처리할 파일이 있는 디렉토리")
    parser.add_argument("--all", action="store_true", help="디렉토리 내 모든 파일 처리")
    parser.add_argument("--limit", type=int, default=None, help="처리할 파일 개수 제한")
    parser.add_argument("--output", type=str, default="data/metadata.json", help="출력 JSON 파일 경로")
    
    args = parser.parse_args()
    
    extractor = MetadataExtractor()
    results = []

    files_to_process = []
    if args.file:
        files_to_process.append(args.file)
    elif args.all:
        if not os.path.exists(args.dir):
            print(f"디렉토리를 찾을 수 없습니다: {args.dir}")
            return
        files = [os.path.join(args.dir, f) for f in os.listdir(args.dir) if os.path.isfile(os.path.join(args.dir, f))]
        files_to_process.extend(files)
    else:
        print("--file 또는 --all 옵션을 지정해주세요.")
        return

    if args.limit:
        files_to_process = files_to_process[:args.limit]
        print(f"파일 처리를 {args.limit}개로 제한합니다.")

    print(f"총 {len(files_to_process)}개의 파일을 처리합니다...")
    
    for file_path in tqdm(files_to_process):
        try:
            # 텍스트 로드 (앞부분만 사용하여 메타데이터 추출)
            loader = get_loader(file_path)
            docs = loader.load()
            
            if not docs:
                continue
                
            full_text = "\n".join([d.page_content for d in docs])
            
            # 메타데이터 추출 실행
            metadata = extractor.extract(full_text)
            
            # Dictionary 변환
            meta_dict = metadata.model_dump()
            meta_dict["source_file"] = os.path.basename(file_path)
            
            results.append(meta_dict)
            
            # 단일 파일 모드일 경우 즉시 출력
            if args.file:
                print(json.dumps(meta_dict, ensure_ascii=False, indent=2))
                
        except Exception as e:
            print(f"파일 처리 중 오류 발생 {file_path}: {e}")

    # 결과 저장
    if args.all or len(results) > 1:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"추출된 메타데이터를 저장했습니다: {args.output}")

if __name__ == "__main__":
    main()
