import argparse
import json
import os
import sys
from tqdm import tqdm
from rouge_score import rouge_scorer

# 프로젝트 루트 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.common.config import config
from src.indexing.vector_store import VectorStoreWrapper
from src.generation.rag import RAGChain

def main():
    parser = argparse.ArgumentParser(description="RAG 챗봇: 성능 평가 (Evaluation)")
    parser.add_argument("--input", type=str, default="data/test_set.json", help="질문과 정답(Ground Truth)이 포함된 입력 JSON 파일")
    parser.add_argument("--output", type=str, default="data/eval_results.json", help="평가 결과 저장 파일")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"입력 파일을 찾을 수 없습니다: {args.input}")
        # 데모용 더미 데이터 생성
        dummy_data = [
            {"question": "이 사업의 예산은 얼마야?", "ground_truth": "예산 정보는 문서에 따라 다르지만 보통 금액으로 표시됩니다."},
        ]
        with open(args.input, "w", encoding="utf-8") as f:
            json.dump(dummy_data, f, ensure_ascii=False, indent=2)
        print(f"더미 테스트 셋을 생성했습니다: {args.input}")

    with open(args.input, "r", encoding="utf-8") as f:
        test_set = json.load(f)

    # RAG 시스템 초기화
    print("RAG 시스템 초기화 중...")
    vector_store = VectorStoreWrapper(config)
    vector_store.initialize()
    rag_chain = RAGChain(config=config, vector_store_wrapper=vector_store)

    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    results = []
    total_score = 0

    print(f"총 {len(test_set)}개의 항목을 평가합니다...")
    for item in tqdm(test_set):
        question = item.get("question")
        ground_truth = item.get("ground_truth")

        if not question:
            continue

        # 검색 및 답변 생성
        docs = rag_chain.retriever.invoke(question)
        answer = rag_chain.generate_answer(question, docs)

        score = 0.0
        if ground_truth:
            scores = scorer.score(ground_truth, answer)
            score = scores['rougeL'].fmeasure
        
        total_score += score
        
        results.append({
            "question": question,
            "ground_truth": ground_truth,
            "answer": answer,
            "rouge_l": score
        })

    avg_score = total_score / len(test_set) if test_set else 0
    print(f"평균 ROUGE-L 점수: {avg_score:.4f}")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"평가 결과를 저장했습니다: {args.output}")

if __name__ == "__main__":
    main()
