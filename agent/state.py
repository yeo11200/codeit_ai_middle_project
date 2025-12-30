from typing import TypedDict, List, Optional
from langchain_core.documents import Document

class AgentState(TypedDict):
    """
    에이전트 워크플로우 전반에서 유지되는 '상태(State)' 정의입니다.
    각 노드는 이 데이터를 읽고, 수정하여 다음 노드로 전달합니다.
    """

    # 사용자의 현재 질문 (HyDE 노드를 거치면 AI가 확장한 가상 답변이 포함된 긴 텍스트로 업데이트될 수 있음)
    question: str 
    
    # 사용자가 입력한 최초의 순수 질문 (필요시 최종 답변 생성 시 참조하기 위해 보존)
    original_question: str 
    
    # 사용자가 분석 대상으로 직접 선택한 파일 리스트 (필터링 조건으로 활용)
    selected_docs: Optional[List[str]]
    
    # 검색된 문서들의 리스트 
    # [L4 GPU 특화] VRAM 24GB 환경이므로, 많은 양의 페이지(Context)를 담아도 추론 속도가 안정적입니다.
    documents: List[Document] 
    
    # 에이전트가 최종적으로 생성한 분석 결과 답변
    answer: str
    
    # 검색된 문서의 적합성 판정 결과 ('yes': 답변 생성으로 진행, 'no': 다시 검색 루프 실행)
    # 이 값에 따라 graph.py에서 경로가 결정됩니다.
    is_relevant: str 
    
    # 검색 실패 또는 품질 저하 시 무한 루프에 빠지지 않도록 제어하는 카운터
    retry_count: int

    