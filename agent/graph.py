from langgraph.graph import StateGraph, END
from src.agent.nodes import AgentNodes
from src.agent.state import AgentState

def build_rag_graph(rag_chain_instance):
    """
    RAG 에이전트의 사고 흐름(Graph)을 설계하고 빌드하는 함수입니다.
    기존의 rag_chain 인스턴스를 주입받아 각 노드(작업 단위)에 연결합니다.
    """
    # 1. 실제 작업을 수행할 노드 객체 생성 (HyDE, 검색, 검증 등)
    nodes = AgentNodes(rag_chain_instance)
    
    # 2. AgentState 기반의 상태 그래프(워크플로우) 초기화
    workflow = StateGraph(AgentState)

    # --- [노드 등록 단계] ---
    # 각 함수를 그래프 상의 하나의 작업 지점(Node)으로 등록합니다.
    workflow.add_node("hyde", nodes.hyde_node)           # 질문 확장 (가짜 답변 생성)
    workflow.add_node("retrieve", nodes.retrieve_node)   # 문서 검색 (RAG 엔진 활용)
    workflow.add_node("rerank", nodes.rerank_node)       # 결과 재정렬 (L4 GPU 가속 활용)
    workflow.add_node("grade", nodes.grade_node)         # 품질 검증 (Self-Correction 판단)
    workflow.add_node("generate", nodes.generate_node)   # 최종 답변 생성

    # --- [워크플로우 연결 단계] ---
    
    # 에이전트의 시작점 설정 (가장 먼저 HyDE를 통해 질문을 풍부하게 만듦)
    workflow.set_entry_point("hyde")
    
    # 단순 직선 경로 (Edge) 연결
    workflow.add_edge("hyde", "retrieve")     # 질문 확장 후 -> 검색 실행
    workflow.add_edge("retrieve", "rerank")   # 검색 후 -> 문서 순위 재정렬
    workflow.add_edge("rerank", "grade")     # 재정렬 후 -> 문서 적합성 검사 단계로

    # --- [조건부 로직(Conditional Edge) 설정] ---
    # 'grade' 노드의 결과에 따라 경로를 분기합니다.
    workflow.add_conditional_edges(
        "grade",                              # 기준이 되는 노드
        lambda x: x["is_relevant"],           # 판단 함수 (nodes.py의 리턴값 확인)
        {
            "yes": "generate",                # 정보가 충분하면 -> 최종 답변 생성으로 이동
            "no": "hyde"                      # 정보가 부족하면 -> 질문 재구성(HyDE)부터 다시 시도(루프)
        }
    )

    # 최종 답변 생성이 완료되면 워크플로우를 종료(END)합니다.
    workflow.add_edge("generate", END)

    # 모든 설정이 완료된 그래프를 컴파일하여 실행 가능한 '앱' 형태로 반환합니다.
    return workflow.compile()