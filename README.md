# 🤖 RAG ChatBot (RFP Analysis Assistant)

**입찰 공고(RFP) 문서 분석 및 질의응답을 위한 AI 챗봇 프로젝트**입니다.
복잡하고 방대한 RFP 문서에서 핵심 정보를 빠르게 추출하고, 사용자의 질문에 정확하게 답변하기 위해 개발되었습니다.

---

## 📅 1. 프로젝트 배경 및 목적 (Background & Motivation)

### 왜 이 프로젝트를 만들었나요?
- **공공 입찰 공고(RFP)**는 수십~수백 페이지에 달하는 방대한 문서로, 담당자가 일일이 읽고 분석하는 데 **많은 시간과 노력**이 소요됩니다.
- 단순한 키워드 검색으로는 "예산이 얼마인가?", "자격 요건은 무엇인가?"와 같은 문맥적인 질문에 답하기 어렵습니다.
- 이러한 문제를 해결하기 위해 **LLM(Large Language Model)**과 **RAG(Retrieval-Augmented Generation)** 기술을 접목하여, 문서를 이해하고 사람처럼 답변해주는 시스템을 구축하게 되었습니다.

### 어떤 문제를 해결하나요?
- **시간 단축**: 수십 장의 문서를 1초 만에 검색하고 요약합니다.
- **정확도 향상**: 단순 검색이 아닌, 의미 기반 검색(Vector Search)과 재순위화(Re-ranking)를 통해 가장 관련성 높은 정보를 찾습니다.
- **사용성 개선**: CLI가 아닌 친숙한 **채팅 UI**를 통해 누구나 쉽게 질문할 수 있습니다.

---

## 🛠️ 2. 기술 스택 (Tech Stack)

이 프로젝트는 **최신 RAG 파이프라인**과 **Python 단일 스택**으로 구축되었습니다.

### 🔹 Core Engine
- **Language**: Python 3.12
- **LLM**: OpenAI `gpt-5-mini` (속도와 성능의 균형)
- **Framework**: LangChain (RAG 체인 및 프롬프트 관리)

### 🔹 RAG Pipeline (검색 및 데이터 처리)
- **Vector DB**: ChromaDB (로컬 파일 기반의 가벼운 벡터 저장소)
- **Embedding**: `text-embedding-3-small`
- **Retrieval**:
    - **Hybrid Search**: BM25(키워드) + Vector(의미) 결합
    - **Re-ranking**: `FlashRank` (검색 결과의 정확도 재검증)
- **Document Parsing**: `pyhwp` (HWP 문서 텍스트 추출 최적화)

### 🔹 User Interface
- **Framework**: **Streamlit** (데이터 및 AI 앱을 위한 빠른 UI 개발)
- **Features**: 실시간 스트리밍, 사이드바 문서 관리, 설정 제어(슬라이더, 고속 모드)

---

## 📂 3. 프로젝트 구조 및 가이드 (Documents)

전체 코드를 이해하고 활용하기 위해 아래 문서들을 참고하세요.

| 문서 파일 | 설명 | 주요 내용 |
| :--- | :--- | :--- |
| **`PROJECT_GUIDE.md`** | 📘 **통합 가이드북** | 프로젝트 아키텍처, 작동 원리, 트러블슈팅 로그 등 A-Z 상세 설명 |
| **`notebooks/project_summary.ipynb`** | 📓 **요약 노트북** | 웹 UI 없이 코드로 핵심 로직(검색, 생성)을 테스트하고 이해하는 실습 파일 |
| **`app.py`** | 💻 **메인 애플리케이션** | Streamlit 기반의 웹 챗봇 실행 파일 |
| **`src/`** | ⚙️ **핵심 로직** | RAG 체인(`rag.py`), 데이터 적재(`ingest`), 검색기(`retrieval`) 등 |

---

## 🚀 4. 시작하기 (Getting Started)

복잡한 설치 과정 없이 명령어 하나로 실행할 수 있도록 구성했습니다.

### 필수 조건 (Prerequisites)
- **OpenAI API Key**: `.env` 파일에 `OPENAI_API_KEY=sk-...` 설정이 필요합니다.

### 실행 방법 (Run)
터미널에서 아래 명령어를 입력하면 **가상환경 생성부터 패키지 설치, 모듈 실행**까지 자동으로 수행됩니다.

```bash
make run
```
> 실행 후 브라우저가 자동으로 열립니다: `http://localhost:8501`

---

## ✨ 5. 주요 기능 미리보기 (Features)

1.  **📄 문서 관리 (Sidebar)**: 원하는 문서만 선택해서 집중적으로 질문할 수 있습니다.
2.  **⚡ 고속 모드 (Fast Mode)**: 속도가 중요하다면 '고속 모드'를 켜서 3배 더 빠르게 검색하세요.
3.  **🎚️ 답변 길이 조절**: "상세하게" vs "100자 요약" 등 원하는 스타일로 답변을 받으세요.
4.  **⏱️ 실시간 피드백**: 답변 생성 과정을 실시간 스트리밍으로 보고, 소요 시간을 확인하세요.

---

## 📜 6. 라이센스 (License)
MIT License