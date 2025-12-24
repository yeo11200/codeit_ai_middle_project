# 🛠️ RAG ChatBot 프로젝트 기술 보고서 (Technical Report)

본 문서는 **"LLM 기반 제안요청서(RFP) 질의응답 시스템"** 개발 프로젝트의 기술적 구현 사항을 A부터 Z까지 상술한 보고서입니다.

---

## 1. 🏗️ 프로젝트 아키텍처 (Architecture)

### 1.1 전체 구조
-   **단일 애플리케이션 (Standalone Application)**: 복잡한 프론트엔드/백엔드 분리 없이 **Streamlit**을 사용하여 하나의 Python 프로세스에서 구동되는 구조.
-   **RAG (Retrieval-Augmented Generation)**: 외부 지식(RFP 파일)을 검색하여 LLM에 주입하는 패턴 적용.

### 1.2 기술 스택 (Tech Stack)
| 분류 | 기술명 | 용도 및 선정 이유 |
| :--- | :--- | :--- |
| **언어** | Python 3.12 | AI/ML 생태계 표준, 풍부한 라이브러리 |
| **프레임워크** | **Streamlit** | 빠른 프로토타이핑 및 데이터 애플리케이션 UI 구축 |
| **LLM** | **OpenAI (`gpt-5-mini`)** | 높은 한국어 이해도 & 속도 대비 저렴한 비용 |
| **Vector DB** | **ChromaDB** | 서버 설치가 필요 없는 임베디드형 DB, 로컬 환경 최적화 |
| **Embedding** | `text-embedding-3-small` | OpenAI 최신 모델, 한글 의미론적 검색 성능 우수 |
| **검색 엔진** | **Hybrid Search** (BM25 + Vector) | 키워드 매칭(정확성)과 의미 검색(맥락)의 장점 결합 |
| **Re-ranking** | **FlashRank** | 검색 결과의 정확도 보정을 위한 경량화된 재순위화 모델 |
| **HWP 파싱** | `pyhwp` (hwp5txt) | HWP 파일의 텍스트를 깨짐 없이 추출 (subprocess 활용) |

---

## 2. ⚙️ 핵심 구현 내용 (Implementation Details)

### 2.1 데이터 파이프라인 (Data Ingestion)
1.  **HWP 파일 처리**: `olefile` 등 기존 라이브러리의 한계를 극복하기 위해 `hwp5txt` 바이너리를 서브프로세스로 호출하여 순수 텍스트 추출.
2.  **청킹 (Chunking)**: `RecursiveCharacterTextSplitter`를 사용.
    -   `chunk_size=1000`: 문맥 유지를 위한 충분한 길이.
    -   `chunk_overlap=200`: 문단 경계에서의 정보 손실 방지.

### 2.2 검색 고도화 전략 (Advanced Retrieval)
단순한 코사인 유사도 검색만으로는 부족하여 2단계 검색 전략을 수립했습니다.
1.  **1단계 (Broad Retrieval)**: `Top-K=5`
    -   키워드 중심의 **BM25**와 의미 중심의 **Vector Search**를 5:5 가중치로 앙상블.
2.  **2단계 (Precision Re-ranking)**: `FlashRank`
    -   1단계에서 뽑힌 후보군을 Cross-Encoder 모델로 정밀 채점하여, 질문과의 연관성이 가장 높은 상위 3개를 최종 선별.

### 2.3 성능 최적화 (Performance & UX)
1.  **고속 모드 (Fast Mode)**:
    -   속도를 중시하는 사용자를 위해 무거운 **Re-ranking 단계를 생략**하는 토글 기능 구현. (응답 시간 90% 단축)
2.  **답변 길이 조절**:
    -   사용자 니즈에 따라 시스템 프롬프트를 동적으로 교체 (`상세`, `보통`, `요약`, `초요약`).
3.  **캐싱 (Caching)**:
    -   `@st.cache_resource`를 사용하여 무거운 RAG 모델 및 DB 로딩을 최초 1회만 수행하도록 최적화.

---

## 3. ☁️ 인프라 및 배포 (DevOps)

### 3.1 개발 환경 자동화
-   **Makefile**: `make run` 명령어 하나로 가상환경 생성, 의존성 설치, 앱 실행을 원스톱으로 처리.

### 3.2 Streamlit Cloud 배포 최적화
1.  **패키지 경량화**: `opencv`, `qdrant` 등 미사용 대형 라이브러리 제거 (`requirements.txt`).
2.  **호환성 패치**: Streamlit Cloud(Linux)의 구버전 SQLite 문제를 해결하기 위해 `pysqlite3-binary`를 적용하고 `app.py` 구동 시점에 동적으로 모듈을 교체하는 코드 삽입.

---

## 4. 📝 결론 (Conclusion)
본 프로젝트는 **"현업에서 바로 쓸 수 있는 수준"**을 목표로 개발되었습니다.
단순한 챗봇이 아니라, **사용자가 검색 품질(속도 vs 정확도)을 제어**하고, 결과를 **투명하게(소요시간, 출처)** 확인할 수 있는 **"도구(Tool)"**로서의 가치를 극대화했습니다.
