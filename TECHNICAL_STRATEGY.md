# RFP RAG 시스템 기술 전략 발표

## 📋 목차

1. [프로젝트 개요](#프로젝트-개요)
2. [RAG 전략](#rag-전략)
3. [Chunking 전략](#chunking-전략)
4. [Indexing 전략](#indexing-전략)
5. [프롬프팅 전략](#프롬프팅-전략)
6. [라이브러리 선택 및 이유](#라이브러리-선택-및-이유)
7. [성능 최적화 전략](#성능-최적화-전략)

---

## 프로젝트 개요

### 목표

- **RFP(제안요청서) 문서 자동 분석 및 제안서 생성 시스템**
- 한국어 공공기관 입찰 문서(PDF, HWP, DOCX) 처리
- RAG 기반 질의응답 및 제안서 자동 생성

### 핵심 기능

- 문서 파싱 및 전처리
- 벡터 임베딩 및 검색
- 질의응답(Q&A)
- 제안서 자동 생성
- 대화형 제안서 빌드업

---

## RAG 전략

### 1. 아키텍처 개요

```
[문서] → [파싱] → [청킹] → [임베딩] → [벡터 DB]
                                           ↓
[사용자 쿼리] → [임베딩] → [검색] → [Reranking] → [LLM] → [응답]
```

### 2. RAG 파이프라인 설계

#### 2.1 Retrieval 단계

- **Vector Similarity Search**: 코사인 유사도 기반 검색
- **Hybrid Search**: 벡터 검색 + BM25 (선택적)
- **MMR Reranking**: 다양성과 관련성 균형
- **Metadata Filtering**: 문서 ID, 사업명 등으로 필터링

#### 2.2 Augmentation 단계

- 검색된 상위 K개 청크를 컨텍스트로 구성
- 메타데이터(사업명, 공고번호 등) 포함
- 청크 간 구분자(`---`)로 명확한 구분

#### 2.3 Generation 단계

- LangChain의 `ChatPromptTemplate` 사용
- System Prompt + User Prompt 구조
- 컨텍스트 기반 답변 생성

### 3. RAG 구현 세부사항

#### 3.1 Retrieval Agent

```python
# src/retrieval/retrieval_agent.py
- VectorSearch: 벡터 유사도 검색
- HybridSearch: 벡터 + BM25 하이브리드
- Reranker: MMR 알고리즘으로 다양성 확보
- MetadataFilter: 메타데이터 기반 필터링
```

**특징:**

- `top_k * 2`로 더 많은 후보 검색 후 reranking
- MMR lambda 파라미터로 다양성 조절 (기본값: 0.5)
- 메타데이터 필터링으로 정확도 향상

#### 3.2 RAG Chain

```python
# src/generation/rag_chain.py
- 컨텍스트 빌드: 검색 결과를 구조화된 텍스트로 변환
- 프롬프트 템플릿: System + User 메시지 구조
- LLM Fallback: 모델 접근 실패 시 자동 대체
```

**프롬프트 구조:**

- System Prompt: 역할 정의 및 규칙 명시
- User Prompt: 컨텍스트 + 질문
- 출처 명시 요구: 답변의 신뢰성 확보

### 4. RAG 전략의 장점

1. **정확성**: 문서 기반 답변으로 환각(Hallucination) 최소화
2. **확장성**: 새로운 문서 추가 시 재인덱싱만 필요
3. **투명성**: 출처 표시로 검증 가능
4. **유연성**: 다양한 질의 유형 지원

---

## Chunking 전략

### 1. 청킹 방식 선택

#### 1.1 Fixed-Size Chunking (현재 구현)

```python
# src/chunking/chunker.py
- Chunk Size: 1000자 (기본값)
- Chunk Overlap: 200자 (20%)
- 최소 청크 크기: 100자
```

**선택 이유:**

- **구현 단순성**: 빠른 프로토타이핑 가능
- **일관성**: 모든 문서에 동일한 크기 적용
- **예측 가능성**: 토큰 수 예측 용이

**장점:**

- 구현이 간단하고 빠름
- 메모리 사용량 예측 가능
- 임베딩 비용 계산 용이

**단점:**

- 문장/단락 경계 무시
- 의미 단위 분할 어려움
- 긴 문장이 잘릴 수 있음

#### 1.2 Section-Based Chunking (향후 개선)

```python
# src/chunking/section_chunker.py (스텁)
- 섹션 제목 기반 분할
- 의미 단위 보존
- 더 정확한 검색 가능
```

**향후 개선 방향:**

- RFP 문서의 구조적 특성 활용
- "1. 사업 개요", "2. 기술 요구사항" 등 섹션 단위 분할
- 섹션 메타데이터 추가

### 2. 청킹 파라미터 최적화

#### 2.1 Chunk Size

- **1000자 선택 이유:**
  - 한국어 평균 토큰: 1자 ≈ 0.4 토큰
  - 1000자 ≈ 400 토큰 (임베딩 모델 입력 제한 내)
  - 충분한 컨텍스트 제공
  - 검색 정확도와 성능 균형

#### 2.2 Chunk Overlap

- **200자 (20%) 선택 이유:**
  - 문장 경계를 넘어선 정보 보존
  - 검색 시 연속성 확보
  - 너무 큰 overlap은 중복 증가 → 20%가 적절

#### 2.3 최소 청크 크기

- **100자 선택 이유:**
  - 너무 작은 청크는 의미 부족
  - 마지막 조각 병합으로 정보 손실 방지

### 3. 청킹 구현 세부사항

```python
class TextChunker:
    def chunk(self, text: str, doc_id: str, metadata: Dict):
        # 1. 고정 크기로 분할
        # 2. Overlap 적용
        # 3. 마지막 작은 청크 병합
        # 4. 메타데이터 첨부
```

**처리 로직:**

1. 텍스트를 `chunk_size` 단위로 분할
2. 다음 청크 시작 위치: `end - overlap`
3. 마지막 청크가 `min_chunk_size` 미만이면 이전 청크에 병합
4. 각 청크에 `doc_id`, `chunk_index`, `metadata` 첨부

### 4. 메타데이터 관리

**청크별 메타데이터:**

- `doc_id`: 문서 식별자
- `chunk_index`: 청크 순서
- `char_offset_start/end`: 원본 텍스트 위치
- `사업명`, `공고 번호`, `발주 기관` 등: RFP 메타데이터

**메타데이터 활용:**

- 필터링: 특정 사업명/공고번호로 검색
- 출처 표시: 답변에 문서 정보 포함
- 검색 정확도 향상

---

## Indexing 전략

### 1. 벡터 데이터베이스 선택: ChromaDB

#### 1.1 ChromaDB 선택 이유

**장점:**

1. **경량성**: 경량 임베딩 데이터베이스
2. **로컬 저장**: 서버리스, 클라우드 의존성 없음
3. **Python 네이티브**: LangChain 통합 용이
4. **메타데이터 필터링**: 강력한 필터링 기능
5. **무료 오픈소스**: 상업적 사용 가능

**다른 옵션과 비교:**

| DB           | 장점             | 단점                | 선택 이유         |
| ------------ | ---------------- | ------------------- | ----------------- |
| **ChromaDB** | 경량, 로컬, 무료 | 대규모 확장성 제한  | 프로토타입에 적합 |
| Pinecone     | 관리형, 확장성   | 유료, 클라우드 의존 | 초기 비용 부담    |
| Weaviate     | 기능 풍부        | 복잡도, 리소스      | 과도한 기능       |
| Qdrant       | 성능 우수        | 설정 복잡           | 현재 규모에 과함  |

#### 1.2 ChromaDB 구현

```python
# src/indexing/vector_store.py
class VectorStore:
    def __init__(self, config):
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self._create_or_get_collection(name)
```

**특징:**

- **PersistentClient**: 디스크에 영구 저장
- **Collection 기반**: 문서 타입별 분리 가능
- **메타데이터 정리**: None 값, 빈 문자열 제거

### 2. 임베딩 모델 전략

#### 2.1 모델 선택: OpenAI Embeddings

**주요 모델:**

- **Primary**: `text-embedding-3-large` (3072차원)
- **Fallback 1**: `text-embedding-3-small` (1536차원)
- **Fallback 2**: `text-embedding-ada-002` (1536차원)

**선택 이유:**

1. **한국어 지원**: OpenAI 임베딩은 한국어 성능 우수
2. **차원 선택**: 3-large는 정확도, 3-small은 속도/비용
3. **Fallback 전략**: 모델 접근 실패 시 자동 대체

#### 2.2 임베딩 생성 전략

```python
# src/indexing/embedder.py
class Embedder:
    - Batch Processing: 100개씩 배치 처리
    - Retry Logic: 최대 3회 재시도
    - Model Fallback: 자동 모델 전환
```

**최적화:**

- **배치 크기 100**: API 호출 최소화
- **재시도 로직**: 네트워크 오류 대응
- **Fallback**: 모델 접근 권한 문제 자동 해결

### 3. 인덱싱 파이프라인

#### 3.1 인덱싱 프로세스

```
[청크 JSONL]
    ↓
[배치 로드] (100개씩)
    ↓
[임베딩 생성] (OpenAI API)
    ↓
[메타데이터 정리] (None 제거, 타입 변환)
    ↓
[ChromaDB 저장]
```

#### 3.2 메타데이터 정리 전략

**문제:**

- ChromaDB는 `None`, 빈 문자열, 복잡한 타입 거부
- JSON에서 `null` → Python `None` 변환

**해결:**

```python
def _clean_metadata(self, metadata: Dict) -> Dict:
    # 1. None 값 제거
    # 2. 빈 문자열 제거
    # 3. 리스트/딕셔너리 → 문자열 변환
    # 4. bool, int, float, str만 허용
```

**예시:**

```python
# Before
{
    "공고 번호": None,
    "사업명": "교육 시스템",
    "tags": ["교육", "IT"]
}

# After
{
    "사업명": "교육 시스템",
    "tags": "['교육', 'IT']"  # 문자열로 변환
}
```

### 4. 인덱싱 성능 최적화

#### 4.1 배치 처리

- **배치 크기 100**: API 호출 수 최소화
- **병렬 처리 가능**: 향후 개선 가능

#### 4.2 에러 처리

- **재시도 로직**: 일시적 오류 대응
- **Fallback 모델**: 접근 권한 문제 해결
- **로깅**: 실패 원인 추적

---

## 프롬프팅 전략

### 1. 프롬프트 구조 설계

#### 1.1 RAG 프롬프트 (Q&A)

```python
SYSTEM_PROMPT = """당신은 RFP(제안요청서) 문서 분석 전문가입니다.
제공된 문서 컨텍스트를 기반으로 사용자의 질문에 정확하고 상세하게 답변하세요.

규칙:
1. 반드시 제공된 컨텍스트만을 기반으로 답변하세요.
2. 컨텍스트에 없는 정보는 추측하지 말고 "문서에 명시되지 않음"이라고 명시하세요.
3. 답변의 출처를 명확히 표시하세요 (문서 ID, 섹션명 등).
4. 한국어로 자연스럽고 전문적으로 답변하세요.
"""

USER_PROMPT = """다음은 RFP 문서의 관련 부분입니다:

{context}

사용자 질문: {question}

위 컨텍스트를 기반으로 질문에 답변하세요. 답변 끝에 출처를 명시하세요.
"""
```

**설계 원칙:**

1. **명확한 역할 정의**: "RFP 문서 분석 전문가"
2. **제약 조건 명시**: 컨텍스트 기반만 답변
3. **출처 요구**: 검증 가능성 확보
4. **언어 명시**: 한국어 답변

#### 1.2 제안서 생성 프롬프트

```python
PROPOSAL_PROMPT = """RFP 문서를 분석하여 제안서를 작성하세요.

RFP 정보:
{context}
{conversation_context}
{previous_proposal_context}

8개 섹션으로 제안서를 작성하세요 (각 섹션 3-5문단):

1. 사업 이해 및 배경
2. 제안 개요
3. 기술 제안
4. 사업 수행 계획
5. 조직 및 인력 구성
6. 예산 및 제안 금액
7. 기대 효과 및 성과
8. 차별화 포인트
{additional_instructions}

{update_instruction}

**중요: 최소 2000자 이상 작성하세요. 지금 바로 작성하세요:**
"""
```

**특징:**

- **구조화된 출력**: 8개 섹션 명시
- **최소 길이 요구**: 2000자 이상
- **대화 컨텍스트**: 이전 대화 반영
- **이전 제안서**: 점진적 개선 지원

### 2. 프롬프트 엔지니어링 기법

#### 2.1 Few-Shot Learning

- 예시 포함으로 출력 형식 가이드
- 제안서 섹션 구조 명시

#### 2.2 Chain-of-Thought

- 단계별 사고 과정 유도
- "RFP 분석 → 제안서 작성" 흐름

#### 2.3 Role-Based Prompting

- "전문가" 역할 부여
- 전문적 톤 유도

### 3. 컨텍스트 구성 전략

#### 3.1 컨텍스트 빌드

```python
def _build_context(self, results: List[Dict]) -> str:
    context_parts = []
    for result in results:
        metadata = result.get("metadata", {})
        business_name = metadata.get("사업명", "N/A")
        section_name = metadata.get("section_name", "N/A")

        doc_info = f"[문서: {business_name}, 섹션: {section_name}]"
        chunk_text = result.get("chunk_text", "")

        context_parts.append(f"{doc_info}\n{chunk_text}\n")

    return "\n---\n".join(context_parts)
```

**전략:**

- **메타데이터 포함**: 문서 정보로 출처 명확화
- **구분자 사용**: `---`로 청크 구분
- **순서 유지**: 검색 점수 순서 유지

#### 3.2 컨텍스트 길이 제한

**제안서 생성:**

- 최대 3개 청크 사용 (토큰 절약)
- 청크당 400자로 제한
- 총 약 1200자 (약 480 토큰)

**Q&A:**

- 최대 5개 청크 사용
- 더 많은 컨텍스트로 정확도 향상

### 4. 프롬프트 최적화 전략

#### 4.1 토큰 효율성

- 불필요한 설명 제거
- 간결한 지시사항
- 구조화된 출력 요구

#### 4.2 출력 품질 향상

- 최소 길이 명시
- 섹션 구조 명시
- 구체적 지시사항

#### 4.3 대화형 빌드업

- 이전 대화 기록 포함
- 이전 제안서 참조
- 점진적 개선 지시

---

## 라이브러리 선택 및 이유

### 1. 핵심 라이브러리

#### 1.1 LangChain & LangChain-Core

**선택 이유:**

- **표준화된 인터페이스**: LLM, 임베딩, 벡터 DB 통합
- **프롬프트 관리**: `ChatPromptTemplate`로 구조화
- **체인 구성**: RAG 파이프라인 쉽게 구성
- **커뮤니티**: 활발한 커뮤니티와 문서

**사용 예시:**

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(model="gpt-5-nano")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
prompt = ChatPromptTemplate.from_messages([...])
```

**대안 고려:**

- **LlamaIndex**: 더 복잡하고 리소스 집약적
- **Haystack**: 주로 검색 중심, RAG는 LangChain이 더 적합

#### 1.2 ChromaDB

**선택 이유:**

- **경량성**: 서버리스, 경량 벡터 DB
- **로컬 저장**: 클라우드 의존성 없음
- **Python 네이티브**: LangChain 통합 용이
- **메타데이터 필터링**: 강력한 필터링 기능
- **무료**: 상업적 사용 가능

**사용 예시:**

```python
import chromadb
client = chromadb.PersistentClient(path="./data/index/chroma")
collection = client.get_or_create_collection("rfp_chunks")
```

**대안 고려:**

- **Pinecone**: 유료, 클라우드 의존
- **Weaviate**: 과도한 기능, 복잡도
- **Qdrant**: 현재 규모에 과함

#### 1.3 OpenAI API (LangChain-OpenAI)

**선택 이유:**

- **한국어 성능**: GPT 모델의 한국어 이해도 우수
- **임베딩 품질**: `text-embedding-3-large`의 정확도
- **안정성**: 프로덕션 레벨 안정성
- **Fallback 지원**: 모델 접근 실패 시 대체 모델

**사용 예시:**

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

llm = ChatOpenAI(
    model="gpt-5-nano",
    temperature=0.2,
    max_tokens=4000
)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)
```

**대안 고려:**

- **Claude (Anthropic)**: 한국어 지원 제한
- **로컬 LLM (Llama)**: 성능 및 한국어 지원 부족
- **한국어 특화 모델**: API 안정성 및 통합 어려움

### 2. 문서 처리 라이브러리

#### 2.1 PyPDF

**선택 이유:**

- **경량**: 가벼운 PDF 파서
- **순수 Python**: 외부 의존성 최소
- **텍스트 추출**: 기본 텍스트 추출에 충분

**사용 예시:**

```python
import pypdf
reader = pypdf.PdfReader(file_path)
text = "\n".join([page.extract_text() for page in reader.pages])
```

**대안 고려:**

- **pdfplumber**: 더 정확하지만 무거움
- **PyMuPDF (fitz)**: 성능 우수하지만 복잡도 높음

#### 2.2 pyhwp / hwp5

**선택 이유:**

- **한국어 전용**: HWP 파일 전용 파서
- **오픈소스**: 무료 사용 가능
- **텍스트 추출**: 기본 텍스트 추출 지원

**사용 예시:**

```python
import hwp5
from hwp5.proc import restext

hwp5_file = hwp5.open(hwp_path)
text_parts = [str(text) for text in restext.extract(hwp5_file)]
```

**대안 고려:**

- **olefile**: 기본 파싱만 가능, 한글 인코딩 문제
- **상용 라이브러리**: 비용 부담

#### 2.3 python-docx

**선택 이유:**

- **표준 라이브러리**: DOCX 처리 표준
- **간단한 API**: 사용이 쉬움
- **텍스트 추출**: 기본 기능 충분

**사용 예시:**

```python
import docx
document = docx.Document(docx_path)
text = "\n".join([para.text for para in document.paragraphs])
```

### 3. 검색 및 유사도 라이브러리

#### 3.1 rapidfuzz

**선택 이유:**

- **Fuzzy Matching**: 오타, 변형 처리
- **성능**: C++ 기반으로 빠름
- **한국어 지원**: 기본적인 한국어 지원

**사용 예시:**

```python
from rapidfuzz import fuzz
similarity = fuzz.ratio(query, text)
```

#### 3.2 rank-bm25 (향후 사용)

**선택 이유:**

- **BM25 알고리즘**: 키워드 기반 검색
- **하이브리드 검색**: 벡터 + BM25 결합
- **Python 구현**: 사용이 쉬움

**향후 활용:**

- Hybrid Search 구현
- 키워드 매칭 강화

### 4. API 및 웹 프레임워크

#### 4.1 FastAPI

**선택 이유:**

- **고성능**: 비동기 지원
- **자동 문서화**: Swagger UI 자동 생성
- **타입 검증**: Pydantic으로 자동 검증
- **현대적**: Python 3.7+ 최신 기능 활용

**사용 예시:**

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class ProposalRequest(BaseModel):
    query: Optional[str] = None
    top_k: Optional[int] = 30

@app.post("/api/generate-proposal")
async def generate_proposal(request: ProposalRequest):
    ...
```

**대안 고려:**

- **Flask**: 동기식, 성능 제한
- **Django**: 과도한 기능, 무거움

#### 4.2 Pydantic

**선택 이유:**

- **타입 검증**: 자동 데이터 검증
- **FastAPI 통합**: 완벽한 통합
- **문서화**: 자동 API 문서 생성

### 5. 유틸리티 라이브러리

#### 5.1 python-dotenv

**선택 이유:**

- **환경 변수 관리**: `.env` 파일 지원
- **보안**: API 키 등 민감 정보 관리
- **편의성**: 설정 관리 용이

#### 5.2 PyYAML

**선택 이유:**

- **설정 파일**: YAML 형식으로 설정 관리
- **가독성**: JSON보다 읽기 쉬움
- **중첩 구조**: 복잡한 설정 표현

---

## 성능 최적화 전략

### 1. 임베딩 최적화

#### 1.1 배치 처리

- **배치 크기 100**: API 호출 수 최소화
- **병렬 처리 가능**: 향후 개선 가능

#### 1.2 Fallback 전략

- **자동 모델 전환**: 접근 권한 문제 해결
- **재시도 로직**: 일시적 오류 대응

### 2. 검색 최적화

#### 2.1 Reranking

- **MMR 알고리즘**: 다양성과 관련성 균형
- **Lambda 파라미터**: 0.5로 설정 (균형)

#### 2.2 컨텍스트 제한

- **제안서 생성**: 3개 청크만 사용 (토큰 절약)
- **Q&A**: 5개 청크 사용 (정확도 우선)

### 3. LLM 최적화

#### 3.1 모델 선택

- **gpt-5-nano**: 비용 효율성
- **Fallback**: gpt-5-mini, gpt-4

#### 3.2 토큰 관리

- **max_tokens**: 제안서 4000, Q&A 2000
- **컨텍스트 제한**: 입력 토큰 최소화

### 4. 프롬프트 최적화

#### 4.1 간결성

- 불필요한 설명 제거
- 구조화된 지시사항

#### 4.2 명확성

- 구체적 요구사항 명시
- 출력 형식 명시

---

## 결론

### 핵심 전략 요약

1. **RAG**: 벡터 검색 + MMR Reranking으로 정확도 향상
2. **Chunking**: Fixed-size (1000자)로 빠른 구현, 향후 Section-based 개선
3. **Indexing**: ChromaDB로 경량 벡터 DB, OpenAI 임베딩으로 한국어 성능
4. **프롬프팅**: 구조화된 프롬프트로 일관된 출력

### 향후 개선 방향

1. **Section-Based Chunking**: 의미 단위 분할
2. **Hybrid Search**: BM25 통합으로 키워드 검색 강화
3. **로컬 LLM**: 비용 절감을 위한 옵션
4. **평가 시스템**: Recall@K, Precision@K 등 정량적 평가

### 학습 포인트

- **라이브러리 선택**: 프로젝트 규모와 요구사항에 맞는 선택
- **Fallback 전략**: 프로덕션 환경에서 필수
- **메타데이터 관리**: 검색 정확도 향상의 핵심
- **프롬프트 엔지니어링**: 출력 품질의 핵심

---

## 참고 자료

- [LangChain 문서](https://python.langchain.com/)
- [ChromaDB 문서](https://www.trychroma.com/)
- [OpenAI Embeddings 가이드](https://platform.openai.com/docs/guides/embeddings)
- [RAG 최적화 가이드](https://www.pinecone.io/learn/retrieval-augmented-generation/)
