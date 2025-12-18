# 📌 입찰메이트 RFP RAG 프로젝트

이 프로젝트는 **기업/정부 RFP(제안요청서) 문서를 PDF/HWP로 수집**하고,  
**RAG + LangChain 기반**으로 핵심정보 추출·요약·Q&A를 제공하는 **사내 검색/분석 시스템**을 만드는 것을 목표로 합니다.

---

## 🚀 빠른 시작 (Quick Start)

**실행 가이드**: [EXECUTION_GUIDE.md](./EXECUTION_GUIDE.md) 참조  
**테스트 가이드**: [TESTING_GUIDE.md](./TESTING_GUIDE.md) 참조  
**환경변수 설정**: [ENV_SETUP.md](./ENV_SETUP.md) 참조  
**API 가이드**: [API_GUIDE.md](./API_GUIDE.md) 참조

### 기본 실행 순서

```bash
# 1. 환경 설정
export OPENAI_API_KEY=your_key_here  # 필수! ENV_SETUP.md 참조
pip install -r requirements.txt

# 2. 전체 파이프라인 실행
python entrypoint/train.py --config config/local.yaml --step all

# 3. 검색 테스트
python entrypoint/inference.py --config config/local.yaml --mode qa --query "질문"
```

자세한 실행 방법은 **[EXECUTION_GUIDE.md](./EXECUTION_GUIDE.md)**를 참조하세요.  
테스트 방법은 **[TESTING_GUIDE.md](./TESTING_GUIDE.md)**를 참조하세요.

---

아래 폴더 구조는 **실험(노트북)과 운영(코드)을 분리**하고,  
**재현 가능성 / 배포 가능성 / 테스트 가능성**을 동시에 확보하기 위한 표준 레이아웃입니다.

---

## 1) `config/` — 설정 파일 (코드와 분리)

**목적**  
환경별(Local/Prod) 설정을 코드에서 분리하여 배포·실험 시 실수를 줄입니다.

**예시 구성**
- `local.yaml`  
  - 로컬 개발용  
  - Chroma / FAISS  
  - 작은 모델  
  - 로그 레벨: DEBUG
- `prod.yaml`  
  - 운영 환경용  
  - Qdrant  
  - 보수적 파라미터  
  - 로그 레벨: INFO
- `schema.yaml` (선택)  
  - 필드 추출 JSON 스키마  
  - 예산 / 마감일 / 제출 방식 등

**보통 포함되는 설정**
- 데이터 경로 (`data/raw`, `data/preprocessed` 등)
- 청크 크기 / 오버랩
- 임베딩 모델, LLM 모델
- Vector DB 설정 (컬렉션명, host, api key)
- Retrieval 옵션 (top-k, MMR 여부, 필터링 규칙)
- OCR 옵션 (paddle / tesseract, dpi, 전처리)

---

## 2) `data/` — 데이터 생명주기 (원본 → 가공 → 피처 → 결과)

**목적**  
원본 데이터를 보존하면서 모든 가공 단계를 **추적 가능**하게 관리합니다.

**추천 구조**
- `data/raw/`  
  - 제공받은 원본 PDF / HWP  
  - `data_list.csv`
- `data/preprocessed/`  
  - 파싱 결과 텍스트  
  - OCR 결과  
  - 정규화 텍스트
- `data/features/`  
  - 청킹 결과 (JSONL / Parquet)
- `data/index/`  
  - Vector DB 로컬 퍼시스트 파일
- `data/predictions/`  
  - 요약 / 필드 추출 / 배치 QA 결과
- `data/eval/`  
  - 평가셋 (질문, 정답 doc_id, 근거)  
  - 평가 결과 리포트

**핵심 원칙**
- `raw` 데이터는 읽기 전용처럼 취급
- 생성물은 항상 다음 단계 폴더에 저장
- 실험별 스냅샷은  
  `data/predictions/run_YYYYMMDD_HHMM/` 형식 권장

---

## 3) `entrypoint/` — 실행 스크립트 (파이프라인 출入口)

**목적**  
실제 실행은 여기서만 수행하고, 로직은 `src/`에 둡니다.

- `entrypoint` → 조립 / 실행
- `src` → 기능 / 모듈

**예시 파일**
- `train.py`  
  - 문서 로딩  
  - PDF/HWP 파싱  
  - OCR (필요 시)  
  - 청킹  
  - 임베딩  
  - Vector DB 적재
- `inference.py`  
  - 질의 처리  
  - 문서 후보 식별 (메타 + 유사도)  
  - Retrieval  
  - 답변 / 요약 / 필드 추출
- `evaluate.py` (추천)  
  - 평가셋 기반 지표 산출  
  - 문서 식별 정확도  
  - 근거 포함률 등

---

## 4) `notebooks/` — 탐색 / 실험 전용 (운영 로직 금지)

**목적**  
EDA와 실험을 자유롭게 수행하되, 운영 코드 오염을 방지합니다.

**예시**
- `01_eda_metadata.ipynb`  
  - `data_list.csv` 분포  
  - 기관명 / 사업명 표기 흔들림 분석
- `02_pdf_vs_ocr_quality.ipynb`  
  - OCR 품질 비교  
  - 전처리 파라미터 튜닝
- `03_chunking_experiments.ipynb`  
  - 청킹 사이즈 / 오버랩 실험
- `04_retrieval_ablation.ipynb`  
  - similarity vs MMR vs hybrid 비교

**원칙**
- 노트북은 분석/보고서 용도만 사용
- 확정된 로직은 `src/`로 이전하여 모듈화

---

## 5) `src/` — 핵심 코드 (모듈화 + 테스트 가능)

**목적**  
프로젝트의 실제 제품 코드가 위치하는 영역입니다.

**추천 하위 구조**
- `src/common/`  
  - 설정 로더, 로깅, 유틸, 상수
- `src/ingest/`  
  - PDF/HWP 로더  
  - OCR 파이프라인  
  - 텍스트 정규화
- `src/chunking/`  
  - 기본 청킹  
  - 섹션 기반 청킹 (심화)
- `src/indexing/`  
  - 임베딩 생성  
  - Vector DB 업서트  
  - 메타데이터 스키마 관리
- `src/retrieval/`  
  - 메타 fuzzy 매칭  
  - similarity / MMR / hybrid  
  - re-ranking
- `src/generation/`  
  - 요약 체인  
  - Q&A 체인  
  - 필드 추출(JSON 스키마)
- `src/eval/`  
  - 평가셋 로더  
  - 지표 계산  
  - 리포트 생성
- `src/api/` (선택)  
  - FastAPI 엔드포인트

---

## 6) `tests/` — 자동화 검증 (침묵 실패 방지)

**목적**  
문서 파이프라인 특유의 “조용한 실패”를 방지합니다.

**추천 테스트**
- PDF 파서가 빈 텍스트를 반환하면 실패
- OCR 결과 길이가 비정상적으로 짧으면 실패
- 청킹 결과 개수 / 길이 분포 검증
- 필수 메타데이터 누락 검증
- Retrieval 결과가 doc_id 필터를 준수하는지 검증

---

## 7) `docker/` + 환경 파일 — 재현 가능성 확보

**목적**  
로컬 / 서버 / CI 어디서든 동일하게 실행 가능하도록 합니다.

**추천 파일**
- `Dockerfile`
- `docker-compose.yml` (선택: Qdrant / Redis 포함 시)
- `.env.example`  
  - 필수 환경변수 템플릿
- `.env`  
  - 로컬 전용 (Git 커밋 금지)

**환경변수 예시**
- `OPENAI_API_KEY` 또는 LLM API Key
- `QDRANT_URL`, `QDRANT_API_KEY`
- `DATA_DIR`, `PERSIST_DIR`

---

## 8) Pinned Dependencies — 버전 고정

**목적**  
의존성 변경으로 인한 예기치 않은 오류를 방지합니다.

**추천 방식**
- `requirements.txt` : 정확한 버전 고정
- `requirements-dev.txt` : 테스트 / 포맷터 / 린터
- 또는 `pyproject.toml` + lock 파일

---

## 📁 전체 디렉터리 트리

```text
project/
├─ config/
│  ├─ local.yaml
│  └─ prod.yaml
├─ data/
│  ├─ raw/
│  ├─ preprocessed/
│  ├─ features/
│  ├─ index/
│  ├─ predictions/
│  └─ eval/
├─ entrypoint/
│  ├─ train.py
│  ├─ inference.py
│  └─ evaluate.py
├─ notebooks/
├─ src/
│  ├─ common/
│  ├─ ingest/
│  ├─ chunking/
│  ├─ indexing/
│  ├─ retrieval/
│  ├─ generation/
│  ├─ eval/
│  └─ api/
├─ tests/
├─ docker/
│  ├─ Dockerfile
│  └─ docker-compose.yml
├─ .env.example
├─ requirements.txt
└─ README.md