# 🌐 API 가이드 (API Guide)

RFP RAG 시스템을 REST API로 사용하는 방법을 설명합니다.

---

## 🚀 서버 실행

### 기본 실행

```bash
# API 서버 시작
python entrypoint/api_server.py

# 또는 직접 uvicorn 사용
uvicorn src.api.app:app --host 0.0.0.0 --port 8001
```

### 개발 모드 (자동 리로드)

```bash
python entrypoint/api_server.py --reload
```

### 포트 변경

```bash
python entrypoint/api_server.py --host 0.0.0.0 --port 8081
```

---

## 📚 API 문서

서버 실행 후 브라우저에서 접속:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## 🔌 API 엔드포인트

### 1. Health Check

**GET** `/health`

서버 상태 확인

**응답 예시:**

```json
{
  "status": "healthy",
  "agents_initialized": true
}
```

---

### 2. 검색 (Search)

**POST** `/api/search`

문서 검색

**요청:**

```json
{
  "query": "교육 관련 사업",
  "top_k": 10,
  "filters": null,
  "use_hybrid": false,
  "use_rerank": true
}
```

**응답:**

```json
{
  "query": "교육 관련 사업",
  "results": [
    {
      "chunk_id": "doc1_0",
      "doc_id": "doc1",
      "chunk_text": "...",
      "score": 0.85,
      "metadata": {...}
    }
  ],
  "total_found": 10,
  "search_time": 0.123
}
```

**cURL 예시:**

```bash
curl -X POST "http://localhost:8000/api/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "교육 관련 사업",
    "top_k": 10
  }'
```

---

### 3. 질문 답변 (Q&A)

**POST** `/api/qa`

질문에 대한 답변 생성

**요청:**

```json
{
  "query": "이 사업의 예산은 얼마인가요?"
}
```

**응답:**

```json
{
  "answer": "이 사업의 예산은 1억 2천만원입니다...",
  "sources": [
    {
      "chunk_id": "doc1_0",
      "doc_id": "doc1",
      "chunk_text": "...",
      "score": 0.92,
      "metadata": {...}
    }
  ],
  "confidence": "high",
  "query": "이 사업의 예산은 얼마인가요?"
}
```

**cURL 예시:**

```bash
curl -X POST "http://localhost:8000/api/qa" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "이 사업의 예산은 얼마인가요?"
  }'
```

---

### 4. 문서 요약 (Summarize)

**POST** `/api/summarize`

문서 요약 생성

**요청:**

```json
{
  "doc_id": "20241218257",
  "top_k": 20
}
```

**응답:**

```json
{
  "summary": "이 사업은...",
  "key_points": ["사업 개요: ...", "주요 요구사항: ..."],
  "budget": "1억 2천만원",
  "deadline": "2024-12-23",
  "requirements": ["자격 요건 1", "자격 요건 2"],
  "doc_id": "20241218257"
}
```

**cURL 예시:**

```bash
curl -X POST "http://localhost:8000/api/summarize" \
  -H "Content-Type: application/json" \
  -d '{
    "doc_id": "20241218257"
  }'
```

---

### 5. 정보 추출 (Extract)

**POST** `/api/extract`

구조화된 정보 추출

**요청:**

```json
{
  "doc_id": "20241218257",
  "schema": {
    "budget": {
      "type": "float",
      "description": "사업 예산 금액 (원)"
    },
    "deadline": {
      "type": "datetime",
      "description": "입찰 참여 마감일"
    }
  }
}
```

**응답:**

```json
{
  "extracted_info": {
    "budget": 120000000.0,
    "deadline": "2024-12-23 10:00:00",
    "submission_method": "전자입찰",
    "required_qualifications": ["자격1", "자격2"]
  },
  "doc_id": "20241218257"
}
```

**cURL 예시:**

```bash
curl -X POST "http://localhost:8000/api/extract" \
  -H "Content-Type: application/json" \
  -d '{
    "doc_id": "20241218257"
  }'
```

---

## 🐍 Python 클라이언트 예시

```python
import requests

BASE_URL = "http://localhost:8000"

# 검색
response = requests.post(
    f"{BASE_URL}/api/search",
    json={"query": "교육 관련 사업", "top_k": 10}
)
results = response.json()
print(results)

# Q&A
response = requests.post(
    f"{BASE_URL}/api/qa",
    json={"query": "이 사업의 예산은 얼마인가요?"}
)
answer = response.json()
print(answer["answer"])

# 문서 요약
response = requests.post(
    f"{BASE_URL}/api/summarize",
    json={"doc_id": "20241218257"}
)
summary = response.json()
print(summary["summary"])
```

---

## 🔒 보안 고려사항

### 프로덕션 환경

1. **CORS 설정**: `app.py`에서 `allow_origins`를 특정 도메인으로 제한
2. **인증 추가**: API 키 또는 JWT 토큰 인증
3. **Rate Limiting**: 요청 제한 추가
4. **HTTPS**: SSL/TLS 사용

### 예시: API 키 인증 추가

```python
from fastapi import Header, HTTPException

API_KEY = "your-secret-api-key"

@app.post("/api/qa")
async def qa(request: QARequest, x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    # ... rest of the code
```

---

## 📊 모니터링

### 로그 확인

API 서버는 자동으로 로그를 출력합니다:

- 요청/응답 로그
- 에러 로그
- 성능 메트릭

### 헬스 체크

```bash
curl http://localhost:8000/health
```

---

## 🚀 배포

### Docker 사용 (예시)

```dockerfile
FROM python:3.9

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### systemd 서비스 (Linux)

```ini
[Unit]
Description=RFP RAG API
After=network.target

[Service]
User=your_user
WorkingDirectory=/path/to/codeit_ai_middle_project
ExecStart=/path/to/venv/bin/python entrypoint/api_server.py
Restart=always

[Install]
WantedBy=multi-user.target
```

---

## 📝 참고

- **API 문서**: http://localhost:8000/docs
- **실행 가이드**: [EXECUTION_GUIDE.md](./EXECUTION_GUIDE.md)
- **프로젝트 개요**: [README.md](./README.md)

---

**참고**: API 서버를 실행하기 전에 인덱싱이 완료되어 있어야 합니다.
