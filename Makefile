# Makefile for BidMate RAG System

# --- 변수 설정 ---
PYTHON = .venv/bin/python
PIP = .venv/bin/pip
STREAMLIT = .venv/bin/streamlit
DATA_SOURCE = /home/soobeom/shared_data/rfp_raw/data/files
DATA_TARGET = data/01-raw

# --- 기본 명령어 ---
help:
	@echo "🤖 입찰메이트 자동화 명령어 모음 🤖"
	@echo "make run    : [강추] 초기세팅 + DB생성 + 앱실행 (All-in-One)"
	@echo "make clean  : 프로젝트 초기화 (삭제)"

# --- 1. 환경 셋업 ---
setup: venv link_data install_req
	@echo "✅ 기초 환경 설정 완료"

venv:
	@test -d .venv || python3 -m venv .venv

link_data:
	@mkdir -p data
	@rm -rf $(DATA_TARGET)
	@ln -s $(DATA_SOURCE) $(DATA_TARGET)
	@echo "🔗 데이터 연결 완료"

install_req:
	@$(PIP) install -r requirements.txt > /dev/null 2>&1
	@echo "📦 라이브러리 설치 완료"

# --- 2. DB 생성 ---
db:
	@echo "📚 벡터 DB 생성 중..."
	@$(PYTHON) scripts/main.py

# --- 3. 앱 실행 ---
app:
	@echo "🚀 Streamlit 앱 실행..."
	@$(STREAMLIT) run app.py

# --- 4. 한 방에 실행 ---
run: setup db app

# --- 5. 청소 ---
clean:
	@rm -rf .venv rfp_database data
	@echo "🧹 초기화 완료"
