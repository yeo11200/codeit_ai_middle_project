# Makefile for BidMate RAG System
# 작성일: 2024-12-31

# --- [변수 설정] ---
# 가상환경 내의 실행 파일 경로
PYTHON = .venv/bin/python
PIP = .venv/bin/pip
STREAMLIT = .venv/bin/streamlit

# 데이터 경로 설정
# (주의) 서버 환경에 따라 DATA_SOURCE 경로가 맞는지 확인 필요
DATA_SOURCE = /home/soobeom/shared_data/rfp_raw/data/files
DATA_TARGET_DIR = data
DATA_TARGET_LINK = $(DATA_TARGET_DIR)/01-raw

# .PHONY: 파일 이름과 충돌하지 않도록 명령어임을 명시
.PHONY: help setup venv link_data install_req db app run clean

# --- [기본 명령어] ---
help:
	@echo "🤖 [BidMate] RAG 분석기 자동화 명령어"
	@echo "------------------------------------------------------------------"
	@echo " make run    : [추천] 설치부터 실행까지 한 방에 (Setup + DB + App)"
	@echo " make app    : 앱 실행 (이미 설치된 경우)"
	@echo " make db     : 벡터 데이터베이스 재생성 (rfp_database_*)"
	@echo " make clean  : 프로젝트 초기화 (가상환경 및 DB 삭제)"
	@echo "------------------------------------------------------------------"

# --- 1. 환경 셋업 (순서: 가상환경 -> 데이터연결 -> 패키지설치) ---
setup: venv link_data install_req
	@echo "✅ [설정 완료] 모든 준비가 끝났습니다."

venv:
	@echo "🐍 가상환경(.venv) 생성 중..."
	@test -d .venv || python3 -m venv .venv

link_data:
	@echo "🔗 데이터 폴더 연결 중..."
	@mkdir -p $(DATA_TARGET_DIR)
	@# 기존 링크가 있으면 삭제 후 다시 연결 (경로 꼬임 방지)
	@rm -rf $(DATA_TARGET_LINK)
	@if [ -d "$(DATA_SOURCE)" ]; then \
		ln -s $(DATA_SOURCE) $(DATA_TARGET_LINK); \
		echo "   -> 연결 성공: $(DATA_TARGET_LINK)"; \
	else \
		echo "⚠️ [경고] 원본 데이터 경로($(DATA_SOURCE))가 없습니다. 경로를 확인하세요."; \
	fi

install_req:
	@echo "📦 라이브러리 설치 중 (시간이 좀 걸릴 수 있습니다)..."
	@$(PIP) install --upgrade pip > /dev/null
	@$(PIP) install -r requirements.txt > /dev/null
	@echo "   -> 설치 완료!"

# --- 2. DB 생성 ---
db:
	@echo "📚 벡터 DB 구축 시작 (bge-m3 / kure-v1)..."
	@# main.py 실행 시 두 개의 DB 폴더가 생성됨
	@$(PYTHON) scripts/main.py

# --- 3. 앱 실행 ---
app:
	@echo "🧹 파이썬 캐시(__pycache__) 제거 중..."
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@echo "🚀 Streamlit 앱을 깨끗한 상태로 실행합니다..."
	@$(STREAMLIT) run app.py --server.runOnSave true

# --- 4. 원클릭 실행 ---
run: setup db app

# --- 5. 청소 (초기화) ---
clean:
	@echo "🧹 프로젝트 정리(초기화) 중..."
	@rm -rf .venv
	@# 생성된 DB 폴더들(rfp_database_bge, rfp_database_kure 등) 모두 삭제
	@rm -rf rfp_database*
	@rm -rf $(DATA_TARGET_DIR)
	@# 캐시 파일들도 삭제
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@echo "✨ 깨끗하게 지워졌습니다. 다시 시작하려면 'make run'을 입력하세요."