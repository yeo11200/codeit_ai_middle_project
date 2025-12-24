VENV = .venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip
STREAMLIT = $(VENV)/bin/streamlit

# 가상환경 생성 및 의존성 설치 (requirements.txt가 변경되면 다시 실행됨)
$(VENV)/bin/activate: requirements.txt
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	touch $(VENV)/bin/activate

# 앱 실행 (가상환경이 없으면 자동으로 만듦)
run: $(VENV)/bin/activate
	$(STREAMLIT) run app.py

# 청소
clean:
	rm -rf $(VENV)
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
