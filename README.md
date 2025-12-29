

## 요약 

가장 빠르게 실행하는 방법입니다. 터미널에서 아래 3단계만 진행하세요.

**1. 프로젝트 폴더로 이동**
Github에서 클론(Clone)한 프로젝트 폴더 내부로 이동합니다.
(폴더명은 클론 시 설정에 따라 다를 수 있습니다.)

```bash
# 예시: cd mid_project 또는 cd [프로젝트폴더명]
cd mid_project

```

**2. API 키 설정 (필수)**
OpenAI API 키가 필요합니다. (`.env` 파일 생성)

```bash
# 아래 sk-proj-... 부분을 본인의 실제 키로 변경해서 입력하세요!
echo "OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxx" > .env

```

**3. 원클릭 실행 (자동화)**
환경 설정, 데이터 연결, 라이브러리 설치, DB 생성, 앱 실행을 한 번에 수행합니다.

```bash
make run

```

* **실행 성공 시:** 브라우저가 열리거나 터미널에 `External URL`이 표시됩니다.

---

## 📂 수동 설정 가이드 (Detailed Setup)

`make` 명령어가 작동하지 않거나, 단계별로 확인하고 싶다면 아래 절차를 따르세요.

### 1. 데이터 폴더 연결 (Data Linking)

**[중요]** 우리는 저장 공간 효율화를 위해 수범님의 공유 데이터를 참조합니다.
데이터를 복사하지 않고 `ln -s` 명령어(바로가기)를 사용합니다.

```bash
# 1. 데이터를 담을 폴더 생성
mkdir -p data

# 2. 공유 스토리지의 원본 파일(files)을 내 프로젝트(01-raw)로 연결
# (경로를 정확히 입력해야 합니다!)
ln -s /home/soobeom/shared_data/rfp_raw/data/files ./data/01-raw

# 3. 연결 확인 (파일 목록이 쭉 뜨면 성공)
ls -F ./data/01-raw/

```

### 2. 가상환경 및 라이브러리 설치

패키지 충돌 방지를 위해 가상환경을 사용합니다.

```bash
# 가상환경(.venv) 생성
python3 -m venv .venv

# 가상환경 활성화 (필수: 터미널 앞에 (.venv) 표시 확인)
source .venv/bin/activate

# 필수 라이브러리 설치
pip install -r requirements.txt

```

### 3. 벡터 DB 생성 (최초 1회)

문서를 읽어서 검색용 데이터베이스를 구축합니다.

```bash
python scripts/main.py

```

### 4. 앱 실행

```bash
streamlit run app.py

```

---

## 🕹️ 명령어 모음 (Makefile)

편리한 실행을 위해 `Makefile`을 구성해 두었습니다.

| 명령어 | 설명 | 비고 |
| --- | --- | --- |
| **`make run`** | **(추천)** 초기 셋업부터 실행까지 한 번에 수행 | `setup` + `db` + `app` |
| **`make app`** | **(재실행용)** 이미 세팅된 상태에서 웹 앱만 다시 켤 때 | `Ctrl+C`로 끈 후 사용 |
| `make db` | 문서 데이터가 변경되어 DB를 다시 만들어야 할 때 |  |
| `make clean` | 가상환경 및 DB를 삭제하여 프로젝트 초기화 | 에러가 심할 때 사용 |

---

## 🛠️ 자주 묻는 질문 (FAQ)

**Q. 앱을 껐다가 다시 켜려면 어떻게 해요?**
A. 터미널에서 `make app`을 입력하거나, 가상환경이 켜진 상태에서 `streamlit run app.py`를 입력하세요.

**Q. "처리할 문서가 없습니다" 에러가 떠요.**
A. 데이터 연결이 끊어진 경우입니다. `ls -l data/01-raw` 명령어로 파일이 보이는지 확인하고, 안 보이면 **1. 데이터 폴더 연결** 단계를 다시 수행하세요.

**Q. API 키가 없다고 나와요.**
A. `.env` 파일이 없거나 내용이 비어있는 경우입니다. `cat .env` 명령어로 파일 내용을 확인하세요.

---

## 📋 프로젝트 정보

* **Raw Data Path**: `/home/soobeom/shared_data/rfp_raw/data/files`
* **Python Version**: 3.8+ (Recommended)
* **Main Frameworks**: LangChain, ChromaDB, Streamlit, Ollama(Planned)

```

```