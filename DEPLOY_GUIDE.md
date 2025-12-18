# 🚀 Streamlit Cloud 배포 가이드 (Deployment Guide)

내일 발표를 위해 만든 프로젝트를 **Streamlit Community Cloud**에 무료로 쉽게 배포하는 방법입니다.

## 📋 1. 사전 준비 (완료됨)
이미 다음 작업들은 프로젝트 내에서 미리 처리해 두었습니다.
1.  **`requirements.txt` 최적화**: 배포에 필요한 패키지만 남기고 무거운 라이브러리는 정리했습니다.
2.  **ChromaDB 호환성 패치**: 클라우드 환경에서 DB 오류가 나지 않도록 `app.py`에 코드를 추가했습니다. (`pysqlite3` 적용)

---

## ☁️ 2. GitHub 설정 (상황별 선택)
현재 팀 저장소(Repository)를 사용 중이라면 아래 두 가지 방법 중 하나를 선택하세요.

### 방법 A: 팀 저장소 그대로 사용하기 (가장 간편)
1.  **[Streamlit Cloud](https://share.streamlit.io/) 접속**.
2.  **New app** -> **Use existing repo** 선택.
3.  저장소 목록에 **현재 팀 저장소**가 보이는지 확인하세요.
4.  보인다면 선택하고, **Branch**를 **`내_브랜치_이름`**(예: `feature/rag-test`)으로 설정하세요.
    *   *주의: 팀 조직(Organization) 설정에 따라 Streamlit 접근이 차단되어 있을 수 있습니다. 목록에 안 보이면 '방법 B'를 쓰세요.*

### 방법 B: 내 계정으로 가져오기 (Fork) - **추천** 👍
권한 문제 없이 가장 확실하게 배포하는 방법입니다.
1.  GitHub 팀 저장소 페이지 우측 상단의 **'Fork'** 버튼 클릭.
2.  **Owner**를 '내 계정'으로 선택하고 **Create fork** 클릭.
3.  이제 코드가 **내 계정의 저장소**로 복사되었습니다.
4.  Streamlit Cloud에서 **내 계정의 저장소**를 선택해 배포합니다.

---

## 🚀 3. Streamlit Cloud 배포 단계
1.  **Deploy an app** 설정:
    *   **Repository**: 사용할 저장소 선택
    *   **Branch**: 작업한 브랜치 이름 (Main 혹은 내 브랜치)
    *   **Main file path**: `app.py`
2.  **'Advanced settings' (중요! API 키 설정)** 🔑
    *   설정 창 하단이나 배포 후 Settings에서 **Secrets** 메뉴를 찾습니다.
    *   `.env` 파일 내용을 복사하여 붙여넣기 하거나 직접 입력합니다.
    ```toml
    OPENAI_API_KEY = "sk-..."
    ```
    *   *(주의: Streamlit Secrets 형식은 TOML 입니다. `KEY = "VALUE"` 형태)*
3.  **'Deploy!'** 버튼 클릭. 🎈

---

## 🛠️ 4. 문제 해결 (Troubleshooting)

**Q. "ModuleNotFoundError" 오류가 나요.**
*   `requirements.txt`가 저장소에 잘 올라갔는지 확인하세요.

**Q. "sqlite3 version too old" 오류가 나요.**
*   제가 `app.py`에 패치 코드를 넣어두어서 괜찮을 겁니다. 여전히 문제라면 `requirements.txt`에 `pysqlite3-binary`가 있는지 확인하세요.

**Q. 챗봇이 문서를 못 찾아요.**
*   GitHub에 `data/files` 폴더와 HWP 파일들이 올라갔는지 확인해 보세요. (보안이나 용량 문제로 `.gitignore` 되어 있을 수 있습니다.)

**발표 화이팅하세요! 🎉**
