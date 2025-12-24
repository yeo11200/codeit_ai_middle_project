import subprocess
import os

file_path = "data/files/국가과학기술지식정보서비스_통합정보시스템 고도화 용역.hwp"

# 파일 존재 여부 확인
if not os.path.exists(file_path):
    print(f"파일을 찾을 수 없습니다: {file_path}")
    # 대체 HWP 파일 찾기
    files = [f for f in os.listdir("data/files") if f.endswith(".hwp")]
    if files:
        file_path = os.path.join("data/files", files[0])
        print(f"대체 파일 사용: {file_path}")
    else:
        print("HWP 파일을 찾을 수 없습니다.")
        exit(1)

try:
    print(f"{file_path}에서 텍스트 추출 확인 중...")
    # venv 환경 내의 hwp5txt 실행 경로 (필요 시 수정)
    # 테스트를 위해 절대 경로를 사용하거나 PATH에 있는 'hwp5txt'를 바로 호출
    result = subprocess.run(
        ["/home/soobeom/SB-venv/bin/hwp5txt", file_path],
        capture_output=True,
        text=True,
        encoding="utf-8"
    )
    
    if result.returncode == 0:
        print("성공!")
        print("미리보기:")
        print(result.stdout[:500])
    else:
        print("실패 (Return Code):", result.returncode)
        print("Stderr:", result.stderr)
        
except Exception as e:
    print(f"예외 발생: {e}")
