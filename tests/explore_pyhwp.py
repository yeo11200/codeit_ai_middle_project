try:
    from hwp5.hwp5txt import Hwp5Txt
    print("Hwp5Txt 모듈 임포트 성공")
except ImportError:
    print("Hwp5Txt 임포트 실패. 대체 방법 시도 중...")

import sys
import os

# 더미 HWP 파일 생성이 어려우므로, 라이브러리 접근 가능 여부만 주로 확인하는 스크립트입니다.
# 실제 파일이 있다면 해당 파일로 테스트할 수 있도록 함수 구조만 잡아둡니다.

def test_extraction(file_path):
    try:
        # hwp5txt는 주로 커맨드라인 도구로 사용되거나 파일 객체가 필요합니다.
        # 여기서는 단순히 API 확인용으로 남겨둡니다.
        pass
    except Exception as e:
        print(e)

if __name__ == "__main__":
    # 라이브러리 접근 가능성 확인
    try:
        import hwp5.utils
        print("hwp5.utils 임포트 확인됨")
    except ImportError as e:
        print(f"hwp5 임포트 오류: {e}")
