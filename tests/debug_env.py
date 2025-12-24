import sys
import os

print("Sys 경로:", sys.path)
import site
print("Site 패키지 경로:", site.getsitepackages())

# site-packages 내의 모든 폴더 목록 확인
for path in site.getsitepackages():
    if os.path.exists(path):
        print(f"경로 내 목록 확인 중: {path}:")
        try:
            items = os.listdir(path)
            for item in items:
                if 'hwp' in item:
                    print(f"  발견됨: {item}")
        except Exception as e:
            print(e)
