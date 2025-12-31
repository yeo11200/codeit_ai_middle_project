import os

# 설정: 결과를 저장할 파일명
OUTPUT_FILE = "project_context.txt"

# 설정: 무시할 폴더 이름들 (여기에 data, venv 등을 추가하세요)
IGNORE_DIRS = {'.git', '.venv', '__pycache__', '.ipynb_checkpoints', 'rfp_database', 'data', 'rfp_database_v1'}

# 설정: 포함할 파일 확장자 (필요한 것만 골라서)
INCLUDE_EXTS = {'.py', '.md', '.txt', '.yml', '.yaml', '.json', 'Makefile', 'Dockerfile'}

def main():
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as outfile:
        # 헤더 작성
        outfile.write(f"# Project Export\n# Created by script\n\n")
        
        for root, dirs, files in os.walk("."):
            # 무시할 폴더는 탐색 리스트에서 제거 (하위 폴더까지 싹 무시됨)
            dirs[:] = [d for d in dirs if d not in IGNORE_DIRS and not d.startswith('rfp_database')]
            
            for file in files:
                file_ext = os.path.splitext(file)[1]
                # Makefile 처럼 확장자 없는 파일도 포함하고 싶으면 조건 추가
                if file_ext in INCLUDE_EXTS or file in ['Makefile', '.gitignore', 'requirements.txt']:
                    file_path = os.path.join(root, file)
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            
                        # 출력 포맷: 파일 경로와 내용을 구분해서 기록
                        outfile.write(f"\n{'='*50}\n")
                        outfile.write(f"File: {file_path}\n")
                        outfile.write(f"{'='*50}\n")
                        outfile.write(content + "\n")
                        print(f"Included: {file_path}")
                        
                    except Exception as e:
                        print(f"Skipped (Error): {file_path} - {e}")

    print(f"\n[완료] 모든 코드가 '{OUTPUT_FILE}'에 저장되었습니다.")

if __name__ == "__main__":
    main()