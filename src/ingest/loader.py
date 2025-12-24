import os
import subprocess
from typing import List, Dict, Any
from abc import ABC, abstractmethod
import pypdf

from langchain_core.documents import Document

class BaseLoader(ABC):
    def __init__(self, file_path: str):
        self.file_path = file_path

    @abstractmethod
    def load(self) -> List[Document]:
        """문서를 로드하여 Document 객체 리스트를 반환합니다."""
        pass

class PDFLoader(BaseLoader):
    """PDF 문서를 로드하는 클래스"""
    def load(self) -> List[Document]:
        try:
            reader = pypdf.PdfReader(self.file_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return [Document(page_content=text, metadata={"source": self.file_path, "type": "pdf"})]
        except Exception as e:
            print(f"PDF 로드 중 오류 발생 ({self.file_path}): {e}")
            return []

class HWPLoader(BaseLoader):
    """HWP(한글) 문서를 로드하는 클래스"""
    def load(self) -> List[Document]:
        try:
            # pyhwp 라이브러리의 hwp5txt 커맨드라인 도구 사용
            # 가상환경(venv) 혹은 시스템 PATH에 hwp5txt가 존재해야 함
            command = ["hwp5txt", self.file_path]
            
            # 외부 프로세스로 실행하여 텍스트 추출 (호환성 및 안정성 확보)
            result = subprocess.run(
                command, 
                capture_output=True, 
                text=True, 
                encoding="utf-8",
                check=False
            )
            
            if result.returncode != 0:
                print(f"hwp5txt 실행 실패 ({os.path.basename(self.file_path)}): {result.stderr[:200]}")
                return []
            
            text = result.stdout
            if not text.strip():
                 print(f"경고: 추출된 텍스트가 비어있습니다 ({self.file_path})")
            
            # [Fix] 파일명도 검색되도록 내용에 포함 (파일명: ... \n\n 내용...)
            filename = os.path.basename(self.file_path)
            enhanced_text = f"파일명: {filename}\n\n{text}"

            return [Document(page_content=enhanced_text, metadata={"source": self.file_path, "type": "hwp"})]
        
        except FileNotFoundError:
             print(f"오류: '{self.file_path}' 처리 중 'hwp5txt' 명령어를 찾을 수 없습니다. pyhwp가 설치되었는지 확인해주세요.")
             return []
        except Exception as e:
             print(f"HWP 로드 중 오류 발생 ({self.file_path}): {e}")
             return []

class TextLoader(BaseLoader):
    """일반 텍스트(.txt, .md) 파일을 로드하는 클래스"""
    def load(self) -> List[Document]:
        with open(self.file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return [Document(page_content=content, metadata={"source": self.file_path, "type": "text"})]

def get_loader(file_path: str) -> BaseLoader:
    """파일 확장자에 맞는 로더를 반환하는 팩토리 함수"""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return PDFLoader(file_path)
    elif ext == ".hwp":
        return HWPLoader(file_path)
    elif ext in [".txt", ".md"]:
        return TextLoader(file_path)
    else:
        # 지원하지 않는 형식일 경우 예외 발생
        raise ValueError(f"지원하지 않는 파일 형식입니다: {ext}")
