from typing import List
# from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

class TextSplitter:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # self.splitter = RecursiveCharacterTextSplitter(
        #     chunk_size=chunk_size,
        #     chunk_overlap=chunk_overlap
        # )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        문서 리스트를 더 작은 청크(Chunk) 단위로 분할합니다.
        
        Args:
            documents (List[Document]): 분할할 원본 문서 리스트
            
        Returns:
            List[Document]: 분할된 문서 리스트
        """
        # 디버깅을 위해 현재 설정된 청크 크기와 중복 구간을 출력합니다.
        print(f"문서 {len(documents)}개를 분할합니다 (크기={self.chunk_size}, 중복={self.chunk_overlap})")
        
        # 실제 구현에서는 RecursiveCharacterTextSplitter 등을 사용하여 텍스트를 자릅니다.
        # 현재는 프로젝트 구조상 전체 문서를 하나의 청크로 처리하는 임시 로직이 적용되어 있습니다.
        # (필요 시 주석 해제하여 실제 분할 로직 적용 가능)
        
        chunked_docs = []
        for doc in documents:
            # 원본 문서를 그대로 유지 (Skeleton 코드 특성상 단순화됨)
            chunked_docs.append(doc)
            
        return chunked_docs
