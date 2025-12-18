import unittest
from src.common.config import ConfigLoader
from src.ingest.loader import get_loader, Document
from src.chunking.splitter import TextSplitter

class TestBasic(unittest.TestCase):
    def test_config_loader(self):
        # 설정 파일 로더 테스트
        loader = ConfigLoader("config/local.yaml")
        self.assertIsNotNone(loader.config)
        self.assertEqual(loader.get("project_name"), "bid_mate_rag")

    def test_text_loader(self):
        # 더미 텍스트 파일 생성
        with open("test_dummy.txt", "w") as f:
            f.write("Hello World")
        
        try:
            loader = get_loader("test_dummy.txt")
            docs = loader.load()
            self.assertEqual(len(docs), 1)
            self.assertEqual(docs[0].page_content, "Hello World") # content -> page_content 수정 반영
        finally:
            import os
            if os.path.exists("test_dummy.txt"):
                os.remove("test_dummy.txt")

    def test_splitter(self):
        # 텍스트 분할기 테스트
        splitter = TextSplitter(chunk_size=10, chunk_overlap=0)
        docs = [Document(page_content="Short text", metadata={})] # content -> page_content
        chunks = splitter.split_documents(docs)
        self.assertEqual(len(chunks), 1)

if __name__ == '__main__':
    unittest.main()
