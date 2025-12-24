import os
import glob
import re
import olefile
import zlib
import struct
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from dotenv import load_dotenv

# --- [1. 설정] ---
# .env 파일 활성화 (환경변수 로드)
load_dotenv()

# 키가 잘 로드됐는지 확인 
if not os.getenv("OPENAI_API_KEY"):
    print("에러: .env 파일이 없거나 키가 설정되지 않았습니다!")
    exit()

# 데이터 경로 (구조에 맞게 수정됨)
DATA_DIR = "./data/01-raw"
# DB 저장 경로
DB_PATH = "./rfp_database"

# --- [2. 함수 정의] ---
def clean_text(text):
    # 특수문자, 제어문자 제거
    pattern = r"[^가-힣a-zA-Z0-9\s\.,\-\(\)\[\]\<\>\'\"\/%·~ㆍ]"
    text = re.sub(pattern, "", text)
    text = text.replace("\x0b", " ").replace("\x1f", " ")
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def get_hwp_text(filename):
    try:
        f = olefile.OleFileIO(filename)
        dirs = f.listdir()
        if not any(d[0] == "BodyText" for d in dirs): return ""
        
        nums = []
        for d in dirs:
            if d[0] == "BodyText":
                try: nums.append(int(d[1].replace("Section", "")))
                except: pass
        nums.sort()
        
        header = f.openstream("FileHeader")
        is_compressed = (header.read()[36] & 1) == 1
        
        text = ""
        for i in nums:
            b_data = f.openstream(f"BodyText/Section{i}").read()
            if is_compressed: b_data = zlib.decompress(b_data, -15)
            
            i = 0
            while i < len(b_data):
                header = struct.unpack_from("<I", b_data, i)[0]
                rec_len = (header >> 20) & 0xfff
                if (header & 0x3ff) == 67:
                    rec_payload = b_data[i+4:i+4+rec_len]
                    text += rec_payload.decode('utf-16-le', errors='ignore') + "\n"
                i += 4 + rec_len
        return clean_text(text)
    except: return ""

def get_pdf_text(filename):
    try:
        reader = PdfReader(filename)
        text = "".join([page.extract_text() for page in reader.pages])
        return clean_text(text)
    except: return ""

# --- [3. 메인 실행] ---
print(f"'{DATA_DIR}' 폴더의 데이터를 처리합니다...")

docs = []
# 모든 파일 탐색
files = glob.glob(os.path.join(DATA_DIR, "*.*"))

for f in files:
    ext = f.split('.')[-1].lower()
    content = ""
    
    if ext == 'hwp':
        content = get_hwp_text(f)
    elif ext == 'pdf':
        content = get_pdf_text(f)
    else:
        # csv나 zip 같은 건 일단 건너뜀
        print(f"건너뜀 (지원하지 않는 형식): {os.path.basename(f)}")
        continue
        
    if content:
        docs.append(Document(page_content=content, metadata={"source": f}))
        print(f"읽기 성공: {os.path.basename(f)}")

if not docs:
    print("처리할 문서가 없습니다! data/01-raw 폴더에 파일이 있는지 확인하세요.")
    exit()

print(f"\n총 {len(docs)}개 문서 로드 완료. 벡터 DB 생성 시작...")

# 청킹 & 저장
splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
chunks = splitter.split_documents(docs)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
# 저장
vectordb = Chroma.from_documents(chunks, embeddings, persist_directory=DB_PATH)

print(f"모든 작업 완료! DB가 '{DB_PATH}'에 저장되었습니다.")