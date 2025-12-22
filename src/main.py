import os
import requests

from getpass import getpass

import torch
from tqdm import tqdm
#from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
#from transformers import pipeline

#from langchain_community.embeddings import HuggingFaceEmbeddings
#from langchain_community.vectorstores import FAISS
#from langchain_community.llms import HuggingFacePipeline
from langchain_community.document_loaders import PyPDFLoader, TextLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


import re


def main():
    # ==========================================
    # 0. 초기 설정
    # ==========================================

    # OpenAI API key
    openai_api_key = getpass("Enter your OpenAI API key: ")

    # API키 설정
    os.environ["OPENAI_API_KEY"] = openai_api_key

    print('You Entered OpenAI API key')

    documents = document_load()
    splits = document_chunk(documents)
    vectorstore = document_embed(splits)
    retriever, llm = make_retriever(vectorstore)
    prompt = make_prompt()
    rag_chain = make_chain(retriever, prompt, llm)
    make_question(rag_chain)


def document_load():
    # 1. 문서 로드
    pdf_path = "./(사)벤처기업협회_2024년 벤처확인종합관리시스템 기능 고도화 용역사업.pdf"
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # 2. 텍스트 전처리 (노이즈 제거)
    for doc in documents:
        # 페이지 번호, 과도한 공백, 목차 점 등을 제거
        doc.page_content = re.sub(r'^\s*-\s*\d+\s*-\s*$', '', doc.page_content, flags=re.MULTILINE)
        doc.page_content = re.sub(r'·{2,}', '', doc.page_content)
        doc.page_content = re.sub(r'\s+', ' ', doc.page_content).strip()

    return documents

def document_chunk(documents):
    # 3. 문서 분할 (Chunk Size 대폭 상향)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # 핵심 수정 사항
        chunk_overlap=200,  # 핵심 수정 사항
        separators=["\n\n", "\n", " ", ""]
    )

    splits = text_splitter.split_documents(documents)
    print(f"분할된 청크 개수: {len(splits)}")

    return splits

def document_embed(splits):
    # 4. 임베딩 및 벡터 저장소 생성
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=OpenAIEmbeddings(model="text-embedding-3-small")
    )

    return vectorstore

def make_retriever(vectorstore):
    # 5. 검색기 및 체인 생성 (k값 상향)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    return retriever, llm

def make_prompt():
    template = """
    당신은 벤처기업협회의 담당 AI입니다.
    아래 [규정]을 참고하여 질문에 친절하고 상세하게 답해주세요.
    문서에 없는 내용은 "문서에 나와있지 않습니다"라고 답하세요.

    [규정]
    {context}

    [질문]: {question}
    """
    prompt = PromptTemplate.from_template(template)

    return prompt

def make_chain(retriever, prompt, llm):
    rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )

    return rag_chain

def make_question(rag_chain):
    # 6. 질문 테스트
    ask = "벤처확인종합시스템 기능 고도화에 대해서 상세하게 알려 줘."
    print(ask)
    response = rag_chain.invoke(ask)
    print(response)

if __name__ == "__main__":
    main()