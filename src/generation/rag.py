from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import torch
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

class RAGChain:
    def __init__(self, config, vector_store_wrapper):
        self.config = config
        self.vector_store_wrapper = vector_store_wrapper
        
        # [안전장치] 설정값이 없으면 기본 모델(gpt-5-mini) 사용
        llm_cfg = config.get('llm', {})
        model_name = llm_cfg.get('model', 'gpt-5-mini')
        
        # self.llm = ChatOpenAI(
        #     model=model_name,
        #     temperature=0
        # )
        self.llm = get_huggingface_llm()

        self.prompt = ChatPromptTemplate.from_template("""
        당신은 RFP(제안요청서) 분석 전문가입니다. 
        아래 제공된 [Context]를 바탕으로 질문에 대해 명확하고 구체적으로 답변해 주세요.
        답변 시, 근거가 되는 내용이 어느 문서의 어떤 부분인지 참고하여 답변하세요.

        [Context]
        {context}

        [Question]
        {question}

        [Answer]:
        """)

    def generate_answer(self, question, selected_docs=None):
        # 1. 검색 필터 및 K값 설정 (동적 할당)
        # 기본은 5개만 가져오지만...
        search_kwargs = {"k": 3}
        
        if selected_docs:
             # [핵심 수정] 사용자가 문서를 '콕 집었을 때'는 경로를 맞춰주고
             # 검색 개수(k)를 30개까지 확 늘려서 앞/뒤 내용을 다 긁어오게 합니다.
             full_paths = [f"./data/01-raw/{doc}" for doc in selected_docs]
             search_kwargs["filter"] = {"source": {"$in": full_paths}}

             ##search_kwargs["k"] = 30  # <--- 문서를 지정했으면 30페이지 정도는 읽어봐야 정확함!
             # [수정] k=30은 너무 많아 에러를 유발합니다. 
             # Llama-3-8B의 컨텍스트 창(8k)을 고려해 10~15 정도로 타협합니다.
             search_kwargs["k"] = 10

        # [cite_start]2. 동적 검색기 생성 [cite: 25]
        retriever = self.vector_store_wrapper.vector_store.as_retriever(
            search_kwargs=search_kwargs
        )
        
        # 3. 문서 검색 및 답변 생성
        docs = retriever.invoke(question)
        
        def format_docs(documents):
            # 뒤에 붙어있던 # 제거
            return "\n\n".join([d.page_content for d in documents])
        chain = self.prompt | self.llm | StrOutputParser()
        
        answer = chain.invoke({
            "context": format_docs(docs),
            "question": question
        })
        
        return answer, docs


# -----------------------------------------------------------------------------
# 1. LLM 로드 함수 (GCP GPU 사용)
# -----------------------------------------------------------------------------
def get_huggingface_llm(model_id="beomi/Llama-3-Open-Ko-8B"):
    print(f"🔄 LLM 모델 로드 중... ({model_id})")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,  # 메모리 최적화
            device_map="auto",          # GPU 자동 할당
            trust_remote_code=True
        )
        
        # 텍스트 생성 파이프라인
        hf_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.1,
            top_p=0.95,
            repetition_penalty=1.1,
            return_full_text=False
        )
        print("✅ LLM 모델 로드 완료!")
        return HuggingFacePipeline(pipeline=hf_pipeline)
        
    except Exception as e:
        print(f"❌ LLM 로드 실패: {e}")
        # 테스트를 위해 CPU로 강제 실행하거나 종료할 수 있음
        raise e        