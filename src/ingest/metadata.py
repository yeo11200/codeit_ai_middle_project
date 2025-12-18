from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from src.common.config import config

class RFPMetadata(BaseModel):
    """RFP 문서에서 추출된 핵심 정보입니다."""
    project_name: Optional[str] = Field(None, description="프로젝트 또는 입찰 공고의 공식 명칭입니다.")
    organization: Optional[str] = Field(None, description="공고를 발주한 기관명입니다 (예: 공공기관, 기업 등).")
    budget: Optional[str] = Field(None, description="사업 예산 또는 추정 가격입니다 (예: '1억원'). 단위를 포함하세요.")
    period: Optional[str] = Field(None, description="사업 기간입니다 (예: '5개월', '2024-12-31까지').")
    deadline: Optional[str] = Field(None, description="제안서 제출 마감 기한입니다 (가능하면 YYYY-MM-DD HH:MM 형식).")

class MetadataExtractor:
    def __init__(self):
        llm_name = config.get("model.llm_name", "gpt-5")
        # 구조화된 출력(JSON)을 지원하는 모델 사용
        self.llm = ChatOpenAI(model=llm_name, temperature=0)
        
        # 정보 추출 체인 정의
        # LangChain의 with_structured_output 기능을 사용하여 Pydantic 모델로 바로 변환
        if hasattr(self.llm, "with_structured_output"):
             self.extract_chain = self.llm.with_structured_output(RFPMetadata)
        else:
             # 구버전 LangChain이거나 모델이 지원하지 않는 경우 예외 처리 필요
             pass 

    def extract(self, text: str) -> RFPMetadata:
        """
        제공된 텍스트(주로 문서의 앞부분)에서 메타데이터를 추출합니다.
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", "당신은 제안요청서(RFP) 분석 전문가입니다. 제공된 텍스트에서 다음 핵심 정보를 추출하세요. 항목을 찾을 수 없으면 null로 설정하세요."),
            ("human", "{text}")
        ])
        
        chain = prompt | self.extract_chain
        
        try:
            # 토큰 제한을 피하기 위해 입력 텍스트 길이를 제한 (보통 앞 10000자 내에 주요 정보가 있음)
            input_text = text[:10000] 
            return chain.invoke({"text": input_text})
        except Exception as e:
            print(f"메타데이터 추출 중 오류 발생: {e}")
            return RFPMetadata()
