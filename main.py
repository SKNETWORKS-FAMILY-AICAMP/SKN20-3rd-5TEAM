# -*- coding: utf-8 -*-
"""
대피소 안내 챗봇 API 서버
FastAPI 기반 웹 API

=== 프로젝트 개요 ===
이 스크립트는 대피소 안내 챗봇의 백엔드 서버를 구현합니다.
FastAPI를 사용하여 REST API를 제공하며, 다음과 같은 주요 기능을 포함합니다:
1. 지명 추출 및 대피소 검색
2. RAG (Retrieval-Augmented Generation) 기반 재난행동요령 안내
3. 대피소 데이터 조회 (GPS 기반 근처 대피소)
4. 시스템 상태 확인

=== 사용 기술 스택 및 역할 ===

[백엔드 프레임워크]
- FastAPI: 비동기 웹 API 서버 프레임워크
  * REST API 엔드포인트 구현
  * Request/Response 데이터 검증 (Pydantic)
  * 자동 API 문서 생성 (Swagger UI)
  * CORS 미들웨어 설정
  * Lifespan 이벤트로 서버 초기화/종료 관리

[AI/LLM]
- OpenAI GPT-4o-mini: 자연어 처리 및 의도 분류
  * 사용자 질문 의도 분류 (대피소 검색 / 재난행동요령 / 일반 대화)
  * JSON 응답 형식으로 구조화된 결과 반환
  * Temperature=0 설정으로 일관된 분류 결과 보장

- OpenAI Embeddings (text-embedding-3-small): 텍스트 벡터화
  * 대피소 정보 및 재난행동요령 문서 임베딩
  * 의미 기반 유사도 검색 지원

[벡터 데이터베이스]
- ChromaDB: 벡터 저장소 및 유사도 검색
  * 대피소 메타데이터 저장 (시설명, 주소, 좌표, 수용인원)
  * 재난행동요령 문서 저장 및 검색
  * 메타데이터 필터링 (type: shelter / disaster_guideline)
  * Persist 디렉토리를 통한 영구 저장

[LangChain]
- langchain-chroma: ChromaDB와 LangChain 통합
  * Vector Store 추상화 계층 제공
  * 문서 검색 및 유사도 계산
  
- langchain-openai: OpenAI 모델 LangChain 통합
  * 임베딩 모델 래퍼 제공

[데이터 처리]
- Pandas: 대피소 CSV 데이터 처리
  * 대피소 정보 DataFrame 관리
  * 좌표 기반 필터링 및 정렬

[외부 API]
- 카카오 로컬 API (Kakao Local API): 장소 검색 및 좌표 변환
  * 키워드 검색을 통한 지명 → 위경도 좌표 변환
  * 카테고리 정보로 랜드마크 우선순위 판단
  * REST API 방식 (requests 라이브러리 사용)

[데이터 검증]
- Pydantic: Request/Response 데이터 모델 정의
  * LocationExtractRequest: 사용자 쿼리 입력
  * LocationExtractResponse: 대피소 검색 결과 반환
  * BaseModel 상속으로 자동 검증 및 직렬화

[환경 설정]
- python-dotenv: 환경 변수 관리
  * .env 파일에서 API 키 로드 (OPENAI_API_KEY, KAKAO_REST_API_KEY)
  * 민감 정보 소스코드 분리

[서버 실행]
- Uvicorn: ASGI 서버
  * FastAPI 앱 실행
  * Hot reload 지원 (개발 모드)
  * SSL/TLS 지원 (HTTPS 서버)

[거리 계산 알고리즘]
- Haversine Formula: 구면상의 두 점 사이 최단 거리 계산
  * 사용자 위치 ↔ 대피소 위치 간 직선 거리 (km)
  * 가장 가까운 대피소 5곳 추출

[주요 처리 흐름]
1. 사용자 쿼리 입력
2. LLM 기반 의도 분류 (find_shelter / disaster_guide / general_chat)
3-1. 대피소 검색: 카카오 API → 좌표 변환 → ChromaDB 메타데이터 → Haversine 거리 계산 → 정렬
3-2. 재난행동요령: ChromaDB 유사도 검색 → 관련 문서 반환
4. JSON 응답 반환

[프로젝트 구조]
- data_loaders: CSV/JSON 파일 로딩 모듈
- documents: 문서 변환 모듈 (DataFrame → LangChain Documents)
- embedding_and_vectordb: 임베딩 생성 및 ChromaDB 초기화 모듈
"""
import sys
from pathlib import Path
from typing import List, Dict, Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request, Body
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import os
import requests
from dotenv import load_dotenv

# -----------------------------------------------------------------------------
# 1. 환경 설정 및 경로 설정
# -----------------------------------------------------------------------------

# 프로젝트 루트 경로를 시스템 경로에 추가하여 모듈 import가 가능하도록 설정
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# .env 파일에서 환경 변수 로드 (API Key 등)
load_dotenv()

# 프로젝트 모듈 임포트
# data_loaders: 데이터 파일(csv, json) 로딩 유틸리티
# documents: 문서 변환 유틸리티
# embedding_and_vectordb: 벡터 DB 생성 및 관리
from data_loaders import load_shelter_csv, load_all_disaster_jsons
from documents import csv_to_documents, json_to_documents
from embedding_and_vectordb import create_embeddings_and_vectordb
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
import pandas as pd
import json
import re

# LangGraph 관련 import
from langchain_community.retrievers import BM25Retriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import tool
from typing import TypedDict, Annotated

# Simple EnsembleRetriever implementation
class EnsembleRetriever:
    """간단한 앙상블 리트리버 구현"""
    def __init__(self, retrievers, weights=None):
        self.retrievers = retrievers
        self.weights = weights or [1.0 / len(retrievers)] * len(retrievers)
    
    def invoke(self, query):
        all_docs = []
        for retriever, weight in zip(self.retrievers, self.weights):
            try:
                docs = retriever.invoke(query)
                for doc in docs:
                    doc.metadata['retriever_weight'] = weight
                    all_docs.append(doc)
            except:
                continue
        # 중복 제거 및 가중치 기반 정렬
        seen = set()
        unique_docs = []
        for doc in all_docs:
            doc_id = doc.page_content[:100]
            if doc_id not in seen:
                seen.add(doc_id)
                unique_docs.append(doc)
        return unique_docs[:10]

# OpenAI 클라이언트 초기화
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
except (ImportError, Exception) as e:
    print(f"[WARNING] OpenAI 클라이언트 초기화 실패: {e}")
    OPENAI_AVAILABLE = False
    openai_client = None

# -----------------------------------------------------------------------------
# 2. Pydantic 모델 정의 (Request/Response 스키마)
# -----------------------------------------------------------------------------

class LocationExtractRequest(BaseModel):
    query: str

class LocationExtractResponse(BaseModel):
    success: bool
    location: Optional[str] = None
    coordinates: Optional[tuple] = None
    shelters: List[Dict] = []
    total_count: int = 0
    message: str = ""

class ChatbotRequest(BaseModel):
    message: str
    session_id: Optional[str] = "default"

class ChatbotResponse(BaseModel):
    response: str
    session_id: str


# -----------------------------------------------------------------------------
# 2-1. 의도 분류 함수 (find_location.py의 llm_intent_classifier 참조)
# -----------------------------------------------------------------------------

def classify_user_intent(query: str) -> str:
    """
    사용자 질문의 의도를 분류합니다.
    
    Returns:
        "find_shelter": 대피소 찾기 의도
        "disaster_guide": 재난행동요령 질문
        "general_chat": 일반 대화
    """
    # 1차: 명확한 키워드가 있으면 바로 분류 (빠른 경로)
    if "대피소" in query or "피난" in query or "피난처" in query:
        print(f"  [의도분류] 대피소 키워드 발견 -> find_shelter")
        return "find_shelter"
    
    if not OPENAI_AVAILABLE or not openai_client:
        print("  [의도분류] OpenAI 사용 불가 - 키워드 기반 분류로 대체")
        return keyword_intent_classifier(query)
    
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """당신은 사용자 질문의 의도를 분류하는 전문가입니다.

사용자 질문을 세 가지 카테고리로 분류하세요:

1. **대피소 찾기 (find_shelter)**:
   - 대피소, 피난처, 안전한 장소를 찾는 질문
   - **지역명, 장소명, 건물명, 랜드마크만 입력된 경우 (매우 중요!)**
   - 주소, 동네, 구, 시, 역, 건물 등의 위치 정보
   - 예: "강남역 대피소", "서울역", "마포구", "잠실 롯데월드", "명동", "여의도", "근처 피난처"

2. **재난행동요령 (disaster_guide)**:
   - 재난 상황 대처 방법을 묻는 질문
   - 행동 요령, 대피 방법, 안전 수칙 문의
   - "어떻게", "방법", "대처", "행동요령" 등의 의문문
   - 예: "지진 났을 때 어떻게 해?", "화재 발생시 행동요령", "태풍 대비법"

3. **일반 대화 (general_chat)**:
   - 인사, 도움말, 사용법 문의
   - 대피소나 재난과 관련 없는 질문
   - 예: "안녕하세요", "도움말", "사용법"

**중요**: 지역명이나 장소명만 언급되면 무조건 find_shelter로 분류하세요!

응답 형식 (JSON):
{"intent": "find_shelter" 또는 "disaster_guide" 또는 "general_chat", "confidence": 0.0~1.0, "reason": "판단 이유"}"""
                },
                {
                    "role": "user",
                    "content": query
                }
            ],
            temperature=0,
            max_tokens=150,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        intent = result.get("intent", "general_chat")
        confidence = result.get("confidence", 0.0)
        reason = result.get("reason", "")
        
        print(f"  [의도분류] LLM 결과: {intent} (신뢰도: {confidence}, 이유: {reason})")
        
        # 신뢰도가 낮으면 키워드 기반으로 재확인
        if confidence < 0.6:
            print(f"  [의도분류] 신뢰도 낮음({confidence}) - 키워드 기반으로 재확인")
            return keyword_intent_classifier(query)
        
        return intent
        
    except Exception as e:
        print(f"  [의도분류] LLM 오류: {e} - 키워드 기반으로 대체")
        return keyword_intent_classifier(query)


def keyword_intent_classifier(query: str) -> str:
    """
    키워드 기반 의도 분류 (LLM 사용 불가 시 폴백)
    """
    print(f"  [키워드분류] 쿼리 분석: '{query}'")
    
    # 대피소 관련 키워드
    shelter_keywords = ["대피소", "대피", "피난", "피난처", "안전한 곳", "숨을 곳", "비상대피", "근처", "주변"]
    
    # 재난행동요령 관련 키워드
    disaster_keywords = [
        "지진", "화재", "태풍", "홍수", "산사태", "폭풍", "해일", "쓰나미", "tsunami",
        "행동요령", "대처법", "대처방법", "대비", "안전수칙", "어떻게", "방법", "해야",
        "화산", "방사능", "가스", "댐", "산불", "폭발", "분화", "낙뢰",
        "발생", "났을", "일어나", "생기면", "경우"
    ]
    
    # 한국 지역명 패턴
    location_pattern = r'(구|동|역|시|읍|면|리|로|길|대로)'
    
    # 일반 대화 키워드
    general_keywords = ["안녕", "도움말", "사용법", "설명", "뭐야", "날씨", "고마워", "감사"]
    
    # 매칭된 키워드 추적
    matched_shelter = [k for k in shelter_keywords if k in query]
    matched_disaster = [k for k in disaster_keywords if k in query]
    matched_general = [k for k in general_keywords if k in query]
    
    print(f"  [키워드분류] 대피소 키워드: {matched_shelter}")
    print(f"  [키워드분류] 재난 키워드: {matched_disaster}")
    print(f"  [키워드분류] 일반 키워드: {matched_general}")
    
    # 1. 일반 대화 먼저 확인
    if matched_general and not (matched_shelter or matched_disaster):
        print(f"  [키워드분류] 결과: general_chat")
        return "general_chat"
    
    # 2. 재난행동요령 확인
    if matched_disaster:
        # 단, 대피소 키워드도 함께 있으면 대피소 검색으로 간주
        if matched_shelter:
            print(f"  [키워드분류] 결과: find_shelter (재난+대피소)")
            return "find_shelter"
        print(f"  [키워드분류] 결과: disaster_guide")
        return "disaster_guide"
    
    # 3. 대피소 검색 확인
    if matched_shelter or re.search(location_pattern, query):
        print(f"  [키워드분류] 결과: find_shelter")
        return "find_shelter"
    
    # 4. 짧은 질문은 대피소 검색으로 간주 (지역명일 가능성)
    if len(query.strip()) <= 5:
        print(f"  [키워드분류] 결과: find_shelter (짧은 쿼리)")
        return "find_shelter"
    
    # 5. 기본값은 일반 대화
    print(f"  [키워드분류] 결과: general_chat (기본값)")
    return "general_chat"


# -----------------------------------------------------------------------------
# 2-2. LangGraph 하이브리드 검색 및 Agent 생성 함수
# -----------------------------------------------------------------------------

def create_hybrid_retrievers():
    """하이브리드 리트리버 생성 (Vector + BM25)"""
    if vectorstore is None:
        return None, None
    
    try:
        # 1. Vector Retriever
        shelter_vector_retriever = vectorstore.as_retriever(
            search_kwargs={"k": 5, "filter": {"type": "shelter"}}
        )
        guideline_vector_retriever = vectorstore.as_retriever(
            search_kwargs={"k": 3, "filter": {"type": "disaster_guideline"}}
        )
        
        # 2. BM25 Retriever 생성
        def create_bm25_retriever(doc_type: str):
            try:
                all_docs = vectorstore.get(where={"type": doc_type})
                if not all_docs or 'documents' not in all_docs:
                    return None
                
                documents = []
                for i, text in enumerate(all_docs['documents']):
                    metadata = all_docs['metadatas'][i] if 'metadatas' in all_docs else {}
                    documents.append(Document(page_content=text, metadata=metadata))
                
                bm25_retriever = BM25Retriever.from_documents(documents)
                bm25_retriever.k = 5
                return bm25_retriever
            except Exception as e:
                print(f"⚠️ BM25 Retriever 생성 실패 ({doc_type}): {e}")
                return None
        
        shelter_bm25 = create_bm25_retriever("shelter")
        guideline_bm25 = create_bm25_retriever("disaster_guideline")
        
        # 3. Ensemble (Hybrid) Retriever
        shelter_hybrid = EnsembleRetriever(
            retrievers=[shelter_vector_retriever, shelter_bm25] if shelter_bm25 else [shelter_vector_retriever],
            weights=[0.6, 0.4] if shelter_bm25 else [1.0]
        )
        
        guideline_hybrid = EnsembleRetriever(
            retrievers=[guideline_vector_retriever, guideline_bm25] if guideline_bm25 else [guideline_vector_retriever],
            weights=[0.7, 0.3] if guideline_bm25 else [1.0]
        )
        
        return shelter_hybrid, guideline_hybrid
        
    except Exception as e:
        print(f"⚠️ 하이브리드 리트리버 생성 실패: {e}")
        return None, None


def create_langgraph_app():
    """LangGraph Agent 생성"""
    
    # 1. LLM 초기화
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    llm_creative = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)  # 일반 지식용
    
    # 2. 의도 분류 체인
    intent_classification_prompt = ChatPromptTemplate.from_messages([
        ("system", """당신은 사용자 질문의 의도를 정확하게 분류하는 AI입니다.

질문을 다음 카테고리 중 하나로 분류하세요:

1. **shelter_search**: 특정 위치의 대피소 찾기
   - 예: "한라산 근처 대피소", "강남역 대피소", "동대문맨션 대피소 정보"
   
2. **shelter_count**: 특정 조건의 대피소 개수 세기
   - 예: "서울 대피소 개수", "지하 대피소 몇 개", "부산 민방위 대피소는 총 몇 개"
   
3. **shelter_capacity**: 수용인원 기준 대피소 찾기
   - 예: "천 명 이상 수용 가능한 대피소", "100명 수용 가능한 근처 대피소"
   
4. **disaster_guideline**: 재난 행동요령 질문
   - 예: "지진 발생 시 행동요령", "화재 대처법", "산사태 났을 때"
   
5. **hybrid_location_disaster**: 위치 + 재난 상황 복합 질문
   - 예: "설악산 근처인데 산사태 발생 시", "강남역에서 지진 나면"
   
6. **general_knowledge**: 재난 관련 일반 지식 (정의, 원인 등)
   - 예: "지진이 뭐야", "쓰나미란", "태풍의 원인"
   
7. **general_chat**: 일반 대화
   - 예: "안녕", "고마워", "날씨 어때"

**응답 형식**: JSON
{{
    "intent": "카테고리명",
    "confidence": 0.95,
    "reason": "분류 근거"
}}"""),
        ("user", "{query}")
    ])
    
    intent_chain = intent_classification_prompt | llm | StrOutputParser()
    
    # 3. 질문 재정의 체인 (검색 정확도 향상)
    query_rewrite_prompt = ChatPromptTemplate.from_messages([
        ("system", """당신은 검색 쿼리를 최적화하는 전문가입니다.

사용자의 질문을 BM25 키워드 검색에 최적화된 형태로 재작성하세요.

**재작성 규칙**:
1. 핵심 키워드만 추출 (불필요한 조사, 어미 제거)
2. 동의어 추가 (예: "대피소" → "대피소 피난처")
3. 지역명은 다양한 형태로 표현 (예: "서울" → "서울 서울시 서울특별시")
4. 위치 유형 명확화 (예: "지하" → "지하 지하층")
5. 최대 10단어 이내로 간결하게

**예시**:
- "한라산 근처 대피소는?" → "한라산 제주 대피소 피난처"
- "서울에 있는 지하 대피소" → "서울 서울시 지하 지하층 대피소"
- "동대문맨션 대피소" → "동대문맨션 동대문 대피소"

**응답**: 재작성된 쿼리만 출력 (설명 없이)"""),
        ("user", "{original_query}")
    ])
    
    query_rewrite_chain = query_rewrite_prompt | llm | StrOutputParser()
    
    # 4. 하이브리드 리트리버 생성
    shelter_hybrid, guideline_hybrid = create_hybrid_retrievers()
    
    # 5. Tools 정의
    @tool
    def search_shelter_by_location(query: str) -> str:
        """
        특정 위치의 대피소를 검색합니다.
        카카오 API로 좌표를 찾고, 가장 가까운 대피소 5곳을 반환합니다.
        
        Args:
            query: 위치 정보 (지명, 건물명, 주소 등)
        """
        try:
            # 쿼리 재정의
            rewritten = query_rewrite_chain.invoke({"original_query": query})
            print(f"[search_shelter_by_location] 재정의: {query} → {rewritten}")
            
            # 카카오 API로 좌표 검색
            kakao_api_key = os.getenv("KAKAO_REST_API_KEY")
            if not kakao_api_key:
                return "카카오 API 키가 설정되지 않았습니다."
            
            headers = {"Authorization": f"KakaoAK {kakao_api_key}"}
            url = "https://dapi.kakao.com/v2/local/search/keyword.json"
            params = {"query": rewritten}
            
            response = requests.get(url, headers=headers, params=params)
            data = response.json()
            
            if not data.get("documents"):
                return f"'{query}' 위치를 찾을 수 없습니다."
            
            # 첫 번째 결과의 좌표
            place = data["documents"][0]
            user_lat = float(place["y"])
            user_lon = float(place["x"])
            place_name = place["place_name"]
            
            print(f"[search_shelter_by_location] 좌표: {place_name} ({user_lat}, {user_lon})")
            
            # Haversine 거리 계산
            def haversine(lat1, lon1, lat2, lon2):
                from math import radians, sin, cos, sqrt, atan2
                R = 6371
                dlat = radians(lat2 - lat1)
                dlon = radians(lon2 - lon1)
                a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
                c = 2 * atan2(sqrt(a), sqrt(1-a))
                return R * c
            
            # 모든 대피소 가져오기
            all_data = vectorstore.get(where={"type": "shelter"})
            shelters = []
            
            for i, metadata in enumerate(all_data['metadatas']):
                try:
                    lat = float(metadata.get('latitude', 0))
                    lon = float(metadata.get('longitude', 0))
                    if lat == 0 or lon == 0:
                        continue
                    
                    distance = haversine(user_lat, user_lon, lat, lon)
                    shelters.append({
                        'name': metadata.get('facility_name', 'N/A'),
                        'address': metadata.get('address', 'N/A'),
                        'distance': distance,
                        'capacity': int(metadata.get('capacity', 0)),
                        'shelter_type': metadata.get('shelter_type', 'N/A'),
                        'facility_type': metadata.get('facility_type', 'N/A')
                    })
                except Exception as e:
                    continue
            
            # 거리순 정렬
            shelters.sort(key=lambda x: x['distance'])
            top_5 = shelters[:5]
            
            if not top_5:
                return f"'{place_name}' 근처에 대피소를 찾을 수 없습니다."
            
            # 결과 포맷팅
            result = f"📍 **{place_name}** 근처 대피소 {len(top_5)}곳\n\n"
            for i, s in enumerate(top_5, 1):
                result += f"{i}. **{s['name']}**\n"
                result += f"   📍 거리: {s['distance']:.2f}km\n"
                result += f"   📍 주소: {s['address']}\n"
                result += f"   📍 위치: {s['shelter_type']}\n"
                result += f"   📍 수용인원: {s['capacity']:,}명\n\n"
            
            return result.strip()
            
        except Exception as e:
            print(f"[ERROR] search_shelter_by_location: {e}")
            return f"검색 중 오류 발생: {str(e)}"
    
    @tool
    def count_shelters(query: str) -> str:
        """
        특정 조건(지역, 위치유형 등)에 맞는 대피소 개수를 셉니다.
        
        Args:
            query: 검색 조건 (예: "서울 지하", "부산 민방위")
        """
        try:
            # 쿼리 재정의
            rewritten = query_rewrite_chain.invoke({"original_query": query})
            print(f"[count_shelters] 재정의: {query} → {rewritten}")
            
            if shelter_hybrid is None:
                return "검색 시스템이 초기화되지 않았습니다."
            
            # 하이브리드 검색
            results = shelter_hybrid.invoke(rewritten)
            
            # 중복 제거
            seen = set()
            count = 0
            for doc in results:
                name = doc.metadata.get('facility_name', '')
                if name and name not in seen:
                    seen.add(name)
                    count += 1
            
            if count == 0:
                return f"'{query}' 조건에 맞는 대피소를 찾을 수 없습니다."
            
            return f"**'{query}'** 조건에 맞는 대피소는 총 **{count}개**입니다."
            
        except Exception as e:
            print(f"[ERROR] count_shelters: {e}")
            return f"검색 중 오류 발생: {str(e)}"
    
    @tool
    def search_shelter_by_capacity(query: str) -> str:
        """
        수용인원 기준으로 대피소를 검색합니다.
        
        Args:
            query: 수용인원 조건 (예: "천 명 이상", "100명 수용 가능")
        """
        try:
            # 숫자 추출
            import re
            numbers = re.findall(r'\d+', query)
            if not numbers:
                return "수용인원을 명확히 입력해주세요. (예: 1000명 이상)"
            
            min_capacity = int(numbers[0])
            
            # 숫자 단위 처리 (천, 만)
            if '천' in query:
                min_capacity *= 1000
            elif '만' in query:
                min_capacity *= 10000
            
            print(f"[search_shelter_by_capacity] 최소 수용인원: {min_capacity}명")
            
            # 모든 대피소 가져오기
            all_data = vectorstore.get(where={"type": "shelter"})
            shelters = []
            
            for metadata in all_data['metadatas']:
                capacity = int(metadata.get('capacity', 0))
                if capacity >= min_capacity:
                    shelters.append({
                        'name': metadata.get('facility_name', 'N/A'),
                        'address': metadata.get('address', 'N/A'),
                        'capacity': capacity,
                        'shelter_type': metadata.get('shelter_type', 'N/A')
                    })
            
            # 수용인원 내림차순 정렬
            shelters.sort(key=lambda x: x['capacity'], reverse=True)
            top_10 = shelters[:10]
            
            if not top_10:
                return f"{min_capacity:,}명 이상 수용 가능한 대피소를 찾을 수 없습니다."
            
            result = f"📊 **{min_capacity:,}명 이상** 수용 가능한 대피소 **{len(shelters)}곳** 중 상위 10곳\n\n"
            for i, s in enumerate(top_10, 1):
                result += f"{i}. **{s['name']}** ({s['capacity']:,}명)\n"
                result += f"   📍 {s['address']}\n"
                result += f"   📍 위치: {s['shelter_type']}\n\n"
            
            return result.strip()
            
        except Exception as e:
            print(f"[ERROR] search_shelter_by_capacity: {e}")
            return f"검색 중 오류 발생: {str(e)}"
    
    @tool
    def search_disaster_guideline(query: str) -> str:
        """
        재난 행동요령을 검색합니다.
        
        Args:
            query: 재난 유형 (예: "지진", "화재", "산사태")
        """
        try:
            # 쿼리 재정의
            rewritten = query_rewrite_chain.invoke({"original_query": query})
            print(f"[search_disaster_guideline] 재정의: {query} → {rewritten}")
            
            if guideline_hybrid is None:
                return "가이드라인 검색 시스템이 초기화되지 않았습니다."
            
            # 하이브리드 검색
            results = guideline_hybrid.invoke(rewritten)
            
            if not results:
                return f"'{query}' 관련 행동요령을 찾을 수 없습니다."
            
            # 상위 3개 결과 통합
            combined = "\n\n".join([doc.page_content for doc in results[:3]])
            
            return f"🚨 **{query} 행동요령**\n\n{combined}"
            
        except Exception as e:
            print(f"[ERROR] search_disaster_guideline: {e}")
            return f"검색 중 오류 발생: {str(e)}"
    
    @tool
    def answer_general_knowledge(query: str) -> str:
        """
        재난 관련 일반 지식 질문에 답변합니다. (정의, 원인, 특징 등)
        VectorDB에 없는 정보는 LLM의 사전 학습 지식을 활용합니다.
        
        Args:
            query: 일반 지식 질문 (예: "지진이 뭐야", "쓰나미란")
        """
        try:
            print(f"[answer_general_knowledge] 질문: {query}")
            
            # LLM에게 직접 질문 (사전 학습 지식 활용)
            prompt = f"""당신은 재난 안전 전문가입니다.
다음 질문에 정확하고 간결하게 답변하세요.

질문: {query}

답변 형식:
- 핵심 정의를 2-3문장으로 설명
- 주요 특징이나 원인을 불릿 포인트로 정리
- 전문 용어는 쉽게 풀어서 설명
- 최대 200자 이내로 간결하게"""
            
            response = llm_creative.invoke([HumanMessage(content=prompt)])
            
            return f"💡 **{query}**\n\n{response.content}"
            
        except Exception as e:
            print(f"[ERROR] answer_general_knowledge: {e}")
            return "죄송합니다. 답변 생성 중 오류가 발생했습니다."
    
    # 6. Tools 리스트
    tools = [
        search_shelter_by_location,
        count_shelters,
        search_shelter_by_capacity,
        search_disaster_guideline,
        answer_general_knowledge
    ]
    
    # 7. LLM에 Tools 바인딩
    llm_with_tools = llm.bind_tools(tools)
    
    # 8. State 정의
    class AgentState(TypedDict):
        messages: Annotated[list[BaseMessage], add_messages]
        intent: str
        rewritten_query: str
    
    # 9. 시스템 프롬프트
    SYSTEM_PROMPT = """당신은 대한민국의 재난 안전 전문 AI 도우미입니다.

**핵심 원칙**:
1. **정확성 우선**: 제공된 도구 결과만 사용하고, 없는 정보는 지어내지 마세요
2. **의도 파악**: 사용자 질문의 의도를 정확히 분류하고 적절한 도구를 선택하세요
3. **복합 질문 처리**: 여러 의도가 섞인 질문은 순차적으로 처리하세요

**도구 선택 가이드**:
- 특정 위치 대피소 찾기 → **search_shelter_by_location**
- 대피소 개수 세기 → **count_shelters**
- 수용인원 기준 검색 → **search_shelter_by_capacity**
- 재난 행동요령 → **search_disaster_guideline**
- 재난 관련 일반 지식 (정의, 원인) → **answer_general_knowledge**

**복합 질문 예시**:
"설악산 근처인데 산사태 발생 시 어떻게 해야 해?"
→ 1단계: search_shelter_by_location("설악산")
→ 2단계: search_disaster_guideline("산사태")
→ 3단계: 두 결과를 자연스럽게 통합

**응답 형식**:
- 구체적이고 실용적인 정보 제공
- 중요 정보는 **볼드체** 강조
- 숫자는 쉼표 구분 (1,000명)
- 이모지 적절히 활용 (📍🚨💡📊)
"""
    
    # 10. 노드 함수들
    def intent_classifier_node(state: AgentState):
        """의도 분류 노드"""
        messages = state["messages"]
        last_message = messages[-1].content
        
        print(f"\n[의도분류 노드] 입력: {last_message}")
        
        try:
            # 의도 분류
            intent_result = intent_chain.invoke({"query": last_message})
            import json
            intent_data = json.loads(intent_result)
            intent = intent_data["intent"]
            
            print(f"[의도분류 노드] 결과: {intent} (신뢰도: {intent_data.get('confidence', 0)})")
            
            return {"intent": intent}
            
        except Exception as e:
            print(f"[의도분류 노드] 오류: {e}, 기본값 사용")
            return {"intent": "general_chat"}
    
    def query_rewrite_node(state: AgentState):
        """질문 재정의 노드"""
        messages = state["messages"]
        last_message = messages[-1].content
        intent = state.get("intent", "")
        
        # 일반 대화나 일반 지식은 재정의 불필요
        if intent in ["general_chat", "general_knowledge"]:
            return {"rewritten_query": last_message}
        
        print(f"\n[질문재정의 노드] 입력: {last_message}")
        
        try:
            rewritten = query_rewrite_chain.invoke({"original_query": last_message})
            print(f"[질문재정의 노드] 결과: {rewritten}")
            return {"rewritten_query": rewritten}
        except Exception as e:
            print(f"[질문재정의 노드] 오류: {e}")
            return {"rewritten_query": last_message}
    
    def agent_node(state: AgentState):
        """에이전트 추론 노드 (도구 선택 및 실행)"""
        messages = state["messages"]
        intent = state.get("intent", "")
        
        print(f"\n[에이전트 노드] 의도: {intent}")
        
        # 시스템 프롬프트 추가
        if not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
        
        # LLM 호출 (도구 선택)
        response = llm_with_tools.invoke(messages)
        
        return {"messages": [response]}
    
    def should_continue(state: AgentState):
        """도구 실행 필요 여부 판단"""
        messages = state["messages"]
        last_message = messages[-1]
        
        # 도구 호출이 있으면 도구 실행
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        
        # 없으면 종료
        return END
    
    # 11. 그래프 구성
    workflow = StateGraph(AgentState)
    
    # 노드 추가
    workflow.add_node("intent_classifier", intent_classifier_node)
    workflow.add_node("query_rewrite", query_rewrite_node)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", ToolNode(tools))
    
    # 엣지 연결
    workflow.add_edge(START, "intent_classifier")
    workflow.add_edge("intent_classifier", "query_rewrite")
    workflow.add_edge("query_rewrite", "agent")
    workflow.add_conditional_edges("agent", should_continue, ["tools", END])
    workflow.add_edge("tools", "agent")  # 도구 실행 후 다시 에이전트로
    
    # 12. 메모리 체크포인트
    memory = MemorySaver()
    
    # 13. 컴파일
    app = workflow.compile(checkpointer=memory)
    
    print("[LangGraph] 앱 생성 완료")
    print(f"  - 노드: intent_classifier → query_rewrite → agent ⇄ tools")
    print(f"  - 도구: {len(tools)}개")
    
    return app


# -----------------------------------------------------------------------------
# 3. FastAPI Lifespan (수명 주기) 핸들러
# -----------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    서버 시작/종료 시 실행되는 초기화 및 정리 작업
    앱 실행 시:
    - Vector DB 로드 및 초기화
    - 대피소 데이터 로드
    - LangGraph Agent 초기화
    앱 종료 시:
    - 리소스 정리 (현재는 별도 정리 작업 없음)
    """
    global vectorstore, shelter_df, embeddings
    global shelter_hybrid_retriever, guideline_hybrid_retriever, langgraph_app
    
    # OpenAI 임베딩 초기화
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=os.getenv("OPENAI_API_KEY"))
        print("[lifespan] 임베딩 모델 초기화 성공")
    except Exception as e:
        embeddings = None
        print(f"[lifespan] 임베딩 모델 초기화 실패: {e}")
    
    # 벡터 DB 로드 시도
    try:
        vectorstore = Chroma(
            collection_name="shelter_and_disaster_guidelines",
            embedding_function=embeddings,
            persist_directory="chroma_db"
        )
        print("[lifespan] 벡터DB 로드 성공")
    except Exception as e:
        vectorstore = None
        print(f"[lifespan] 벡터DB 로드 실패: {e}")
    
    # 대피소 데이터 로드
    try:
        shelter_data = load_shelter_csv("shelter.csv", data_dir="./data")
        shelter_df = pd.DataFrame(shelter_data)
        print(f"[lifespan] 대피소 데이터 로드 성공: {len(shelter_df)}개")
    except Exception as e:
        shelter_df = None
        print(f"[lifespan] 대피소 데이터 로드 실패: {e}")
    
    # 하이브리드 리트리버 및 LangGraph 초기화
    try:
        shelter_hybrid_retriever, guideline_hybrid_retriever = create_hybrid_retrievers()
        langgraph_app = create_langgraph_app()
        print("[lifespan] LangGraph Agent 초기화 완료")
    except Exception as e:
        shelter_hybrid_retriever = None
        guideline_hybrid_retriever = None
        langgraph_app = None
        print(f"[lifespan] LangGraph 초기화 실패: {e}")
    
    yield # 애플리케이션 실행 중
    
    # 여기에 종료 시 필요한 정리 작업 코드를 추가할 수 있음

# FastAPI 앱 인스턴스 생성 및 lifespan 핸들러 연결
app = FastAPI(title="대피소 안내 챗봇 API", lifespan=lifespan)

# 전역 변수 초기화
vectorstore = None
shelter_df = None
embeddings = None
shelter_hybrid_retriever = None
guideline_hybrid_retriever = None
langgraph_app = None




# -----------------------------------------------------------------------------
# 4. API 엔드포인트: 지명 추출 및 통합 검색
# -----------------------------------------------------------------------------

@app.post("/api/location/extract")
async def extract_location(request: LocationExtractRequest = Body(...)):
    """
    사용자 질의(Query)를 분석하여 적절한 응답을 제공합니다.
    
    =========================================================================
    [NEW] LangGraph Agent 기반 통합 처리
    =========================================================================
    
    기존 방식 (의도 분류 → 분기 처리)에서 Agent 자동 처리로 변경:
    
    1. **Agent가 질문 분석**
       - 사용자 질문의 의도를 자동으로 파악
       - 필요한 도구를 스스로 선택하여 실행
    
    2. **사용 가능한 도구**
       - search_shelter: 지역명/건물명으로 대피소 검색 (하이브리드)
       - search_shelter_by_kakao: 카카오 API + 좌표 기반 대피소 검색
       - search_guideline: 재난 행동요령 검색
       - get_shelter_statistics: 대피소 통계
    
    3. **Agent의 장점**
       - 자동 의도 분류 (별도 classify_user_intent 불필요)
       - 복잡한 질문 처리 (여러 도구 조합 가능)
       - 대화 맥락 유지 (세션 기반 메모리)
    
    4. **폴백 처리**
       - LangGraph 초기화 실패 시 기존 로직 사용
    
    =========================================================================
    """
    
    # 리소스 확인
    if vectorstore is None or shelter_df is None:
        return LocationExtractResponse(success=False, message="서버 초기화가 완료되지 않았습니다.")
    
    # 쿼리 유효성 검사
    query = request.query.strip()
    if not query:
        return LocationExtractResponse(success=False, message="입력 문장이 비어 있습니다.")

    print(f"[API] 사용자 쿼리: '{query}'")
    
    # =========================================================================
    # 쿼리 유형 분류: 단순 대피소 위치 질문 vs 복잡한 질문
    # =========================================================================
    # 단순 질문: "강남역 근처 대피소", "명동 대피소 어디야" → 지도 표시용 좌표/대피소 배열 반환
    # 복잡한 질문: "강남역인데 지진 나면 어디로", "명동에서 화재 발생 시 대처법" → Agent 텍스트 응답
    
    # 재난 관련 키워드 목록
    disaster_keywords = [
        "지진", "홍수", "태풍", "화재", "폭발", "산사태", "쓰나미", 
        "화산", "방사능", "가스", "붕괴", "테러", "전쟁",
        "행동요령", "대처법", "대응", "조치", "주의사항", "발생하면", "발생 시"
    ]
    
    # 질문에 재난 키워드가 포함되어 있는지 확인
    has_disaster_context = any(keyword in query for keyword in disaster_keywords)
    
    # 단순 대피소 위치 질문인지 확인
    shelter_keywords = ["대피소", "피난소", "피난처", "근처", "주변", "어디"]
    has_shelter_request = any(keyword in query for keyword in shelter_keywords)
    
    # 라우팅 결정
    use_agent = False
    
    if has_disaster_context:
        # 재난 키워드가 있으면 무조건 Agent 사용 (복잡한 질문)
        use_agent = True
        print(f"[INFO] 재난 맥락 감지 → LangGraph Agent 사용")
    elif not has_shelter_request:
        # 대피소 관련 키워드도 없고 재난 키워드도 없으면 Agent 사용 (일반 대화 또는 통계)
        use_agent = True
        print(f"[INFO] 일반 질문 감지 → LangGraph Agent 사용")
    else:
        # 대피소 키워드만 있으면 기존 로직 사용 (단순 위치 질문)
        use_agent = False
        print(f"[INFO] 단순 대피소 위치 질문 감지 → 기존 로직 사용 (지도 표시)")
    
    # =========================================================================
    # LangGraph Agent 사용 (복잡한 질문 처리)
    # =========================================================================
    if use_agent and langgraph_app is not None:
        try:
            print(f"[INFO] LangGraph Agent로 처리 시작")
            
            # 세션 ID 생성 (요청별 고유 ID)
            session_id = f"session_{hash(query) % 100000}"
            config = {"configurable": {"thread_id": session_id}}
            
            # Agent 실행
            # - Agent가 질문을 분석하여 자동으로 도구 선택
            # - search_shelter, search_guideline 등 적절한 도구 실행
            # - 여러 도구를 조합하여 사용 가능
            result = langgraph_app.invoke(
                {"messages": [HumanMessage(content=query)]},
                config=config
            )
            
            # Agent의 최종 응답 추출
            bot_response = result["messages"][-1].content
            print(f"[INFO] LangGraph Agent 응답 완료 (길이: {len(bot_response)})")
            
            # Agent 응답을 LocationExtractResponse 형식으로 반환
            # - 텍스트 응답 형식 (message에 포함)
            # - 좌표/대피소 배열은 비어있음 (지도 표시 불가)
            return LocationExtractResponse(
                success=True,
                location=None,
                coordinates=None,
                shelters=[],
                total_count=0,
                message=bot_response
            )
            
        except Exception as e:
            print(f"[ERROR] LangGraph Agent 실행 실패: {e}")
            print(f"[INFO] 기존 로직으로 폴백")
            # 에러 발생 시 아래 기존 로직으로 폴백
    
    elif use_agent and langgraph_app is None:
        print(f"[WARNING] LangGraph Agent가 초기화되지 않음, 기존 로직 사용")
    
    # =========================================================================
    # 기존 로직 (LangGraph 실패 시 폴백)
    # =========================================================================
    
    # -----------------------
    # 1차: LLM 기반 의도 분류
    # -----------------------
    intent = classify_user_intent(query)
    print(f"[API] 분류된 의도: '{intent}'")
    
    # -----------------------
    # 2차: 의도별 처리 로직
    # -----------------------
    
    # CASE 1: 일반 대화
    if intent == "general_chat":
        print(f"[API] general_chat 처리")
        return LocationExtractResponse(
            success=True,
            message="안녕하세요! 저는 대피소 안내 챗봇입니다. 지역명을 입력하시면 주변 대피소를 찾아드리고, 재난 상황에 대한 행동요령도 안내해 드립니다."
        )
    
    # CASE 2: 재난행동요령 관련 질문
    elif intent == "disaster_guide":
        # Vector DB에서 재난행동요령 문서 검색
        print(f"[DEBUG] 재난행동요령 검색 쿼리: {query}")
        
        # filter를 사용하여 disaster_guideline 타입만 검색
        try:
            results = vectorstore.similarity_search(
                query, 
                k=5,
                filter={"type": "disaster_guideline"}
            )
            print(f"[DEBUG] disaster_guideline 필터링 검색 결과: {len(results)}개")
        except:
            # filter 지원 안 되면 전체 검색 후 필터링
            all_results = vectorstore.similarity_search(query, k=20)
            results = [doc for doc in all_results if doc.metadata.get("type") == "disaster_guideline"]
            print(f"[DEBUG] 전체 검색 후 필터링 결과: {len(results)}개")
        
        # 검색 결과 디버깅
        for i, doc in enumerate(results[:3]):
            doc_type = doc.metadata.get("type", "NONE")
            category = doc.metadata.get("category", "N/A")
            keyword = doc.metadata.get("keyword", "N/A")
            print(f"[DEBUG] 문서 {i+1}: type={doc_type}, category={category}, keyword={keyword}")
            print(f"[DEBUG]   내용: {doc.page_content[:150]}...")
        
        # 재난행동요령 문서가 없으면 에러
        if not results or len(results) == 0:
            print("[ERROR] VectorStore에 disaster_guideline 문서가 없습니다!")
            return LocationExtractResponse(
                success=False, 
                message="재난행동요령 데이터베이스에 문제가 있습니다. 시스템 관리자에게 문의하세요."
            )
        
        # 가장 관련성 높은 문서 선택
        disaster_doc = results[0]
        
        print(f"[DEBUG] 선택된 재난문서 - category: {disaster_doc.metadata.get('category')}, keyword: {disaster_doc.metadata.get('keyword')}")
        print(f"[DEBUG] 문서 길이: {len(disaster_doc.page_content)}")
        
        # 응답 메시지 구성 (카테고리와 키워드 정보 포함)
        category = disaster_doc.metadata.get('category', '')
        keyword = disaster_doc.metadata.get('keyword', '')
        header = f"📋 {category} - {keyword}\n\n" if category and keyword else ""
        
        return LocationExtractResponse(
            success=True,
            location=None,
            coordinates=None,
            shelters=[],
            total_count=0,
            message=header + disaster_doc.page_content[:1500]  # 답변 길이 증가
        )
        
    # CASE 3: 대피소 관련 질문
    elif intent == "find_shelter":
        print(f"[API] find_shelter 처리 시작 - query: '{query}'")
        
        # =====================================================================
        # STEP 1: 사용자 쿼리에서 순수 지명만 추출 (불필요한 단어 제거)
        # =====================================================================
        # 예: "강남역 대피소 알려줘" -> "강남역"
        # 예: "명동 근처 피난소" -> "명동"
        location_query = query
        
        # 대피소 관련 키워드 제거 리스트
        remove_keywords = [
            "대피소", "피난소", "피난처", "근처", "주변", "가까운", "어디", "위치",
            "찾아줘", "알려줘", "검색", "보여줘", "있어", "는?", "은?", "?", "!",
            "좀", "요", "주세요", "해줘", "해주세요", "있나요", "있어요"
        ]
        
        for keyword in remove_keywords:
            location_query = location_query.replace(keyword, "")
        
        # 공백 정리 (여러 공백을 하나로 통합)
        location_query = " ".join(location_query.split()).strip()
        
        print(f"[DEBUG] 정제된 위치 쿼리: '{location_query}'")
        
        # 정제 후 비어있으면 원본 쿼리 사용
        if not location_query:
            location_query = query
            print(f"[DEBUG] 정제 결과가 비어있어 원본 쿼리 사용")
        
        # =====================================================================
        # STEP 2: 카카오 로컬 API 키 확인
        # =====================================================================
        kakao_key = os.getenv("KAKAO_REST_API_KEY")
        if not kakao_key:
            print(f"[ERROR] KAKAO_REST_API_KEY 없음")
            return LocationExtractResponse(success=False, message="KAKAO_REST_API_KEY 환경변수가 없습니다.")
        
        # =====================================================================
        # STEP 3: 여러 지명이 포함된 경우 우선순위 판단
        # =====================================================================
        # 예: "잠실 롯데월드" -> "롯데월드" 우선 선택 (관광명소)
        # 우선순위: 1=관광명소/문화시설, 2=교통시설(역), 3=행정구역, 4=기타
        location_parts = location_query.split()
        selected_location = location_query
        
        if len(location_parts) > 1:
            print(f"[DEBUG] 여러 지명 감지: {location_parts}, 카카오 API로 우선순위 판단")
            
            url = "https://dapi.kakao.com/v2/local/search/keyword.json"
            headers = {"Authorization": f"KakaoAK {kakao_key}"}
            
            best_candidate = None
            best_priority = 999
            
            # 카테고리별 우선순위 정의
            priority_categories = {
                1: ["관광명소", "문화시설", "여가시설", "공공기관", "테마파크"],
                2: ["교통,수송", "지하철역"],
                3: ["행정구역"],
            }
            
            # 각 지명을 카카오 API로 검색하여 카테고리 확인
            for part in location_parts:
                resp = requests.get(url, headers=headers, params={"query": part, "size": 5})
                if resp.status_code == 200:
                    docs = resp.json().get("documents", [])
                    if docs:
                        doc = docs[0]
                        category_name = doc.get("category_name", "")
                        print(f"[DEBUG] '{part}' 검색 결과 - category: {category_name}")
                        
                        # 카테고리 우선순위 판단
                        priority = 4  # 기본값 (기타)
                        for pri, keywords in priority_categories.items():
                            if any(keyword in category_name for keyword in keywords):
                                priority = pri
                                break
                        
                        # 더 높은 우선순위(낮은 숫자)면 선택
                        if priority < best_priority:
                            best_priority = priority
                            best_candidate = part
                            print(f"[DEBUG] 우선순위 {priority}: '{part}' 선택 (category: {category_name})")
            
            # 우선순위가 가장 높은 지명 선택
            if best_candidate:
                selected_location = best_candidate
                print(f"[DEBUG] 최종 선택된 위치: '{selected_location}' (우선순위: {best_priority})")
            else:
                # API 검색 실패시 첫 번째 지명 사용
                selected_location = location_parts[0]
                print(f"[DEBUG] API 검색 실패, 첫 번째 지명 사용: '{selected_location}'")
        
        location_query = selected_location
        
        # =====================================================================
        # STEP 4: 카카오 API를 사용하여 최종 위치 검색 (위/경도 좌표 획득)
        # =====================================================================
        url = "https://dapi.kakao.com/v2/local/search/keyword.json"
        headers = {"Authorization": f"KakaoAK {kakao_key}"}
        params = {"query": location_query, "size": 1}
        
        print(f"[DEBUG] 카카오 API 최종 검색 - query: '{location_query}'")
        resp = requests.get(url, headers=headers, params=params)
        print(f"[DEBUG] 카카오 API 응답 - status: {resp.status_code}")
        
        if resp.status_code != 200:
            return LocationExtractResponse(success=False, message=f"카카오 API 오류: {resp.status_code}")
            
        data = resp.json()
        print(f"[DEBUG] 카카오 API 결과 개수: {len(data.get('documents', []))}")
        
        # =====================================================================
        # 카카오 API 검색 실패 시 LangGraph Agent를 사용한 대피소 검색
        # =====================================================================
        if not data.get("documents"):
            print(f"[WARNING] 카카오 API에서 '{location_query}' 위치를 찾지 못함")
            print(f"[INFO] LangGraph Agent를 사용하여 대피소 검색 시도")
            
            # LangGraph Agent가 초기화되어 있는지 확인
            if langgraph_app is None:
                print(f"[ERROR] LangGraph Agent가 초기화되지 않음")
                return LocationExtractResponse(
                    success=False, 
                    message=f"'{location_query}' 위치를 찾을 수 없습니다. 다른 지역명을 입력해 주세요."
                )
            
            try:
                # LangGraph Agent에게 대피소 검색 요청
                # - Agent가 search_shelter 도구를 사용하여 하이브리드 검색 수행
                # - 질문 재정의(Query Rewriting)를 통해 검색 정확도 향상
                # - Vector DB + BM25 앙상블 검색으로 키워드 매칭 강화
                print(f"[DEBUG] LangGraph Agent 호출 - 쿼리: '{query}'")
                
                # 세션 ID 생성 (임시)
                session_id = f"temp_{hash(query) % 10000}"
                config = {"configurable": {"thread_id": session_id}}
                
                # Agent 실행
                result = langgraph_app.invoke(
                    {"messages": [HumanMessage(content=query)]},
                    config=config
                )
                
                # Agent의 응답 추출
                bot_response = result["messages"][-1].content
                print(f"[DEBUG] LangGraph Agent 응답 (길이: {len(bot_response)})")
                
                # Agent 응답을 message로 반환
                # - 좌표 기반 검색이 아닌 하이브리드 검색 결과
                # - shelters 배열은 비어있지만 message에 대피소 정보 포함
                return LocationExtractResponse(
                    success=True,
                    location=location_query,
                    coordinates=None,  # 좌표 정보 없음 (카카오 API 실패)
                    shelters=[],  # Agent 응답은 텍스트 형태로 message에 포함
                    total_count=0,
                    message=bot_response  # Agent의 검색 결과 텍스트
                )
                
            except Exception as e:
                print(f"[ERROR] LangGraph Agent 실행 실패: {e}")
                return LocationExtractResponse(
                    success=False, 
                    message=f"'{location_query}' 위치를 찾을 수 없습니다. 다른 지역명을 입력해 주세요."
                )
        
        # =====================================================================
        # 카카오 API 검색 성공 시 기존 로직 사용 (좌표 기반 대피소 검색)
        # =====================================================================
        # - 카카오 API로 획득한 위/경도 좌표 사용
        # - Haversine 공식으로 사용자 위치 ↔ 대피소 간 직선 거리 계산
        # - 거리순 정렬 후 가장 가까운 5개 대피소 반환
        
        # 좌표 추출
        place = data["documents"][0]
        lat = float(place["y"])  # 위도
        lon = float(place["x"])  # 경도
        place_name = place.get("place_name", location_query)
        
        print(f"[DEBUG] 좌표 추출 성공 - place_name: {place_name}, lat: {lat}, lon: {lon}")
        
        # =====================================================================
        # STEP 5: VectorStore에서 모든 대피소 데이터 가져와서 거리 계산
        # =====================================================================
        import math
        
        def haversine(lat1, lon1, lat2, lon2):
            """
            Haversine 공식: 구면상의 두 점 사이의 최단 거리 계산
            
            Args:
                lat1, lon1: 첫 번째 점의 위도/경도 (사용자 위치)
                lat2, lon2: 두 번째 점의 위도/경도 (대피소 위치)
            
            Returns:
                float: 두 점 사이의 거리 (단위: km)
            """
            R = 6371  # 지구 반지름 (km)
            phi1, phi2 = math.radians(lat1), math.radians(lat2)
            d_phi = math.radians(lat2 - lat1)
            d_lambda = math.radians(lon2 - lon1)
            a = math.sin(d_phi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(d_lambda/2)**2
            return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        # VectorStore에서 shelter 타입 문서만 필터링하여 가져오기
        # - filter: {"type": "shelter"} 조건으로 대피소 데이터만 추출
        all_data = vectorstore.get(where={"type": "shelter"})
        all_metadatas = all_data.get('metadatas', [])
        
        print(f"[DEBUG] VectorStore에서 {len(all_metadatas)}개 대피소 메타데이터 가져옴")
        
        shelters = []
        
        # 각 대피소의 좌표를 사용하여 거리 계산
        for metadata in all_metadatas:
            # shelter 타입 문서만 처리 (이중 확인)
            if metadata.get('type') != 'shelter':
                continue
                
            # 좌표 정보 추출 (documents.py에서 영문 키로 저장됨)
            s_lat = metadata.get('lat')  # 대피소 위도
            s_lon = metadata.get('lon')  # 대피소 경도
            
            if s_lat is not None and s_lon is not None:
                try:
                    s_lat = float(s_lat)
                    s_lon = float(s_lon)
                    
                    # Haversine 공식으로 사용자 위치 ↔ 대피소 간 거리 계산
                    distance = haversine(lat, lon, s_lat, s_lon)
                    
                    # 대피소 정보 객체 생성
                    shelter_info = {
                        'name': metadata.get('facility_name', 'N/A'),  # 시설명
                        'address': metadata.get('address', 'N/A'),     # 주소
                        'lat': s_lat,                                   # 위도
                        'lon': s_lon,                                   # 경도
                        'capacity': int(metadata.get('capacity', 0)),  # 수용인원
                        'distance': distance                            # 거리 (km)
                    }
                    shelters.append(shelter_info)
                    
                except (ValueError, TypeError):
                    # 좌표 변환 실패 시 해당 대피소는 건너뜀
                    continue
        
        print(f"[DEBUG] 총 {len(shelters)}개 대피소의 거리 계산 완료")
        
        # =====================================================================
        # STEP 6: 거리순 정렬 후 상위 5개 반환
        # =====================================================================
        shelters.sort(key=lambda x: x['distance'])  # 거리 오름차순 정렬
        top_shelters = shelters[:5]  # 가장 가까운 5개 선택
        
        print(f"[DEBUG] 상위 5개 대피소 선택 완료")
        for i, s in enumerate(top_shelters):
            print(f"[DEBUG]   {i+1}. {s['name']} ({s['distance']:.2f}km)")
        
        # 결과 반환
        # - success: True (검색 성공)
        # - location: 검색된 장소명 (예: "강남역")
        # - coordinates: (위도, 경도) 튜플
        # - shelters: 가장 가까운 대피소 5개 리스트
        # - total_count: VectorDB에 저장된 전체 대피소 개수
        return LocationExtractResponse(
            success=True,
            location=place_name,
            coordinates=(lat, lon),
            shelters=top_shelters,
            total_count=len(all_metadatas),
            message="OK"
        )
        
    # CASE 3: 기타 질문
    else:
        return LocationExtractResponse(success=False, message="대피소/재난행동요령 관련 질문이 아닙니다.")


# -----------------------------------------------------------------------------
# 5. 미들웨어 설정
# -----------------------------------------------------------------------------

# CORS (Cross-Origin Resource Sharing) 설정
# 모든 도메인에서의 요청을 허용 (개발 환경 편의성)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 오리진 허용
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메서드 허용
    allow_headers=["*"],  # 모든 HTTP 헤더 허용
)


# -----------------------------------------------------------------------------
# 6. 추가 Request/Response 모델
# -----------------------------------------------------------------------------

class ChatRequest(BaseModel):
    query: str
    user_lat: Optional[float] = None # 사용자의 위도 (선택 사항)
    user_lon: Optional[float] = None # 사용자의 경도 (선택 사항)


class ChatResponse(BaseModel):
    response: str
    shelters: List[Dict]
    location: Dict


class ShelterSearchRequest(BaseModel):
    location: str # 검색할 지명
    top_k: int = 5 # 반환할 결과 개수


# -----------------------------------------------------------------------------
# 7. 기본 API 엔드포인트 (웹, 상태확인)
# -----------------------------------------------------------------------------

@app.get("/")
async def read_root():
    """
    메인 페이지 (웹 인터페이스)
    - shelter_1.0.html 파일을 제공합니다.
    """
    template_path = Path(__file__).parent / "shelter_1.0.html"
    if not template_path.exists():
        raise HTTPException(
            status_code=404, 
            detail=f"템플릿 파일을 찾을 수 없습니다: {template_path}"
        )
    return FileResponse(str(template_path))


@app.get("/api/health")
async def health_check():
    """
    서버 헬스 체크
    - 로드밸런서나 모니터링 시스템에서 서버 생존 여부를 확인할 때 사용
    """
    return {
        "status": "ok",
        "vectorstore_ready": vectorstore is not None,
        "shelter_data_ready": shelter_df is not None
    }


@app.get("/api/status")
async def get_api_status():
    """
    상세 API 상태 확인
    - DB 로드 상태, LLM API 키 존재 여부 등 시스템 전반적인 상태 반환
    """
    # OPENAI_API_KEY 확인 (환경변수)
    openai_available = bool(os.getenv("OPENAI_API_KEY"))
    
    return {
        "server_ready": True,
        "llm_available": openai_available,
        "vectorstore_ready": vectorstore is not None,
        "total_shelters": len(shelter_df) if shelter_df is not None else 0,
        "shelter_data_ready": shelter_df is not None
    }


# -----------------------------------------------------------------------------
# 8. 대피소 조회/검색 API (현재 사용하지 않음 - /api/location/extract로 통합됨)
# -----------------------------------------------------------------------------

# @app.get("/api/shelters/all") - 사용 안 함
# @app.post("/api/shelters/search") - 사용 안 함

@app.get("/api/shelters/nearest")
async def get_nearest_shelters(lat: float, lon: float, k: int = 5):
    """
    현위치 기준 가장 가까운 대피소 검색
    - VectorStore의 메타데이터를 활용한 거리 계산 방식 사용
    - shelter 타입 문서들의 메타데이터에서 좌표 정보를 추출하여 거리 계산
    """
    print(f"[API] get_nearest_shelters 호출됨: lat={lat}, lon={lon}, k={k}")
    print(f"[API] shelter_df 상태: {shelter_df is not None}")
    print(f"[API] vectorstore 상태: {vectorstore is not None}")
    import math

    # 하버사인(Haversine) 공식: 구면상의 두 점 사이의 최단 거리 계산
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371  # 지구 반지름 (km)
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        d_phi = math.radians(lat2 - lat1)
        d_lambda = math.radians(lon2 - lon1)
        a = math.sin(d_phi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(d_lambda/2)**2
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

    # VectorStore 사용 가능 여부 확인
    if vectorstore is None:
        print("[DEBUG] VectorStore가 없어서 shelter_df를 사용합니다.")
        # shelter_df로 폴백 (기존 로직)
        if shelter_df is None:
            return {"user_location": {"lat": lat, "lon": lon}, "shelters": [], "total_count": 0}
        
        shelters = []
        for _, row in shelter_df.iterrows():
            s_lat = row.get('위도(EPSG4326)')
            s_lon = row.get('경도(EPSG4326)')
            
            if s_lat is not None and s_lon is not None:
                try:
                    s_lat = float(s_lat)
                    s_lon = float(s_lon)
                    distance = haversine(lat, lon, s_lat, s_lon)
                    
                    shelters.append({
                        'name': row.get('시설명', 'N/A'),
                        'address': row.get('도로명전체주소', 'N/A'),
                        'lat': s_lat,
                        'lon': s_lon,
                        'capacity': int(row.get('최대수용인원', 0)) if pd.notna(row.get('최대수용인원')) else 0,
                        'distance': distance
                    })
                except Exception:
                    continue
        
        shelters.sort(key=lambda x: x['distance'])
        top_shelters = shelters[:k]
        
        return {
            "user_location": {"lat": lat, "lon": lon},
            "shelters": top_shelters,
            "total_count": len(top_shelters)
        }
    
    # VectorStore를 사용한 대피소 검색
    try:
        print(f"[DEBUG] vectorstore 객체 타입: {type(vectorstore)}")
        print(f"[DEBUG] vectorstore._collection이 있는지: {hasattr(vectorstore, '_collection')}")
        
        # 컬렉션의 전체 문서 수 확인
        collection_count = vectorstore._collection.count()
        print(f"[DEBUG] vectorstore 컬렉션에 {collection_count}개 문서가 저장되어 있습니다.")
        
        # 1. VectorStore에서 shelter 타입 문서만 필터링하여 가져오기
        # where 조건으로 shelter 타입만 필터링
        all_data = vectorstore.get(
            where={"type": "shelter"}
        )
        print(f"[DEBUG] vectorstore.get() 결과: {type(all_data)}, 키들: {all_data.keys() if isinstance(all_data, dict) else 'dict가 아님'}")
        
        all_metadatas = all_data.get('metadatas', [])
        
        print(f"[DEBUG] VectorStore에서 {len(all_metadatas)}개 문서 메타데이터를 가져왔습니다.")
        
        # 디버깅: shelter 타입 문서 개수 확인
        shelter_count = sum(1 for m in all_metadatas if m.get('type') == 'shelter')
        print(f"[DEBUG] VectorStore에 shelter 타입 문서가 {shelter_count}개 있습니다.")
        
        # 디버깅: 첫 번째 shelter 메타데이터 키 확인
        if all_metadatas:
            first_shelter = next((m for m in all_metadatas if m.get('type') == 'shelter'), None)
            if first_shelter:
                print(f"[DEBUG] 첫 번째 shelter 메타데이터 키들: {list(first_shelter.keys())}")
                print(f"[DEBUG] facility_name 값: {first_shelter.get('facility_name', 'KEY 없음')}")
                print(f"[DEBUG] address 값: {first_shelter.get('address', 'KEY 없음')}")
        
        shelters = []
        
        # 2. shelter 타입 문서들만 필터링하고 거리 계산
        for metadata in all_metadatas:
            # shelter 타입 문서만 처리
            if metadata.get('type') != 'shelter':
                continue
                
            # 좌표 정보 추출 (documents.py에서 영문 키로 저장됨)
            s_lat = metadata.get('lat')
            s_lon = metadata.get('lon')
            
            if s_lat is not None and s_lon is not None:
                try:
                    s_lat = float(s_lat)
                    s_lon = float(s_lon)
                    distance = haversine(lat, lon, s_lat, s_lon)
                    
                    # 대피소 정보 구성 (documents.py의 영문 키 사용)
                    shelter_info = {
                        'name': metadata.get('facility_name', 'N/A'),
                        'address': metadata.get('address', 'N/A'),
                        'lat': s_lat,
                        'lon': s_lon,
                        'capacity': int(metadata.get('capacity', 0)),
                        'distance': distance
                    }
                    shelters.append(shelter_info)
                    
                except (ValueError, TypeError) as e:
                    print(f"[WARNING] 좌표 변환 오류: {e}")
                    continue
        
        print(f"[DEBUG] 유효한 대피소 {len(shelters)}개를 찾았습니다.")
        
        # 3. 거리순 정렬 및 상위 k개 선택
        shelters.sort(key=lambda x: x['distance'])
        top_shelters = shelters[:k]
        
        return {
            "user_location": {"lat": lat, "lon": lon},
            "shelters": top_shelters,
            "total_count": len(top_shelters)
        }
        
    except Exception as e:
        print(f"[ERROR] VectorStore 사용 중 오류: {e}")
        # 오류 발생 시 빈 결과 반환
        return {"user_location": {"lat": lat, "lon": lon}, "shelters": [], "total_count": 0}


# -----------------------------------------------------------------------------
# 9. LangGraph Agent 기반 챗봇 엔드포인트
# -----------------------------------------------------------------------------

@app.post("/api/chatbot", response_model=ChatbotResponse)
async def chatbot_endpoint(request: ChatbotRequest):
    """
    LangGraph Agent 기반 고급 챗봇 엔드포인트
    
    특징:
    - 하이브리드 검색 (Vector + BM25)
    - 질문 재정의 (Query Rewriting)
    - Agent + Tools 아키텍처
    - 통계 기능 (수용인원 집계)
    - 세션 기반 대화 기록 유지
    
    Args:
        request: ChatbotRequest (message, session_id)
    
    Returns:
        ChatbotResponse (response, session_id)
    """
    try:
        if langgraph_app is None:
            raise HTTPException(
                status_code=503, 
                detail="챗봇 시스템이 초기화되지 않았습니다. 서버를 재시작해주세요."
            )
        
        # 세션 설정
        config = {"configurable": {"thread_id": request.session_id}}
        
        # LangGraph Agent 실행
        result = langgraph_app.invoke(
            {"messages": [HumanMessage(content=request.message)]},
            config=config
        )
        
        # 마지막 AI 메시지 추출
        bot_response = result["messages"][-1].content
        
        return ChatbotResponse(
            response=bot_response,
            session_id=request.session_id
        )
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] 챗봇 오류: {e}")
        raise HTTPException(status_code=500, detail=f"챗봇 처리 중 오류가 발생했습니다: {str(e)}")


# -----------------------------------------------------------------------------
# 10. 서버 실행
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # HTTPS 지원 서버 실행
    # SSL 인증서 경로 설정
    cert_dir = "shelter_chatbot/cert"
    cert_file = f"{cert_dir}/cert.pem"
    key_file = f"{cert_dir}/key.pem"
    
    # 인증서 파일 존재 확인
    import os
    if os.path.exists(cert_file) and os.path.exists(key_file):
        print(f"[INFO] SSL 인증서를 사용하여 HTTPS 서버 시작")
        # print(f"[INFO] 주소: https://61.78.100.228:8443/")
        print(f"[INFO] 주소: https://222.106.254.193:8443/")
        
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8443,
            ssl_keyfile=key_file,
            ssl_certfile=cert_file,
            reload=False,
            log_level="info"
        )
    else:
        print(f"[WARNING] SSL 인증서 파일을 찾을 수 없습니다.")
        print(f"[INFO] HTTP 서버로 시작합니다.")
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8443,
            reload=False,
            log_level="info"
        )