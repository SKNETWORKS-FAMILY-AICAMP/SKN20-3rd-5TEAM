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
from langchain_openai import OpenAIEmbeddings
import pandas as pd
import json
import re

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
# 3. FastAPI Lifespan (수명 주기) 핸들러
# -----------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    서버 시작/종료 시 실행되는 초기화 및 정리 작업
    앱 실행 시:
    - Vector DB 로드 및 초기화
    - 대피소 데이터 로드
    앱 종료 시:
    - 리소스 정리 (현재는 별도 정리 작업 없음)
    """
    global vectorstore, shelter_df, embeddings
    
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
    
    yield # 애플리케이션 실행 중
    
    # 여기에 종료 시 필요한 정리 작업 코드를 추가할 수 있음

# FastAPI 앱 인스턴스 생성 및 lifespan 핸들러 연결
app = FastAPI(title="대피소 안내 챗봇 API", lifespan=lifespan)

# 전역 변수 초기화
vectorstore = None
shelter_df = None
embeddings = None




# -----------------------------------------------------------------------------
# 4. API 엔드포인트: 지명 추출 및 통합 검색
# -----------------------------------------------------------------------------

@app.post("/api/location/extract")
async def extract_location(request: LocationExtractRequest = Body(...)):
    """
    사용자 질의(Query)를 분석하여 의도(Intent)를 파악하고 적절한 응답을 제공합니다.
    
    처리 로직:
    1. **1차: LLM 기반 의도 분류**
       - classify_user_intent() 함수를 사용하여 의도를 파악합니다.
       - find_shelter: 대피소 찾기
       - disaster_guide: 재난행동요령
       - general_chat: 일반 대화
    
    2. **2차: 의도별 분기 처리**
       - **재난행동요령 (disaster_guide)**:
         - Vector DB에서 관련 재난행동요령 문서를 검색하여 반환합니다.
       - **대피소 검색 (find_shelter)**:
         - 카카오 로컬 API를 사용하여 지명의 위/경도 좌표를 얻습니다.
         - 현재 위치 기준으로 가장 가까운 대피소 5곳을 반환합니다.
       - **일반 대화 (general_chat)**:
         - 안내 메시지를 반환합니다.
    """
    
    # 리소스 확인
    if vectorstore is None or shelter_df is None:
        return LocationExtractResponse(success=False, message="서버 초기화가 완료되지 않았습니다.")
    
    # 쿼리 유효성 검사
    query = request.query.strip()
    if not query:
        return LocationExtractResponse(success=False, message="입력 문장이 비어 있습니다.")

    # -----------------------
    # 1차: LLM 기반 의도 분류
    # -----------------------
    print(f"[API] 사용자 쿼리: '{query}'")
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
        
        # 사용자 쿼리에서 순수 지명만 추출 (불필요한 단어 제거)
        location_query = query
        # 대피소 관련 키워드 제거
        remove_keywords = [
            "대피소", "피난소", "피난처", "근처", "주변", "가까운", "어디", "위치",
            "찾아줘", "알려줘", "검색", "보여줘", "있어", "는?", "은?", "?", "!",
            "좀", "요", "주세요", "해줘", "해주세요", "있나요", "있어요"
        ]
        
        for keyword in remove_keywords:
            location_query = location_query.replace(keyword, "")
        
        # 공백 정리
        location_query = " ".join(location_query.split()).strip()
        
        print(f"[DEBUG] 정제된 위치 쿼리: '{location_query}'")
        
        # 정제 후 비어있으면 원본 쿼리 사용
        if not location_query:
            location_query = query
            print(f"[DEBUG] 정제 결과가 비어있어 원본 쿼리 사용")
        
        # 카카오 로컬 API 키 확인
        kakao_key = os.getenv("KAKAO_REST_API_KEY")
        if not kakao_key:
            print(f"[ERROR] KAKAO_REST_API_KEY 없음")
            return LocationExtractResponse(success=False, message="KAKAO_REST_API_KEY 환경변수가 없습니다.")
        
        # 여러 지명이 포함된 경우, 각 지명을 카카오 API로 검색하여 랜드마크 우선 선택
        location_parts = location_query.split()
        selected_location = location_query
        
        if len(location_parts) > 1:
            print(f"[DEBUG] 여러 지명 감지: {location_parts}, 카카오 API로 우선순위 판단")
            
            url = "https://dapi.kakao.com/v2/local/search/keyword.json"
            headers = {"Authorization": f"KakaoAK {kakao_key}"}
            
            best_candidate = None
            best_priority = 999
            
            # 우선순위: 1=관광명소/문화시설, 2=교통시설(역), 3=행정구역, 4=기타
            priority_categories = {
                1: ["관광명소", "문화시설", "여가시설", "공공기관", "테마파크"],
                2: ["교통,수송", "지하철역"],
                3: ["행정구역"],
            }
            
            for part in location_parts:
                resp = requests.get(url, headers=headers, params={"query": part, "size": 5})
                if resp.status_code == 200:
                    docs = resp.json().get("documents", [])
                    if docs:
                        doc = docs[0]
                        category_name = doc.get("category_name", "")
                        print(f"[DEBUG] '{part}' 검색 결과 - category: {category_name}")
                        
                        # 카테고리 우선순위 판단
                        priority = 4  # 기본값
                        for pri, keywords in priority_categories.items():
                            if any(keyword in category_name for keyword in keywords):
                                priority = pri
                                break
                        
                        # 더 높은 우선순위(낮은 숫자)면 선택
                        if priority < best_priority:
                            best_priority = priority
                            best_candidate = part
                            print(f"[DEBUG] 우선순위 {priority}: '{part}' 선택 (category: {category_name})")
            
            if best_candidate:
                selected_location = best_candidate
                print(f"[DEBUG] 최종 선택된 위치: '{selected_location}' (우선순위: {best_priority})")
            else:
                # API 검색 실패시 첫 번째 지명 사용
                selected_location = location_parts[0]
                print(f"[DEBUG] API 검색 실패, 첫 번째 지명 사용: '{selected_location}'")
        
        location_query = selected_location
        
        # 카카오 API를 사용하여 최종 위치 검색
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
        
        if not data.get("documents"):
            print(f"[ERROR] 카카오 API에서 '{location_query}' 위치를 찾지 못함")
            return LocationExtractResponse(success=False, message=f"'{location_query}' 위치를 찾을 수 없습니다. 다른 지역명을 입력해 주세요.")
            
        # 좌표 추출
        place = data["documents"][0]
        lat = float(place["y"])
        lon = float(place["x"])
        place_name = place.get("place_name", location_query)
        
        print(f"[DEBUG] 좌표 추출 성공 - place_name: {place_name}, lat: {lat}, lon: {lon}")
        
        # VectorStore에서 모든 대피소 데이터 가져와서 거리 계산 (Haversine 공식)
        import math
        
        def haversine(lat1, lon1, lat2, lon2):
            """구면상의 두 점 사이의 최단 거리 계산 (단위: km)"""
            R = 6371  # 지구 반지름 (km)
            phi1, phi2 = math.radians(lat1), math.radians(lat2)
            d_phi = math.radians(lat2 - lat1)
            d_lambda = math.radians(lon2 - lon1)
            a = math.sin(d_phi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(d_lambda/2)**2
            return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        # VectorStore에서 shelter 타입 문서만 필터링하여 가져오기
        all_data = vectorstore.get(where={"type": "shelter"})
        all_metadatas = all_data.get('metadatas', [])
        
        print(f"[DEBUG] VectorStore에서 {len(all_metadatas)}개 대피소 메타데이터 가져옴")
        
        shelters = []
        for metadata in all_metadatas:
            if metadata.get('type') != 'shelter':
                continue
                
            # 좌표 정보 추출 (영문 키 사용)
            s_lat = metadata.get('lat')
            s_lon = metadata.get('lon')
            
            if s_lat is not None and s_lon is not None:
                try:
                    s_lat = float(s_lat)
                    s_lon = float(s_lon)
                    distance = haversine(lat, lon, s_lat, s_lon)
                    
                    shelter_info = {
                        'name': metadata.get('facility_name', 'N/A'),
                        'address': metadata.get('address', 'N/A'),
                        'lat': s_lat,
                        'lon': s_lon,
                        'capacity': int(metadata.get('capacity', 0)),
                        'distance': distance
                    }
                    shelters.append(shelter_info)
                except (ValueError, TypeError):
                    continue
        
        print(f"[DEBUG] 총 {len(shelters)}개 대피소의 거리 계산 완료")
        
        # 거리순 정렬 후 상위 5개 반환
        shelters.sort(key=lambda x: x['distance'])
        top_shelters = shelters[:5]
        
        print(f"[DEBUG] 상위 5개 대피소 선택 완료")
        for i, s in enumerate(top_shelters):
            print(f"[DEBUG]   {i+1}. {s['name']} ({s['distance']:.2f}km)")
        
        return LocationExtractResponse(
            success=True,
            location=place_name,
            coordinates=(lat, lon),
            shelters=top_shelters,
            total_count=len(all_metadatas),  # 전체 대피소 개수
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

# 사용되지 않는 /api/chat 엔드포인트 제거됨

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
        print(f"[INFO] 주소: https://61.78.100.228:8443/")
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