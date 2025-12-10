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

# LangGraph Agent import
from langgraph_agent import create_langgraph_app, create_hybrid_retrievers

# LangChain imports for API
from langchain_core.messages import HumanMessage

# Simple Pydantic models for API
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
# 3. FastAPI Lifespan (수명 주기) 핸들러
# -----------------------------------------------------------------------------


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
        shelter_hybrid_retriever, guideline_hybrid_retriever = create_hybrid_retrievers(vectorstore)
        langgraph_app = create_langgraph_app(vectorstore)
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
    사용자 질의를 LangGraph Agent가 처리합니다.
    
    =========================================================================
    [간소화된 처리 흐름]
    =========================================================================
    
    사용자 쿼리 → LangGraph Agent → 구조화된 응답 반환
    
    - Agent가 자동으로 의도 분류 및 도구 선택
    - 지도 표시용 좌표/대피소 데이터 포함 가능
    - 재난 행동요령, 통계, 일반 대화 모두 처리
    
    =========================================================================
    """
    
    # 리소스 확인
    if langgraph_app is None:
        return LocationExtractResponse(
            success=False, 
            message="서버 초기화가 완료되지 않았습니다."
        )
    
    # 쿼리 유효성 검사
    query = request.query.strip()
    if not query:
        return LocationExtractResponse(
            success=False, 
            message="입력 문장이 비어 있습니다."
        )

    print(f"[API] 사용자 쿼리: '{query}'")
    
    try:
        # LangGraph Agent 실행
        session_id = f"session_{hash(query) % 100000}"
        config = {"configurable": {"thread_id": session_id}}
        
        result = langgraph_app.invoke(
            {"messages": [HumanMessage(content=query)]},
           
            config=config
        )
        
        # Agent의 최종 응답 추출
        final_message = result["messages"][-1]
        
        # Agent가 구조화된 데이터를 반환했는지 확인
        # (structured_data가 state에 있으면 지도 표시용 데이터 포함)
        structured_data = result.get("structured_data", None)
        
        if structured_data:
            # 지도 표시용 데이터가 있는 경우
            print(f"[INFO] 구조화된 응답 반환 (좌표 포함)")
            return LocationExtractResponse(
                success=True,
                location=structured_data.get("location"),
                coordinates=structured_data.get("coordinates"),
                shelters=structured_data.get("shelters", []),
                total_count=structured_data.get("total_count", 0),
                message=final_message.content
            )
        else:
            # 텍스트 응답만 있는 경우
            print(f"[INFO] 텍스트 응답 반환 (길이: {len(final_message.content)})")
            return LocationExtractResponse(
                success=True,
                location=None,
                coordinates=None,
                shelters=[],
                total_count=0,
                message=final_message.content
            )
    
    except Exception as e:
        print(f"[ERROR] LangGraph Agent 실행 실패: {e}")
        import traceback
        traceback.print_exc()
        return LocationExtractResponse(
            success=False,
            message="처리 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요."
        )



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
        print(f"[INFO] 주소: https://61.78.100.233:8443/")
        
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