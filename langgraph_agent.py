# -*- coding: utf-8 -*-
"""
LangGraph Agent 및 Tools 정의
대피소 검색, 재난 행동요령, 통계 기능을 담당하는 AI Agent
"""

import os
import requests
import json
import re
from math import radians, sin, cos, sqrt, atan2
from typing import TypedDict, Annotated, Optional

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver


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


def create_hybrid_retrievers(vectorstore):
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


def create_langgraph_app(vectorstore):
    """LangGraph Agent 생성"""
    
    # 1. LLM 초기화
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    llm_creative = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)  # 일반 지식용
    
    # 2. 의도 분류 체인
    # 2. 의도 분류 체인
    intent_classification_prompt = ChatPromptTemplate.from_messages([
        ("system", """당신은 사용자 질문의 의도를 정확하게 분류하는 AI입니다.

    질문을 다음 카테고리 중 하나로 분류하세요:

    1. **hybrid_location_disaster**: 위치 + 재난 상황 복합 질문 ⭐ 우선순위 1
    - 예: "설악산 근처인데 산사태 발생 시", "강남역에서 지진 나면", "명동 화재"
    - 키워드: 지명 + (지진/화재/산사태/홍수 등)

    2. **shelter_info**: 특정 대피소의 상세 정보 조회 ⭐ 새로 추가
    - 예: "동대문맨션 수용인원", "서울역 대피소 정보", "롯데월드 최대 수용"
    - 키워드: 시설명 + (수용인원/정보/면적 등)

    3. **shelter_search**: 특정 위치의 대피소 찾기
    - 예: "한라산 근처 대피소", "강남역 대피소"
    - 키워드: 지명 + (근처/주변/대피소) WITHOUT 재난 키워드
    
    4. **shelter_count**: 특정 조건의 대피소 개수 세기
    - 예: "서울 대피소 개수", "지하 대피소 몇 개"
    
    5. **shelter_capacity**: 수용인원 기준 대피소 찾기
    - 예: "천 명 이상 수용 가능한 대피소"
    - 키워드: 숫자 + (이상/이하/수용)
    
    6. **disaster_guideline**: 재난 행동요령만 질문
    - 예: "지진 발생 시 행동요령" (위치 정보 없음)
    
    7. **general_knowledge**: 재난 관련 일반 지식
    - 예: "지진이 뭐야", "쓰나미란"
    
    8. **general_chat**: 일반 대화
    - 예: "안녕", "고마워"

    **중요 우선순위**: 
    - "위치 + 재난"이 함께 있으면 무조건 **hybrid_location_disaster**
    - "시설명 + 수용인원/정보"는 **shelter_info**
    - "위치 + 근처/주변"만 있고 재난 없으면 **shelter_search**

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
    shelter_hybrid, guideline_hybrid = create_hybrid_retrievers(vectorstore)
    
    # 5. Tools 정의
    @tool
    def search_shelter_by_location(query: str) -> dict:
        """
        특정 위치의 대피소를 검색합니다.
        카카오 API로 좌표를 찾고, 가장 가까운 대피소 5곳을 반환합니다.
        지도 표시용 구조화된 데이터를 포함합니다.
        
        Args:
            query: 위치 정보 (지명, 건물명, 주소 등)
        
        Returns:
            dict: {"text": str, "structured_data": dict} 형식
        """
        try:
            # 쿼리 재정의
            rewritten = query_rewrite_chain.invoke({"original_query": query})
            print(f"[search_shelter_by_location] 재정의: {query} → {rewritten}")
            
            # 카카오 API로 좌표 검색
            kakao_api_key = os.getenv("KAKAO_REST_API_KEY")
            if not kakao_api_key:
                return {"text": "카카오 API 키가 설정되지 않았습니다.", "structured_data": None}
            
            headers = {"Authorization": f"KakaoAK {kakao_api_key}"}
            url = "https://dapi.kakao.com/v2/local/search/keyword.json"
            params = {"query": rewritten}
            
            response = requests.get(url, headers=headers, params=params)
            data = response.json()
            
            if not data.get("documents"):
                return {"text": f"'{query}' 위치를 찾을 수 없습니다.", "structured_data": None}
             
            place = data["documents"][0]
            user_lat = float(place["y"])
            user_lon = float(place["x"])
            place_name = place["place_name"]
            
            print(f"[search_shelter_by_location] 좌표: {place_name} ({user_lat}, {user_lon})")
            
            # Haversine 거리 계산
            def haversine(lat1, lon1, lat2, lon2):
                R = 6371
                dlat = radians(lat2 - lat1)
                dlon = radians(lon2 - lon1)
                a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
                c = 2 * atan2(sqrt(a), sqrt(1-a))
                return R * c
            
            # 모든 대피소 가져오기
            all_data = vectorstore.get(where={"type": "shelter"})
            shelters = []

            for metadata in all_data['metadatas']:
                try:
                    lat = float(metadata.get('lat', 0))
                    lon = float(metadata.get('lon', 0))
                    if lat == 0 or lon == 0:
                        continue
                    
                    distance = haversine(user_lat, user_lon, lat, lon)
                    shelters.append({
                        'name': metadata.get('facility_name', 'N/A'),
                        'address': metadata.get('address', 'N/A'),
                        'lat': lat,
                        'lon': lon,
                        'distance': distance,
                        'capacity': int(metadata.get('capacity', 0)),
                        'shelter_type': metadata.get('shelter_type', 'N/A'),
                        'facility_type': metadata.get('facility_type', 'N/A')
                    })
                except Exception:
                    continue
            
            # 거리순 정렬
            shelters.sort(key=lambda x: x['distance'])
            top_5 = shelters[:5]
            
            if not top_5:
                return {
                    "text": f"'{place_name}' 근처에 대피소를 찾을 수 없습니다.",
                    "structured_data": None
                }
            
            # 텍스트 결과 포맷팅
            result_text = f"📍 **{place_name}** 근처 대피소 {len(top_5)}곳\n\n"
            for i, s in enumerate(top_5, 1):
                result_text += f"{i}. **{s['name']}**\n"
                result_text += f"   📍 거리: {s['distance']:.2f}km\n"
                result_text += f"   📍 주소: {s['address']}\n"
                result_text += f"   📍 위치: {s['shelter_type']}\n"
                result_text += f"   📍 수용인원: {s['capacity']:,}명\n\n"
            
            # 구조화된 데이터 (지도 표시용)
            structured_data = {
                "location": place_name,
                "coordinates": (user_lat, user_lon),
                "shelters": top_5,
                "total_count": len(all_data['metadatas'])
            }
            
            return {
                "text": result_text.strip(),
                "structured_data": structured_data
            }
            
        except Exception as e:
            print(f"[ERROR] search_shelter_by_location: {e}")
            import traceback
            traceback.print_exc()
            return {
                "text": f"검색 중 오류 발생: {str(e)}",
                "structured_data": None
            }
    
    @tool
    def count_shelters(query: str) -> dict:
        """
        특정 조건(지역, 위치유형 등)에 맞는 대피소 개수를 셉니다.
        지도 표시용 구조화된 데이터를 포함합니다.
        
        Args:
            query: 검색 조건 (예: "서울 지하", "부산 민방위")
        
        Returns:
            dict: {"text": str, "structured_data": dict} 형식
        """
        try:
            # 쿼리 재정의
            rewritten = query_rewrite_chain.invoke({"original_query": query})
            print(f"[count_shelters] 재정의: {query} → {rewritten}")
            
            if shelter_hybrid is None:
                return {
                    "text": "검색 시스템이 초기화되지 않았습니다.",
                    "structured_data": None
                }
            
            # 1단계: VectorDB 전체에서 매칭되는 대피소 찾기 (전체 개수 카운트용)
            all_data = vectorstore.get(where={"type": "shelter"})
            all_shelters = []
            
            # 검색 키워드 추출 (공백으로 분리)
            search_keywords = rewritten.lower().split()
            
            for metadata in all_data['metadatas']:
                # 시설명, 주소, 위치유형에서 키워드 검색
                facility_name = metadata.get('facility_name', '').lower()
                address = metadata.get('address', '').lower()
                shelter_type = metadata.get('shelter_type', '').lower()
                
                # 검색 대상 텍스트 결합
                search_text = f"{facility_name} {address} {shelter_type}"
                
                # 모든 키워드 중 하나라도 포함되면 매칭
                if any(keyword in search_text for keyword in search_keywords):
                    all_shelters.append({
                        'name': metadata.get('facility_name', 'N/A'),
                        'address': metadata.get('address', 'N/A'),
                        'lat': float(metadata.get('lat', 0)),
                        'lon': float(metadata.get('lon', 0)),
                        'distance': 0,
                        'capacity': int(metadata.get('capacity', 0)),
                        'shelter_type': metadata.get('shelter_type', 'N/A'),
                        'facility_type': metadata.get('facility_type', 'N/A')
                    })
            
            total_count = len(all_shelters)
            
            # 2단계: 하이브리드 검색으로 상위 결과 추출 (지도 표시용)
            results = shelter_hybrid.invoke(rewritten)
            
            # 중복 제거 및 대피소 정보 수집
            seen = set()
            top_shelters = []
            for doc in results:
                name = doc.metadata.get('facility_name', '')
                if name and name not in seen:
                    seen.add(name)
                    top_shelters.append({
                        'name': name,
                        'address': doc.metadata.get('address', 'N/A'),
                        'lat': float(doc.metadata.get('lat', 0)),
                        'lon': float(doc.metadata.get('lon', 0)),
                        'distance': 0,
                        'capacity': int(doc.metadata.get('capacity', 0)),
                        'shelter_type': doc.metadata.get('shelter_type', 'N/A'),
                        'facility_type': doc.metadata.get('facility_type', 'N/A')
                    })
                    if len(top_shelters) >= 10:  # 최대 10개
                        break
            
            if total_count == 0:
                return {
                    "text": f"'{query}' 조건에 맞는 대피소를 찾을 수 없습니다.",
                    "structured_data": None
                }
            
            # 중심 좌표 계산 (평균)
            display_shelters = top_shelters if top_shelters else all_shelters[:10]
            avg_lat = sum(s['lat'] for s in display_shelters if s['lat'] != 0) / len([s for s in display_shelters if s['lat'] != 0]) if any(s['lat'] != 0 for s in display_shelters) else 0
            avg_lon = sum(s['lon'] for s in display_shelters if s['lon'] != 0) / len([s for s in display_shelters if s['lon'] != 0]) if any(s['lon'] != 0 for s in display_shelters) else 0
            
            structured_data = {
                "location": query,
                "coordinates": (avg_lat, avg_lon) if avg_lat != 0 else None,
                "shelters": display_shelters,  # 지도에 표시할 10개
                "total_count": total_count  # VectorDB 전체 매칭 개수
            }
            
            return {
                "text": f"**'{query}'** 조건에 맞는 대피소는 총 **{total_count}개**입니다. 📊",
                "structured_data": structured_data
            }
            
        except Exception as e:
            print(f"[ERROR] count_shelters: {e}")
            import traceback
            traceback.print_exc()
            return {
                "text": f"검색 중 오류 발생: {str(e)}",
                "structured_data": None
            }
    
    @tool
    def search_shelter_by_capacity(query: str) -> dict:
        """
        수용인원 기준으로 대피소를 검색합니다.
        지도 표시용 구조화된 데이터를 포함합니다.
        
        Args:
            query: 수용인원 조건 (예: "천 명 이상", "100명 수용 가능")
        
        Returns:
            dict: {"text": str, "structured_data": dict} 형식
        """
        try:
            # 숫자 추출
            numbers = re.findall(r'\d+', query)
            if not numbers:
                return {
                    "text": "수용인원을 명확히 입력해주세요. (예: 1000명 이상)",
                    "structured_data": None
                }
            
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
                        'lat': float(metadata.get('lat', 0)),
                        'lon': float(metadata.get('lon', 0)),
                        'capacity': capacity,
                        'shelter_type': metadata.get('shelter_type', 'N/A'),
                        'distance': 0  # 수용인원 검색은 거리 정보 없음
                    })
            
            # 수용인원 내림차순 정렬
            shelters.sort(key=lambda x: x['capacity'], reverse=True)
            top_10 = shelters[:10]
            
            if not top_10:
                return {
                    "text": f"{min_capacity:,}명 이상 수용 가능한 대피소를 찾을 수 없습니다.",
                    "structured_data": None
                }
            
            result = f"📊 **{min_capacity:,}명 이상** 수용 가능한 대피소 **{len(shelters)}곳** 중 상위 10곳\n\n"
            for i, s in enumerate(top_10, 1):
                result += f"{i}. **{s['name']}** ({s['capacity']:,}명)\n"
                result += f"   📍 {s['address']}\n"
                result += f"   📍 위치: {s['shelter_type']}\n\n"
            
            # 중심 좌표 계산 (평균)
            avg_lat = sum(s['lat'] for s in shelters if s['lat'] != 0) / len([s for s in shelters if s['lat'] != 0]) if any(s['lat'] != 0 for s in shelters) else 0
            avg_lon = sum(s['lon'] for s in shelters if s['lon'] != 0) / len([s for s in shelters if s['lon'] != 0]) if any(s['lon'] != 0 for s in shelters) else 0
            
            structured_data = {
                "location": f"{min_capacity:,}명 이상 수용 가능",
                "coordinates": (avg_lat, avg_lon) if avg_lat != 0 else None,
                "shelters": top_10,
                "total_count": len(shelters)
            }
            
            return {
                "text": result.strip(),
                "structured_data": structured_data
            }
            
        except Exception as e:
            print(f"[ERROR] search_shelter_by_capacity: {e}")
            return {
                "text": f"검색 중 오류 발생: {str(e)}",
                "structured_data": None
            }
    
    @tool
    def search_disaster_guideline(query: str) -> dict:
        """
        재난 행동요령을 검색합니다.
        
        Args:
            query: 재난 유형 (예: "지진", "화재", "산사태")
        
        Returns:
            dict: {"text": str, "structured_data": None} 형식
        """
        try:
            # 쿼리 재정의
            rewritten = query_rewrite_chain.invoke({"original_query": query})
            print(f"[search_disaster_guideline] 재정의: {query} → {rewritten}")
            
            if guideline_hybrid is None:
                return {
                    "text": "가이드라인 검색 시스템이 초기화되지 않았습니다.",
                    "structured_data": None
                }
            
            # 하이브리드 검색
            results = guideline_hybrid.invoke(rewritten)
            
            if not results:
                return {
                    "text": f"'{query}' 관련 행동요령을 찾을 수 없습니다.",
                    "structured_data": None
                }
            
            # 상위 3개 결과 통합
            combined = "\n\n".join([doc.page_content for doc in results[:3]])
            
            return {
                "text": f"🚨 **{query} 행동요령**\n\n{combined}",
                "structured_data": None  # 행동요령은 위치 정보 없음
            }
            
        except Exception as e:
            print(f"[ERROR] search_disaster_guideline: {e}")
            return {
                "text": f"검색 중 오류 발생: {str(e)}",
                "structured_data": None
            }
    
    @tool
    def answer_general_knowledge(query: str) -> dict:
        """
        재난 관련 일반 지식 질문에 답변합니다. (정의, 원인, 특징 등)
        VectorDB에 없는 정보는 LLM의 사전 학습 지식을 활용합니다.
        
        Args:
            query: 일반 지식 질문 (예: "지진이 뭐야", "쓰나미란")
        
        Returns:
            dict: {"text": str, "structured_data": None} 형식
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
            
            return {
                "text": f"💡 **{query}**\n\n{response.content}",
                "structured_data": None  # 일반 지식은 위치 정보 없음
            }
            
        except Exception as e:
            print(f"[ERROR] answer_general_knowledge: {e}")
            return {
                "text": "죄송합니다. 답변 생성 중 오류가 발생했습니다.",
                "structured_data": None
            }
    @tool
    def search_shelter_by_name(query: str) -> dict:
        """
        특정 대피소의 상세 정보를 시설명으로 검색합니다.
        수용인원, 주소, 위치 등 해당 시설의 모든 정보를 반환합니다.
        지도 표시용 구조화된 데이터를 포함합니다.
        
        Args:
            query: 대피소 시설명 (예: "동대문맨션", "서울역", "롯데월드")
        
        Returns:
            dict: {"text": str, "structured_data": dict} 형식
        
        Examples:
            - "동대문맨션 수용인원" → search_shelter_by_name("동대문맨션")
            - "서울역 대피소 정보" → search_shelter_by_name("서울역")
        """
        try:
            print(f"[search_shelter_by_name] 검색 시작: '{query}'")
            
            # VectorStore에서 shelter 타입 문서 가져오기
            all_data = vectorstore.get(where={"type": "shelter"})
            
            # 검색어 정제 (불필요한 단어 제거)
            search_term = query.strip().lower()
            remove_words = ["대피소", "수용인원", "최대수용인원", "몇명", "정보", 
                        "알려줘", "의", "이", "가", "은", "는", "?", "!", " "]
            for word in remove_words:
                search_term = search_term.replace(word, "")
            search_term = search_term.strip()
            
            print(f"[search_shelter_by_name] 정제된 검색어: '{search_term}'")
            
            # 시설명 매칭 (부분 일치)
            matches = []
            for metadata in all_data['metadatas']:
                facility_name = metadata.get('facility_name', '')
                facility_lower = facility_name.lower()
                
                # 양방향 부분 일치
                if search_term in facility_lower or facility_lower in search_term:
                    matches.append({
                        'name': facility_name,
                        'address': metadata.get('address', 'N/A'),
                        'lat': float(metadata.get('lat', 0)),
                        'lon': float(metadata.get('lon', 0)),
                        'capacity': int(metadata.get('capacity', 0)),
                        'shelter_type': metadata.get('shelter_type', 'N/A'),
                        'facility_type': metadata.get('facility_type', 'N/A'),
                        'operating_status': metadata.get('operating_status', 'N/A'),
                        'distance': 0  # 시설명 검색은 거리 정보 없음
                    })
                    print(f"[search_shelter_by_name] 매칭됨: {facility_name}")
            
            if not matches:
                return {
                    "text": f"❌ '{query}' 시설을 찾을 수 없습니다.\n시설명을 정확히 입력해주세요.",
                    "structured_data": None
                }
            
            # 결과 반환
            if len(matches) == 1:
                m = matches[0]
                text = f"""📍 **{m['name']}**

    ✅ **최대 수용인원: {m['capacity']:,}명**
    📍 주소: {m['address']}
    📍 위치: {m['shelter_type']}
    📍 시설 유형: {m['facility_type']}
    📍 운영 상태: {m['operating_status']}"""
                
                # 구조화된 데이터 (지도 표시용)
                structured_data = {
                    "location": m['name'],
                    "coordinates": (m['lat'], m['lon']) if m['lat'] != 0 else None,
                    "shelters": [m],
                    "total_count": 1
                }
                
                return {
                    "text": text,
                    "structured_data": structured_data
                }
            
            else:
                # 여러 개 발견 시
                text = f"📍 **'{search_term}'** 관련 대피소 **{len(matches)}곳** 발견\n\n"
                for i, m in enumerate(matches[:5], 1):  # 상위 5개만
                    text += f"{i}. **{m['name']}**\n"
                    text += f"   ✅ 수용인원: **{m['capacity']:,}명**\n"
                    text += f"   📍 주소: {m['address']}\n"
                    text += f"   📍 위치: {m['shelter_type']}\n\n"
                
                if len(matches) > 5:
                    text += f"💡 외 {len(matches) - 5}곳 더 있습니다."
                
                # 중심 좌표 계산 (평균)
                avg_lat = sum(s['lat'] for s in matches if s['lat'] != 0) / len([s for s in matches if s['lat'] != 0]) if any(s['lat'] != 0 for s in matches) else 0
                avg_lon = sum(s['lon'] for s in matches if s['lon'] != 0) / len([s for s in matches if s['lon'] != 0]) if any(s['lon'] != 0 for s in matches) else 0
                
                structured_data = {
                    "location": search_term,
                    "coordinates": (avg_lat, avg_lon) if avg_lat != 0 else None,
                    "shelters": matches[:5],
                    "total_count": len(matches)
                }
                
                return {
                    "text": text.strip(),
                    "structured_data": structured_data
                }
            
        except Exception as e:
            print(f"[ERROR] search_shelter_by_name: {e}")
            import traceback
            traceback.print_exc()
            return {
                "text": f"❌ 검색 중 오류 발생: {str(e)}",
                "structured_data": None
            }
    @tool
    def search_location_with_disaster(query: str) -> dict:
        """
        특정 위치에서 재난 발생 시 대피소와 행동요령을 함께 제공합니다.
        위치 기반 대피소 검색 + 재난 행동요령을 통합하여 반환합니다.
        지도 표시용 구조화된 데이터를 포함합니다.
        
        Args:
            query: 위치 + 재난 상황 (예: "설악산 근처 산사태", "강남역에서 지진")
        
        Returns:
            dict: {"text": str, "structured_data": dict} 형식
        
        Examples:
            - "설악산 근처인데 산사태 발생 시" → 설악산 대피소 + 산사태 행동요령
            - "명동에서 지진 나면" → 명동 대피소 + 지진 행동요령
        """
        try:
            print(f"[search_location_with_disaster] 복합 질문 처리: {query}")
            
            # 1단계: 위치명 추출 (재난 키워드 제거)
            disaster_keywords = [
                "지진", "홍수", "태풍", "화재", "폭발", "산사태", "쓰나미", 
                "화산", "방사능", "가스", "붕괴", "테러",
                "발생", "발생하면", "발생 시", "났을 때", "나면", "때", 
                "근처인데", "에서", "어떻게", "대처", "행동요령"
            ]
            
            location_query = query
            detected_disaster = None
            
            # 재난 유형 감지 및 제거
            for keyword in disaster_keywords:
                if keyword in query:
                    if keyword in ["지진", "홍수", "태풍", "화재", "폭발", "산사태", 
                                "쓰나미", "화산", "방사능", "가스", "붕괴", "테러"]:
                        detected_disaster = keyword
                    location_query = location_query.replace(keyword, "")
            
            # 위치 쿼리 정제
            location_query = location_query.strip()
            print(f"[search_location_with_disaster] 추출된 위치: '{location_query}'")
            print(f"[search_location_with_disaster] 감지된 재난: '{detected_disaster}'")
            
            if not detected_disaster:
                return {
                    "text": "재난 유형을 파악할 수 없습니다. 예: '설악산 산사태', '강남역 지진'",
                    "structured_data": None
                }
            
            # 2단계: 카카오 API로 위치 좌표 검색
            kakao_api_key = os.getenv("KAKAO_REST_API_KEY")
            if not kakao_api_key:
                return {
                    "text": "카카오 API 키가 설정되지 않았습니다.",
                    "structured_data": None
                }
            
            headers = {"Authorization": f"KakaoAK {kakao_api_key}"}
            url = "https://dapi.kakao.com/v2/local/search/keyword.json"
            params = {"query": location_query}
            
            response = requests.get(url, headers=headers, params=params)
            data = response.json()
            
            if not data.get("documents"):
                return {
                    "text": f"'{location_query}' 위치를 찾을 수 없습니다.",
                    "structured_data": None
                }
            
            # 좌표 추출
            place = data["documents"][0]
            user_lat = float(place["y"])
            user_lon = float(place["x"])
            place_name = place["place_name"]
            
            print(f"[search_location_with_disaster] 좌표: {place_name} ({user_lat}, {user_lon})")
            
            # 3단계: 근처 대피소 검색
            def haversine(lat1, lon1, lat2, lon2):
                from math import radians, sin, cos, sqrt, atan2
                R = 6371
                dlat = radians(lat2 - lat1)
                dlon = radians(lon2 - lon1)
                a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
                c = 2 * atan2(sqrt(a), sqrt(1-a))
                return R * c
            
            all_data = vectorstore.get(where={"type": "shelter"})
            shelters = []
            
            for metadata in all_data['metadatas']:
                try:
                    lat = float(metadata.get('lat', 0))
                    lon = float(metadata.get('lon', 0))
                    if lat == 0 or lon == 0:
                        continue
                    
                    distance = haversine(user_lat, user_lon, lat, lon)
                    shelters.append({
                        'name': metadata.get('facility_name', 'N/A'),
                        'address': metadata.get('address', 'N/A'),
                        'lat': lat,
                        'lon': lon,
                        'distance': distance,
                        'capacity': int(metadata.get('capacity', 0)),
                        'shelter_type': metadata.get('shelter_type', 'N/A')
                    })
                except Exception:
                    continue
            
            shelters.sort(key=lambda x: x['distance'])
            top_3 = shelters[:3]  # 가장 가까운 3곳만
            
            # 4단계: 재난 행동요령 검색
            guideline_text = ""
            if guideline_hybrid:
                try:
                    guideline_results = guideline_hybrid.invoke(detected_disaster)
                    if guideline_results:
                        # 상위 2개 결과만 사용 (간결하게)
                        guideline_text = "\n\n".join([
                            doc.page_content for doc in guideline_results[:2]
                        ])
                except Exception as e:
                    print(f"[search_location_with_disaster] 가이드라인 검색 실패: {e}")
                    guideline_text = f"{detected_disaster} 관련 행동요령을 찾을 수 없습니다."
            
            # 5단계: 통합 결과 생성
            result = f"""🚨 **{place_name} 근처 {detected_disaster} 발생 시 대응 가이드**

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    📍 **가장 가까운 대피소 {len(top_3)}곳**

    """
            
            for i, s in enumerate(top_3, 1):
                result += f"{i}. **{s['name']}** ({s['distance']:.2f}km)\n"
                result += f"   📍 {s['address']}\n"
                result += f"   📍 위치: {s['shelter_type']} | 수용: {s['capacity']:,}명\n\n"
            
            result += f"""━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    🚨 **{detected_disaster} 행동요령**

    {guideline_text}

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    💡 **즉시 행동 체크리스트**
    ✅ 가장 가까운 대피소로 이동
    ✅ 위 행동요령을 숙지하고 침착하게 대응
    ✅ 119 신고 (필요 시)
    """
            
            # 구조화된 데이터 (지도 표시용)
            structured_data = {
                "location": place_name,
                "coordinates": (user_lat, user_lon),
                "shelters": top_3,
                "total_count": len(all_data['metadatas'])
            }
            
            return {
                "text": result.strip(),
                "structured_data": structured_data
            }
            
        except Exception as e:
            print(f"[ERROR] search_location_with_disaster: {e}")
            import traceback
            traceback.print_exc()
            return {
                "text": f"복합 검색 중 오류 발생: {str(e)}",
                "structured_data": None
            }
    # 6. Tools 리스트
    tools = [
        search_shelter_by_location,
        count_shelters,
        search_shelter_by_capacity,
        search_disaster_guideline,
        answer_general_knowledge,
        search_shelter_by_name,
        search_location_with_disaster
    ]
    
    # 7. LLM에 Tools 바인딩
    llm_with_tools = llm.bind_tools(tools)
    
    # 8. State 정의
    class AgentState(TypedDict):
        messages: Annotated[list[BaseMessage], add_messages]
        intent: str
        rewritten_query: str
        structured_data: Optional[dict]  # 지도 표시용 구조화된 데이터
    
    # 9. 시스템 프롬프트
    SYSTEM_PROMPT = """당신은 대한민국의 재난 안전 전문 AI 도우미입니다.

**핵심 원칙**:
1. **정확성 우선**: 제공된 도구 결과만 사용하고, 없는 정보는 지어내지 마세요
2. **의도 파악**: 사용자 질문의 의도를 정확히 분류하고 적절한 도구를 선택하세요
3. **복합 질문 처리**: 여러 의도가 섞인 질문은 순차적으로 처리하세요

**도구 선택 가이드**:
- 위치 + 재난 복합 질문 → search_location_with_disaster
   - "설악산 근처인데 산사태 발생 시" → search_location_with_disaster("설악산 산사태")
   - "강남역에서 지진 나면" → search_location_with_disaster("강남역 지진")
   - "명동 화재 났을 때" → search_location_with_disaster("명동 화재")
   
- 특정 시설명이 포함된 질문 → search_shelter_by_name
   - "동대문맨션 수용인원" → search_shelter_by_name("동대문맨션")
   - "서울역 대피소 정보" → search_shelter_by_name("서울역")
   
- "근처", "주변" 키워드만 → search_shelter_by_location
   - "강남역 근처 대피소" → search_shelter_by_location("강남역")
   - "명동 주변 피난소" → search_shelter_by_location("명동")

- "X명 이상/이하" 조건 → search_shelter_by_capacity
   - "1000명 이상 수용 가능한 대피소" → search_shelter_by_capacity("1000명 이상")

- "개수", "몇 개" → count_shelters
   - "서울 지하 대피소 몇 개?" → count_shelters("서울 지하")

- 재난 행동요령만 → search_disaster_guideline
   - "지진 발생 시 행동요령" → search_disaster_guideline("지진")
   - (위치 정보 없이 행동요령만 필요한 경우)

- 재난 일반 지식 → answer_general_knowledge
   - "지진이 뭐야?" → answer_general_knowledge("지진이 뭐야")

**중요 판단 기준**:
- 질문에 "위치 + 재난"이 함께 있으면 → search_location_with_disaster
- 질문에 "시설명 + 정보 요청"이 있으면 → search_shelter_by_name
- 질문에 "위치 + 근처/주변"만 있으면 → search_shelter_by_location

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
    
    def tools_node_with_structured_data(state: AgentState):
        """도구 실행 노드 (구조화된 데이터 추출 포함)"""
        from langgraph.prebuilt import ToolNode
        
        # 기본 ToolNode 실행
        tool_node = ToolNode(tools)
        result = tool_node.invoke(state)
        
        # 도구 결과에서 structured_data 추출
        messages = result.get("messages", [])
        structured_data = None
        
        for message in messages:
            if hasattr(message, "content"):
                content = message.content
                
                # content가 문자열인 경우 JSON 파싱 시도
                if isinstance(content, str):
                    try:
                        import json
                        parsed = json.loads(content)
                        if isinstance(parsed, dict) and "structured_data" in parsed:
                            structured_data = parsed["structured_data"]
                            message.content = parsed.get("text", content)
                            print(f"[tools_node] structured_data 추출 완료 (JSON): {structured_data is not None}")
                    except (json.JSONDecodeError, TypeError):
                        pass
                
                # content가 dict인 경우 직접 처리
                elif isinstance(content, dict) and "structured_data" in content:
                    structured_data = content["structured_data"]
                    message.content = content.get("text", str(content))
                    print(f"[tools_node] structured_data 추출 완료 (dict): {structured_data is not None}")
        
        return {
            "messages": messages,
            "structured_data": structured_data
        }
    
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
    workflow.add_node("tools", tools_node_with_structured_data)
    
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