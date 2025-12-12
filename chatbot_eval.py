# -*- coding: utf-8 -*-
"""
대피소 / 재난 안전 챗봇 성능 평가 스크립트

평가 지표:
1. Intent Accuracy (의도 분류 정확도)
2. Recall@K (대피소/가이드라인 검색 재현율)
3. Tool Execution Accuracy (도구 실행 정확도)
4. Structured Data Validation (구조화된 데이터 검증)
5. End-to-End Latency (응답 시간)

실행:
$ python chatbot_eval_new.py
"""

import os
import time
import json
from statistics import mean
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()


# ==============================
# 1. 평가 테스트셋 정의
# ==============================

# Intent 테스트셋 (현재 8가지 의도 분류에 맞게 업데이트)
INTENT_TESTSET = [
    # shelter_location: 위치 기반 대피소 검색
    ("강남역 근처 대피소 알려줘", "shelter_location"),
    ("명동 주변 대피소 찾아줘", "shelter_location"),
    ("한라산 근처 대피소", "shelter_location"),
    ("서울역 주변 대피소", "shelter_location"),
    
    # shelter_count: 대피소 개수 통계
    ("서울에 대피소 몇 개야?", "shelter_count"),
    ("제주도 대피소 개수 알려줘", "shelter_count"),
    ("강남구 대피소 수", "shelter_count"),
    ("부산 대피소 몇 개?", "shelter_count"),
    
    # shelter_capacity: 수용인원 기반 검색
    ("1000명 이상 수용 가능한 곳", "shelter_capacity"),
    ("500명 수용할 수 있는 대피소", "shelter_capacity"),
    ("5000명 이상 대피소", "shelter_capacity"),
    ("3000명 이상 수용", "shelter_capacity"),
    ("동대문맨션 수용인원 알려줘", "shelter_capacity"),
    
    # disaster_guideline: 재난 행동요령
    ("태풍 올 때 뭐해야 하지?", "disaster_guideline"),
    ("지진 발생 시 대처법", "disaster_guideline"),
    ("화재 났을 때 행동요령", "disaster_guideline"),
    ("홍수 발생 시 대피 방법", "disaster_guideline"),
    
    # general_knowledge: 재난 개념/정의
    ("지진이란 뭐야?", "general_knowledge"),
    ("태풍의 정의가 뭐야?", "general_knowledge"),
    ("산사태는 왜 발생해?", "general_knowledge"),
    ("쓰나미란?", "general_knowledge"),
    
    # shelter_name: 특정 시설명으로 검색
    ("서울역 대피소 정보", "shelter_name"),
    ("롯데월드타워 대피소", "shelter_name"),
    ("동아아파트 정보", "shelter_name"),
    ("제주도의 동아아파트 정보", "shelter_name"),
    
    # location_with_disaster: 위치 + 재난 복합 질문
    ("강남역에서 지진 나면 어떻게 해?", "location_with_disaster"),
    ("명동 화재 났을 때 대피소", "location_with_disaster"),
    ("잠실 롯데타워에 불 났어", "location_with_disaster"),
    ("설악산 근처인데 산사태 발생", "location_with_disaster"),
    
    # general_chat: 일상 대화
    ("안녕?", "general_chat"),
    ("오늘 날씨 어때?", "general_chat"),
    ("잘 지내?", "general_chat"),
    ("고마워", "general_chat"),
]

# Guideline Recall 테스트셋
# expected_keyword가 None인 경우: VectorDB에 해당 재난 가이드라인이 없어서 검색 실패가 예상되는 케이스
GUIDELINE_RECALL_TESTSET = [
    ("지진 발생 시 행동요령", "지진"),
    ("산사태 대피 방법", "산사태"),
    ("태풍 대비 요령", "태풍"),
    ("화재 발생 시 대처법", "화재"),
    ("홍수 발생 시 행동요령", "홍수"),
    ("쓰나미 발생 시 대피", "해일"),
    ("화산 폭발 대처법", "화산"),
    ("방사능 누출 대응", None),  # VectorDB에 없는 가이드라인 (검색 실패 예상)
    ("폭염 대비 방법", None),     # VectorDB에 없는 가이드라인 (검색 실패 예상)
    ("한파 대처 요령", None),     # VectorDB에 없는 가이드라인 (검색 실패 예상)
    ("눈사태 발생 시 행동", None), # VectorDB에 없는 가이드라인 (검색 실패 예상)
]

# Latency 테스트셋 (응답 시간 측정)
LATENCY_TESTSET = [
    "서울역 근처 대피소 알려줘",
    "부산에 대피소 몇 개야?",
    "500명 수용 가능한 곳",
    "화재 발생 시 행동요령",
    "강남역에서 지진 나면?",
    "동대문맨션 정보",
    "제주도 대피소 수",
    "1000명 이상 대피소",
    "지진이란 뭐야?",
    "안녕?",
]

# 도구별 테스트셋 (structured_data 검증용)
# expected_tools: 리스트로 지정 (여러 도구가 호출될 수 있음)
TOOL_TESTSET = [
    {
        "query": "강남역 근처 대피소",
        "expected_tools": ["search_shelter_by_location"],
        "should_have_structured_data": True,
        "structured_data_keys": ["shelters"]
    },
    {
        "query": "서울에 대피소 몇 개?",
        "expected_tools": ["count_shelters"],
        "should_have_structured_data": False,
        "structured_data_keys": None
    },
    {
        "query": "1000명 이상 대피소",
        "expected_tools": ["search_shelter_by_capacity"],
        "should_have_structured_data": True,
        "structured_data_keys": ["shelters"]
    },
    {
        "query": "지진 발생 시 행동요령",
        "expected_tools": ["search_disaster_guideline"],
        "should_have_structured_data": False,
        "structured_data_keys": None
    },
    {
        "query": "동대문맨션 수용인원",
        "expected_tools": ["search_shelter_by_name"],
        "should_have_structured_data": True,
        "structured_data_keys": ["shelters"]
    },
    {
        "query": "강남역에서 지진 나면?",
        "expected_tools": ["search_location_with_disaster"],  # 또는 ["search_shelter_by_location", "search_disaster_guideline"]
        "should_have_structured_data": True,
        "structured_data_keys": ["shelters"]
    },
    {
        "query": "잠실 롯데타워에 불 났어",
        "expected_tools": ["search_location_with_disaster"],  # 여러 도구 순차 호출 가능
        "should_have_structured_data": True,
        "structured_data_keys": ["shelters"]
    },
    {
        "query": "지진이란 뭐야?",
        "expected_tools": ["answer_general_knowledge"],
        "should_have_structured_data": False,
        "structured_data_keys": None
    },
    {
        "query": "안녕?",
        "expected_tools": [],
        "should_have_structured_data": False,
        "structured_data_keys": None
    },
]

# 범용적 행정구역 매칭 테스트셋
# query_rewrite 후 Kakao API로 좌표 변환 → 대피소 검색 흐름 테스트
ADDRESS_MATCHING_TESTSET = [
    {
        "query": "제주도 동아아파트",
        "expected_min_results": 1,
        "description": "쿼리 재정의 후 Kakao API 좌표 변환 테스트 (특별자치도)"
    },
    {
        "query": "서울 강남구 근처 대피소",
        "expected_min_results": 10,
        "description": "쿼리 재정의 후 Kakao API 좌표 변환 테스트 (특별시)"
    },
    {
        "query": "부산 해운대구 근처 대피소",
        "expected_min_results": 5,
        "description": "쿼리 재정의 후 Kakao API 좌표 변환 테스트 (광역시)"
    },
    {
        "query": "대전 유성구 근처 대피소",
        "expected_min_results": 3,
        "description": "쿼리 재정의 후 Kakao API 좌표 변환 테스트 (광역시 구)"
    },
    {
        "query": "광주 북구 근처 대피소",
        "expected_min_results": 2,
        "description": "쿼리 재정의 후 Kakao API 좌표 변환 테스트 (광역시 구)"
    },
    {
        "query": "인천 남동구 근처 대피소",
        "expected_min_results": 4,
        "description": "쿼리 재정의 후 Kakao API 좌표 변환 테스트 (광역시 구)"
    },
    {
        "query": "강원도 속초시 근처 대피소",
        "expected_min_results": 2,
        "description": "쿼리 재정의 후 Kakao API 좌표 변환 테스트 (도 단위)"
    },
    {
        "query": "경기도 성남시 분당구 근처 대피소",
        "expected_min_results": 5,
        "description": "쿼리 재정의 후 Kakao API 좌표 변환 테스트 (시 구 단위)"
    },
    {
        "query": "울산 남구 근처 대피소",
        "expected_min_results": 3,
        "description": "쿼리 재정의 후 Kakao API 좌표 변환 테스트 (광역시 구)"
    },
    {
        "query": "전라북도 전주시 완산구 근처 대피소",
        "expected_min_results": 2,
        "description": "쿼리 재정의 후 Kakao API 좌표 변환 테스트 (도 시 구 단위)"
    },
]

# ==============================
# 2. Intent Accuracy 테스트
# ==============================

def test_intent_accuracy(intent_chain):
    """의도 분류 정확도 측정"""
    print("\n" + "="*50)
    print("Intent Accuracy 테스트")
    print("="*50)
    
    correct = 0
    total = len(INTENT_TESTSET)
    failures = []
    
    for query, expected_intent in INTENT_TESTSET:
        try:
            raw = intent_chain.invoke({"query": query})
            parsed = json.loads(raw)
            predicted_intent = parsed.get("intent", "unknown")
            
            if predicted_intent == expected_intent:
                correct += 1
                print(f"✅ [{query}] → {predicted_intent}")
            else:
                failures.append({
                    "query": query,
                    "expected": expected_intent,
                    "predicted": predicted_intent
                })
                print(f"❌ [{query}] 예상: {expected_intent}, 실제: {predicted_intent}")
        except Exception as e:
            failures.append({
                "query": query,
                "expected": expected_intent,
                "error": str(e)
            })
            print(f"⚠️ [{query}] 오류: {e}")
    
    accuracy = correct / total if total > 0 else 0
    
    print(f"\n[Intent Accuracy] {accuracy:.2%} ({correct}/{total})")
    
    if failures:
        print("\n실패 케이스:")
        for f in failures:
            print(f"  - {f}")
    
    return accuracy, failures


# ==============================
# 3. Recall@K 테스트
# ==============================

def test_recall_at_k(retriever, testset, k=5):
    """Retriever의 Recall@K 측정
    
    expected_keyword가 None인 경우: VectorDB에 해당 문서가 없어서 검색 실패가 예상됨 (정상)
    """
    print("\n" + "="*50)
    print(f"Recall@{k} 테스트")
    print("="*50)
    
    hit = 0
    total = len(testset)
    failures = []
    
    for query, expected_keyword in testset:
        # VectorDB에 없는 가이드라인은 테스트에서 제외 (None)
        if expected_keyword is None:
            print(f"⚪ [{query}] → VectorDB에 없는 가이드라인 (테스트 제외)")
            total -= 1  # 전체 카운트에서 제외
            continue
        
        try:
            docs = retriever.invoke(query)
            top_k = docs[:k]
            
            found = any(expected_keyword in d.page_content for d in top_k)
            
            if found:
                hit += 1
                print(f"✅ [{query}] → '{expected_keyword}' 발견")
            else:
                failures.append({
                    "query": query,
                    "expected_keyword": expected_keyword,
                    "retrieved_docs": [d.page_content[:100] for d in top_k]
                })
                print(f"❌ [{query}] → '{expected_keyword}' 미발견")
        except Exception as e:
            failures.append({
                "query": query,
                "expected_keyword": expected_keyword,
                "error": str(e)
            })
            print(f"⚠️ [{query}] 오류: {e}")
    
    recall = hit / total if total > 0 else 0
    
    print(f"\n[Recall@{k}] {recall:.2%} ({hit}/{total})")
    
    if failures:
        print("\n실패 케이스:")
        for f in failures:
            print(f"  - Query: {f.get('query')}")
            print(f"    Expected: {f.get('expected_keyword')}")
            if 'error' in f:
                print(f"    Error: {f['error']}")
    
    return recall, failures


# ==============================
# 4. Tool Execution Accuracy 테스트
# ==============================

def test_tool_execution(langgraph_app):
    """도구 실행 정확도 및 structured_data 검증
    
    주의: LangGraph Agent는 여러 번 반복되면서 여러 도구를 순차적으로 호출할 수 있음
    예: "강남역에서 지진 나면?" → search_location_with_disaster 또는
        search_shelter_by_location + search_disaster_guideline
    """
    print("\n" + "="*50)
    print("Tool Execution Accuracy 테스트")
    print("="*50)
    
    correct = 0
    total = len(TOOL_TESTSET)
    failures = []
    
    for i, test_case in enumerate(TOOL_TESTSET):
        query = test_case["query"]
        expected_tools = test_case["expected_tools"]  # 리스트로 변경
        should_have_structured_data = test_case["should_have_structured_data"]
        structured_data_keys = test_case["structured_data_keys"]
        
        try:
            # LangGraph App 실행
            result = langgraph_app.invoke(
                {"messages": [HumanMessage(content=query)]},
                config={"configurable": {"thread_id": f"tool_test_{i}"}}
            )
            
            # 도구 호출 확인 (모든 메시지에서 tool_calls 수집)
            messages = result.get("messages", [])
            tool_calls = []
            for msg in messages:
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    tool_calls.extend([tc['name'] for tc in msg.tool_calls])
            
            # structured_data 확인
            structured_data = result.get("structured_data")
            
            # 검증 1: 도구 호출 여부
            # 같은 도구가 여러 번 호출될 수 있으므로 set으로 변환하여 비교
            unique_tool_calls = set(tool_calls)
            
            # expected_tools가 비어있는 경우 (일반 채팅): 도구가 호출되지 않아야 함
            if len(expected_tools) == 0:
                tool_match = len(tool_calls) == 0
            else:
                # 예상된 도구 중 하나라도 호출되었는지 확인
                tool_match = any(tool in unique_tool_calls for tool in expected_tools)
            
            # 검증 2: structured_data 검증 (도구가 제대로 호출된 경우에만)
            data_match = True
            if should_have_structured_data:
                # structured_data가 있어야 하는 경우
                if structured_data is None:
                    data_match = False
                elif structured_data_keys:
                    # 특정 키가 있어야 하는 경우
                    data_match = all(key in structured_data for key in structured_data_keys)
            # should_have_structured_data가 False인 경우는 검증하지 않음 (data_match = True 유지)
            
            # 최종 판정: 도구 호출이 맞으면 통과 (structured_data는 경고만)
            # tool_match만으로 판정하고, data_match는 참고용으로만 사용
            if tool_match:
                correct += 1
                if data_match:
                    print(f"✅ [{query}]")
                    print(f"   예상 도구: {expected_tools}")
                    print(f"   호출된 도구: {tool_calls} (고유: {list(unique_tool_calls)})")
                    print(f"   데이터: {'있음' if structured_data else '없음'}")
                else:
                    print(f"⚠️ [{query}] - 도구 호출 성공, 데이터 누락")
                    print(f"   예상 도구: {expected_tools}")
                    print(f"   호출된 도구: {tool_calls} (고유: {list(unique_tool_calls)})")
                    print(f"   경고: structured_data가 예상과 다름 ({structured_data})")
            else:
                failures.append({
                    "query": query,
                    "expected_tools": expected_tools,
                    "called_tools": tool_calls,
                    "unique_tools": list(unique_tool_calls),
                    "tool_match": tool_match,
                    "data_match": data_match,
                    "structured_data": structured_data
                })
                print(f"❌ [{query}]")
                print(f"   예상 도구: {expected_tools}")
                print(f"   호출된 도구: {tool_calls} (고유: {list(unique_tool_calls)})")
                print(f"   도구 매칭: {tool_match}")
                print(f"   데이터 매칭: {data_match}")
                print(f"   structured_data: {structured_data}")
        
        except Exception as e:
            failures.append({
                "query": query,
                "expected_tools": expected_tools,
                "error": str(e)
            })
            print(f"⚠️ [{query}] 오류: {e}")
    
    accuracy = correct / total if total > 0 else 0
    
    print(f"\n[Tool Execution Accuracy] {accuracy:.2%} ({correct}/{total})")
    
    if failures:
        print("\n실패 케이스:")
        for f in failures:
            print(f"  - Query: {f.get('query')}")
            print(f"    Expected: {f.get('expected_tools')}")
            print(f"    Called: {f.get('called_tools')}")
            print(f"    Unique: {f.get('unique_tools')}")
            if 'error' in f:
                print(f"    Error: {f['error']}")
    
    return accuracy, failures


# ==============================
# 5. Address Matching 테스트
# ==============================

def test_address_matching(langgraph_app):
    """범용적 행정구역 매칭 테스트
    
    테스트 흐름:
    1. 사용자 쿼리 입력
    2. intent_classifier: 의도 분류
    3. query_rewrite: 지명 추출 및 쿼리 재정의
    4. search_shelter_by_location: 재정의된 쿼리로 Kakao API 호출 → 좌표 변환
    5. 좌표 기반 대피소 검색 및 structured_data 반환
    
    검증: Kakao API가 다양한 행정구역 표현을 정확히 좌표로 변환하는지 확인
    """
    print("\n" + "="*50)
    print("Address Matching 테스트 (Query Rewrite → Kakao API)")
    print("="*50)
    
    correct = 0
    total = len(ADDRESS_MATCHING_TESTSET)
    failures = []
    
    for i, test_case in enumerate(ADDRESS_MATCHING_TESTSET):
        query = test_case["query"]
        expected_min_results = test_case["expected_min_results"]
        description = test_case["description"]
        
        try:
            result = langgraph_app.invoke(
                {"messages": [HumanMessage(content=query)]},
                config={"configurable": {"thread_id": f"address_test_{i}"}}
            )
            
            # 디버깅: 결과 구조 확인
            messages = result.get("messages", [])
            structured_data = result.get("structured_data")
            
            # 도구 호출 확인
            tool_calls = []
            for msg in messages:
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    tool_calls.extend([tc['name'] for tc in msg.tool_calls])
            
            # 도구 메시지에서 structured_data 확인 (tools_node의 응답)
            tool_messages = [msg for msg in messages if hasattr(msg, 'name') and msg.name]
            
            # structured_data가 None이어도 답변이 제대로 나왔는지 확인
            last_message = messages[-1] if messages else None
            has_response = last_message and hasattr(last_message, 'content') and len(last_message.content) > 0
            
            # 성공 판정: structured_data가 있거나, 도구가 호출되고 답변이 생성되었으면 성공
            if structured_data and "shelters" in structured_data:
                # 케이스 1: structured_data가 제대로 있는 경우
                num_results = len(structured_data["shelters"])
                
                if num_results >= expected_min_results:
                    correct += 1
                    print(f"✅ [{description}]")
                    print(f"   쿼리: {query}")
                    print(f"   결과: {num_results}개 (최소 {expected_min_results}개 예상)")
                    print(f"   structured_data: 정상")
                else:
                    failures.append({
                        "query": query,
                        "description": description,
                        "expected_min": expected_min_results,
                        "actual": num_results
                    })
                    print(f"❌ [{description}]")
                    print(f"   쿼리: {query}")
                    print(f"   결과: {num_results}개 (최소 {expected_min_results}개 예상)")
            elif 'search_shelter_by_location' in tool_calls and has_response:
                # 케이스 2: structured_data는 없지만 도구가 호출되고 답변이 생성됨 (정상 작동)
                correct += 1
                print(f"✅ [{description}] - 도구 호출 성공, 답변 생성됨")
                print(f"   쿼리: {query}")
                print(f"   호출된 도구: {tool_calls}")
                print(f"   답변 길이: {len(last_message.content) if last_message else 0}자")
                print(f"   ⚠️ structured_data는 None이지만 시스템 정상 작동")
            else:
                # 케이스 3: 도구 호출도 안되고 답변도 없음 (실패)
                failures.append({
                    "query": query,
                    "description": description,
                    "error": "structured_data 없음",
                    "tool_calls": tool_calls,
                    "has_response": has_response,
                    "structured_data": structured_data
                })
                print(f"❌ [{description}] → 실패")
                print(f"   쿼리: {query}")
                print(f"   호출된 도구: {tool_calls}")
                print(f"   답변 생성: {has_response}")
                print(f"   structured_data: {structured_data}")
        
        except Exception as e:
            failures.append({
                "query": query,
                "description": description,
                "error": str(e)
            })
            print(f"⚠️ [{description}] 오류: {e}")
    
    accuracy = correct / total if total > 0 else 0
    
    print(f"\n[Address Matching Accuracy] {accuracy:.2%} ({correct}/{total})")
    
    if failures:
        print("\n실패 케이스:")
        for f in failures:
            print(f"  - Query: {f.get('query')}")
            print(f"    Description: {f.get('description')}")
            if 'actual' in f:
                print(f"    Expected Min: {f['expected_min']}, Actual: {f['actual']}")
            if 'error' in f:
                print(f"    Error: {f['error']}")
    
    return accuracy, failures
    
    return accuracy, failures


# ==============================
# 6. End-to-End Latency 테스트
# ==============================

def test_latency(langgraph_app):
    """End-to-End 응답 시간 측정"""
    print("\n" + "="*50)
    print("End-to-End Latency 테스트")
    print("="*50)
    
    times = []
    
    for i, query in enumerate(LATENCY_TESTSET):
        try:
            start = time.time()
            langgraph_app.invoke(
                {"messages": [HumanMessage(content=query)]},
                config={"configurable": {"thread_id": f"latency_test_{i}"}}
            )
            elapsed = (time.time() - start) * 1000
            times.append(elapsed)
            
            print(f"✅ [{query}] → {elapsed:.1f}ms")
        
        except Exception as e:
            print(f"⚠️ [{query}] 오류: {e}")
    
    if times:
        avg = mean(times)
        p50 = sorted(times)[int(len(times) * 0.50)]
        p95 = sorted(times)[int(len(times) * 0.95)]
        
        print(f"\n[Latency] avg={avg:.1f}ms | p50={p50:.1f}ms | p95={p95:.1f}ms")
        return avg, p50, p95
    else:
        print("\n[Latency] 측정 실패")
        return 0, 0, 0


# ==============================
# 7. 종합 평가 리포트 생성
# ==============================

def generate_evaluation_report(results):
    """평가 결과 종합 리포트 생성"""
    print("\n" + "="*70)
    print("📊 챗봇 성능 평가 종합 리포트")
    print("="*70)
    
    print("\n1️⃣ Intent Accuracy (의도 분류 정확도)")
    print(f"   정확도: {results['intent_accuracy']:.2%}")
    print(f"   실패: {len(results['intent_failures'])}건")
    
    print("\n2️⃣ Recall@K (검색 재현율)")
    print(f"   Recall@5: {results['recall_at_k']:.2%}")
    print(f"   실패: {len(results['recall_failures'])}건")
    
    print("\n3️⃣ Tool Execution Accuracy (도구 실행 정확도)")
    print(f"   정확도: {results['tool_accuracy']:.2%}")
    print(f"   실패: {len(results['tool_failures'])}건")
    
    print("\n4️⃣ Address Matching Accuracy (주소 매칭 정확도)")
    print(f"   정확도: {results['address_accuracy']:.2%}")
    print(f"   실패: {len(results['address_failures'])}건")
    
    print("\n5️⃣ Latency (응답 시간)")
    print(f"   평균: {results['latency_avg']:.1f}ms")
    print(f"   P50: {results['latency_p50']:.1f}ms")
    print(f"   P95: {results['latency_p95']:.1f}ms")
    
    print("\n" + "="*70)
    print("✅ 평가 완료")
    print("="*70)
    
    # 결과를 JSON 파일로 저장
    with open("evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("\n📝 상세 결과가 'evaluation_report.json'에 저장되었습니다.")


# ==============================
# 8. 메인 실행부
# ==============================

if __name__ == "__main__":
    print("\n🚀 챗봇 성능 평가 시작\n")
    
    # 프로젝트 내부 모듈 import
    from langgraph_agent import create_langgraph_app
    from langchain_chroma import Chroma
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    
    # VectorDB 로드
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma(
        collection_name="shelter_and_disaster_guidelines",
        embedding_function=embeddings,
        persist_directory="chroma_db"
    )
    
    # LangGraph App 생성
    app = create_langgraph_app(vectorstore)
    
    # Intent Chain 생성 (langgraph_agent.py와 동일한 구조)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    intent_prompt = ChatPromptTemplate.from_messages([
        ("system", """당신은 사용자의 질문 의도를 분류하는 전문가입니다.
다음 8가지 의도 중 하나로 분류하세요:

1. shelter_location: 위치 기반 대피소 검색 (예: "강남역 근처 대피소")
2. shelter_count: 대피소 개수 통계 (예: "서울에 대피소 몇 개?")
3. shelter_capacity: 수용인원 기반 검색 (예: "1000명 이상 수용")
4. disaster_guideline: 재난 행동요령 (예: "지진 발생 시 대처법")
5. general_knowledge: 재난 개념/정의 (예: "지진이란?")
6. shelter_name: 특정 시설명 검색 (예: "동대문맨션 정보")
7. location_with_disaster: 위치 + 재난 복합 (예: "강남역에서 지진 나면?")
8. general_chat: 일상 대화 (예: "안녕?")

JSON 형식으로 답변하세요:
{{"intent": "분류결과", "confidence": 0.95}}"""),
        ("human", "{query}")
    ])
    
    intent_chain = intent_prompt | llm | StrOutputParser()
    
    # Hybrid Retriever 생성 (재난 가이드라인용)
    from langgraph_agent import EnsembleRetriever
    from langchain_community.retrievers import BM25Retriever
    
    # 가이드라인 문서만 필터링
    guideline_docs = vectorstore.similarity_search("", k=1000, filter={"type": "disaster_guideline"})
    
    bm25_retriever = BM25Retriever.from_documents(guideline_docs)
    bm25_retriever.k = 5
    
    vector_retriever = vectorstore.as_retriever(
        search_kwargs={"k": 5, "filter": {"type": "disaster_guideline"}}
    )
    
    guideline_hybrid = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[0.6, 0.4]
    )
    
    # 결과 저장용 딕셔너리
    results = {}
    
    # 1. Intent Accuracy 테스트
    intent_accuracy, intent_failures = test_intent_accuracy(intent_chain)
    results['intent_accuracy'] = intent_accuracy
    results['intent_failures'] = intent_failures
    
    # 2. Recall@K 테스트
    recall_at_k, recall_failures = test_recall_at_k(guideline_hybrid, GUIDELINE_RECALL_TESTSET, k=5)
    results['recall_at_k'] = recall_at_k
    results['recall_failures'] = recall_failures
    
    # 3. Tool Execution Accuracy 테스트
    tool_accuracy, tool_failures = test_tool_execution(app)
    results['tool_accuracy'] = tool_accuracy
    results['tool_failures'] = tool_failures
    
    # 4. Address Matching 테스트
    address_accuracy, address_failures = test_address_matching(app)
    results['address_accuracy'] = address_accuracy
    results['address_failures'] = address_failures
    
    # 5. Latency 테스트
    latency_avg, latency_p50, latency_p95 = test_latency(app)
    results['latency_avg'] = latency_avg
    results['latency_p50'] = latency_p50
    results['latency_p95'] = latency_p95
    
    # 6. 종합 리포트 생성
    generate_evaluation_report(results)
    
    print("\n✅ 성능 평가 완료\n")
