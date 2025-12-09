import os
from typing import Annotated, List, Dict, Any, Optional, Literal
from dotenv import load_dotenv

# LangChain & LangGraph Imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict
from langchain_core.documents import Document

# 1. 환경 설정 및 DB 로드
load_dotenv()

# 임베딩 모델 설정
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Chroma DB 로드
try:
    vectorstore = Chroma(
        collection_name="shelter_and_disaster_guidelines",
        embedding_function=embeddings,
        persist_directory="./chroma_db"
    )
except Exception as e:
    print(f"❌ Chroma DB 로드 실패: {e}")
    raise

# 2. 하이브리드 리트리버 구성

# 2-1. Semantic (Vector) Retriever
shelter_vector_retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 5,
        "filter": {"type": "shelter"}
    }
)

guideline_vector_retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 3,
        "filter": {"type": "disaster_guideline"}
    }
)

# 2-2. BM25 (Keyword) Retriever 구성
def create_bm25_retriever(doc_type: str) -> BM25Retriever:
    """BM25 키워드 기반 리트리버 생성"""
    try:
        all_docs = vectorstore.get(where={"type": doc_type})
        
        if not all_docs or 'documents' not in all_docs:
            print(f"⚠️ {doc_type} 문서가 없습니다.")
            return None
        
        documents = []
        for i, text in enumerate(all_docs['documents']):
            metadata = all_docs['metadatas'][i] if 'metadatas' in all_docs else {}
            documents.append(Document(page_content=text, metadata=metadata))
        
        bm25_retriever = BM25Retriever.from_documents(documents)
        bm25_retriever.k = 5
        
        return bm25_retriever
    except Exception as e:
        print(f"⚠️ BM25 Retriever 생성 실패: {e}")
        return None

shelter_bm25_retriever = create_bm25_retriever("shelter")
guideline_bm25_retriever = create_bm25_retriever("disaster_guideline")

# 2-3. Ensemble (Hybrid) Retriever
shelter_hybrid_retriever = None
if shelter_bm25_retriever:
    shelter_hybrid_retriever = EnsembleRetriever(
        retrievers=[shelter_vector_retriever, shelter_bm25_retriever],
        weights=[0.6, 0.4]
    )
else:
    shelter_hybrid_retriever = shelter_vector_retriever

guideline_hybrid_retriever = None
if guideline_bm25_retriever:
    guideline_hybrid_retriever = EnsembleRetriever(
        retrievers=[guideline_vector_retriever, guideline_bm25_retriever],
        weights=[0.7, 0.3]
    )
else:
    guideline_hybrid_retriever = guideline_vector_retriever

# 3. LCEL 방식의 RAG Chain

# 3-1. 프롬프트 템플릿
RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """당신은 재난 안전 도우미입니다.
아래 참고 문서를 바탕으로 사용자의 질문에 **정확하게** 답변하세요.

답변 시 주의사항:
- 참고 문서에 없는 내용은 절대 지어내지 마세요
- 간결하고 명확하게 답변하세요
- 중요한 정보는 **볼드체**로 강조하세요
"""),
    ("human", """참고 문서:
{context}

질문: {question}""")
])

# 3-2. Document 포맷팅 함수
def format_docs(docs: List[Document]) -> str:
    """검색된 문서를 포맷팅"""
    if not docs:
        return "관련 문서를 찾을 수 없습니다."
    
    formatted = []
    for i, doc in enumerate(docs, 1):
        formatted.append(f"[문서 {i}]\n{doc.page_content}")
    
    return "\n\n".join(formatted)

# 3-3. LCEL RAG Chain 생성
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 대피소 검색 체인
shelter_rag_chain = (
    {
        "context": shelter_hybrid_retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough()
    }
    | RAG_PROMPT
    | llm
    | StrOutputParser()
)

# 행동요령 검색 체인
guideline_rag_chain = (
    {
        "context": guideline_hybrid_retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough()
    }
    | RAG_PROMPT
    | llm
    | StrOutputParser()
)

# 4. LCEL 방식의 질문 분류기

CLASSIFICATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """다음 질문을 분류하세요.

분류 기준:
1. simple_shelter: 특정 지역의 대피소 검색
2. simple_guideline: 재난 행동요령 질문
3. statistics: 통계/집계 질문
4. complex: 복잡한 질문

다음 형식으로만 답변하세요:
type: [simple_shelter|simple_guideline|statistics|complex]
confidence: [0.0-1.0]"""),
    ("human", "{query}")
])

def parse_classification(response: str) -> Dict[str, Any]:
    """분류 결과 파싱"""
    lines = response.strip().split('\n')
    result = {"type": "complex", "confidence": 0.5}
    
    for line in lines:
        if line.startswith("type:"):
            result["type"] = line.split(":", 1)[1].strip()
        elif line.startswith("confidence:"):
            try:
                result["confidence"] = float(line.split(":", 1)[1].strip())
            except:
                pass
    
    return result

# 분류 체인
classification_chain = (
    CLASSIFICATION_PROMPT
    | llm
    | StrOutputParser()
    | RunnableLambda(parse_classification)
)

# 4-2. 질문 재정의 체인 추가

QUERY_REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """당신은 검색 쿼리 최적화 전문가입니다.
사용자의 자연어 질문을 검색에 최적화된 형태로 변환하세요.

**변환 규칙**:
1. 구어체 제거: "좀", "알려줘", "있어?" 등 제거
2. 핵심 키워드 추출: 검색에 중요한 단어만 남김
3. 검색 의도 명확화: 대피소/행동요령/통계 등 명시
4. 메타데이터 필터링 정보 추가: 지하/지상, 지역명, 수용인원 등

**예시**:
- "강남구에 있는 대피소 좀 알려줘" → "강남구 대피소"
- "지하에 위치한 대피소는 몇 개야?" → "지하 위치 대피소 개수 통계"
- "지진 났을 때 어떻게 해?" → "지진 발생 시 행동요령"

원본 질문만 변환하고, 추가 설명 없이 변환된 쿼리만 출력하세요.
"""),
    ("human", "{original_query}")
])

# 질문 재정의 체인
query_rewrite_chain = (
    QUERY_REWRITE_PROMPT
    | llm
    | StrOutputParser()
)

# 5. 도구(Tools) 정의 - Query Rewriting 적용

@tool
def search_shelter(query: str) -> str:
    """
    주소, 지역명, 시설명 등을 입력받아 '민방위 대피소' 정보를 검색합니다.
    하이브리드 검색(벡터 + 키워드)을 사용합니다.
    """
    try:
        # 질문 재정의
        rewritten_query = query_rewrite_chain.invoke({"original_query": query})
        print(f"🔄 원본: {query}")
        print(f"🔍 재정의: {rewritten_query}")
        
        # 재정의된 쿼리로 검색
        docs = shelter_hybrid_retriever.invoke(rewritten_query)
        
        if not docs:
            return "검색된 대피소 정보가 없습니다."

        seen = set()
        results = []
        for doc in docs:
            facility_name = doc.metadata.get('facility_name', '알 수 없음')
            if facility_name in seen:
                continue
            seen.add(facility_name)
            
            info = (
                f"📍 시설명: {facility_name}\n"
                f"   - 주소: {doc.metadata.get('address', '주소 정보 없음')}\n"
                f"   - 위치: {doc.metadata.get('shelter_type', '')}\n"
                f"   - 수용인원: {doc.metadata.get('capacity', 0)}명\n"
                f"   - 시설구분: {doc.metadata.get('facility_type', '')}"
            )
            results.append(info)
            
            if len(results) >= 4:
                break
        
        return "\n---\n".join(results)
    
    except Exception as e:
        return f"대피소 검색 중 오류: {str(e)}"

@tool
def search_guideline(query: str) -> str:
    """
    재난 행동 요령을 검색합니다.
    하이브리드 검색(벡터 + 키워드)을 사용합니다.
    """
    try:
        # 질문 재정의
        rewritten_query = query_rewrite_chain.invoke({"original_query": query})
        print(f"🔄 원본: {query}")
        print(f"🔍 재정의: {rewritten_query}")
        
        docs = guideline_hybrid_retriever.invoke(rewritten_query)
        
        if not docs:
            return "관련된 행동 요령을 찾을 수 없습니다."

        results = []
        for doc in docs:
            meta = doc.metadata
            header = f"🚨 [{meta.get('category', '재난')}] {meta.get('situation', '상황')} - {meta.get('title', '')}"
            content = doc.page_content
            results.append(f"{header}\n{content}")
            
        return "\n===\n".join(results)
    
    except Exception as e:
        return f"행동요령 검색 중 오류: {str(e)}"

@tool
def count_shelters_by_capacity(min_capacity: int) -> str:
    """특정 수용인원 이상의 대피소 개수를 집계합니다."""
    try:
        all_shelters = vectorstore.get(where={"type": "shelter"})
        
        if not all_shelters or 'metadatas' not in all_shelters:
            return "대피소 데이터를 가져올 수 없습니다."
        
        count = 0
        filtered_shelters = []
        
        for metadata in all_shelters['metadatas']:
            capacity = metadata.get('capacity', 0)
            try:
                capacity_num = int(capacity)
                if capacity_num >= min_capacity:
                    count += 1
                    filtered_shelters.append({
                        'name': metadata.get('facility_name', '알 수 없음'),
                        'capacity': capacity_num,
                        'address': metadata.get('address', '주소 정보 없음')
                    })
            except (ValueError, TypeError):
                continue
        
        if count == 0:
            return f"수용인원 {min_capacity:,}명 이상의 대피소가 없습니다."
        
        filtered_shelters.sort(key=lambda x: x['capacity'], reverse=True)
        
        result = f"📊 **전국 수용인원 {min_capacity:,}명 이상 대피소: 총 {count}개**\n\n"
        result += "**[상위 5개 대피소]**\n"
        for i, shelter in enumerate(filtered_shelters[:5], 1):
            result += (
                f"{i}. **{shelter['name']}**\n"
                f"   - 수용인원: {shelter['capacity']:,}명\n"
                f"   - 주소: {shelter['address']}\n"
            )
        
        if count > 5:
            result += f"\n*(외 {count - 5}개 더 있음)*"
        
        return result
    
    except Exception as e:
        return f"대피소 통계 조회 중 오류: {str(e)}"

@tool
def get_shelter_statistics() -> str:
    """전국 대피소의 전체 통계 정보를 제공합니다."""
    try:
        all_shelters = vectorstore.get(where={"type": "shelter"})
        
        if not all_shelters or 'metadatas' not in all_shelters:
            return "대피소 통계 데이터를 가져올 수 없습니다."
        
        metadatas = all_shelters['metadatas']
        total_count = len(metadatas)
        
        capacities = []
        regions = {}
        
        for meta in metadatas:
            try:
                cap = int(meta.get('capacity', 0))
                capacities.append(cap)
            except (ValueError, TypeError):
                pass
            
            address = meta.get('address', '')
            if address:
                region = address.split()[0]
                regions[region] = regions.get(region, 0) + 1
        
        if not capacities:
            return "수용인원 데이터가 없습니다."
        
        avg_capacity = sum(capacities) / len(capacities)
        max_capacity = max(capacities)
        min_capacity = min(capacities)
        
        top_regions = sorted(regions.items(), key=lambda x: x[1], reverse=True)[:5]
        
        result = f"""
📊 **전국 대피소 통계**

**기본 정보**
- 총 대피소 수: {total_count:,}개
- 평균 수용인원: {avg_capacity:,.0f}명
- 최대 수용인원: {max_capacity:,}명
- 최소 수용인원: {min_capacity:,}명

**지역별 분포 (상위 5개)**
"""
        for i, (region, count) in enumerate(top_regions, 1):
            result += f"{i}. {region}: {count:,}개\n"
        
        return result.strip()
    
    except Exception as e:
        return f"통계 조회 중 오류: {str(e)}"

tools = [search_shelter, search_guideline, count_shelters_by_capacity, get_shelter_statistics]

# 6. 하이브리드 그래프 상태 정의

class HybridAgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    query_type: Optional[str]
    use_rag: bool
    context: Optional[str]

# 7. 노드 함수들 (LCEL 사용)

llm_with_tools = llm.bind_tools(tools)

def classifier_node(state: HybridAgentState):
    """질문을 분류하는 노드 (LCEL 사용)"""
    last_message = state["messages"][-1]
    query = last_message.content if isinstance(last_message, HumanMessage) else ""
    
    # LCEL 체인 실행
    classification = classification_chain.invoke({"query": query})
    
    return {
        "query_type": classification["type"],
        "use_rag": classification["type"] in ["simple_shelter", "simple_guideline"]
    }

def rag_node(state: HybridAgentState):
    """RAG로 직접 답변하는 노드 (LCEL 사용)"""
    last_message = state["messages"][-1]
    query = last_message.content
    query_type = state.get("query_type", "complex")
    
    try:
        # LCEL 체인 선택 및 실행
        if query_type == "simple_shelter":
            answer = shelter_rag_chain.invoke(query)
        else:  # simple_guideline
            answer = guideline_rag_chain.invoke(query)
        
        return {"messages": [AIMessage(content=answer)]}
    
    except Exception as e:
        return {"messages": [AIMessage(content=f"RAG 처리 중 오류: {str(e)}")]}

# Agent용 시스템 프롬프트 추가
AGENT_SYSTEM_PROMPT = """당신은 대한민국의 재난 안전 도우미 AI입니다.

**중요한 규칙**:
1. 현재 질문에만 집중하세요. 이전 대화와 무관한 새로운 질문이면 완전히 다른 답변을 하세요.
2. 제공된 도구(search_shelter, search_guideline, count_shelters_by_capacity, get_shelter_statistics)를 사용하여 정확한 정보를 찾으세요.
3. 도구 검색 결과에 없는 내용은 절대 지어내지 마세요.
4. 질문이 재난/대피소와 무관하면 "죄송하지만 재난 안전과 관련된 질문에만 답변드릴 수 있습니다."라고 답하세요.

**도구 사용 가이드**:
- 특정 지역 대피소 검색 → search_shelter
- 재난 행동요령 → search_guideline  
- 수용인원 기준 통계 → count_shelters_by_capacity
- 전체 대피소 통계 → get_shelter_statistics
"""

def agent_node(state: HybridAgentState):
    """Agent가 도구를 사용하는 노드"""
    messages = state["messages"]
    
    # 시스템 프롬프트가 없으면 추가
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content=AGENT_SYSTEM_PROMPT)] + messages
    
    return {"messages": [llm_with_tools.invoke(messages)]}

def route_after_classification(state: HybridAgentState) -> str:
    """분류 후 라우팅"""
    if state.get("use_rag", False):
        return "rag"
    else:
        return "agent"

# 8. 하이브리드 그래프 구성

memory = MemorySaver()

workflow = StateGraph(HybridAgentState)

workflow.add_node("classifier", classifier_node)
workflow.add_node("rag", rag_node)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", ToolNode(tools))

workflow.add_edge(START, "classifier")
workflow.add_conditional_edges(
    "classifier",
    route_after_classification,
    {"rag": "rag", "agent": "agent"}
)
workflow.add_edge("rag", END)
workflow.add_conditional_edges("agent", tools_condition)
workflow.add_edge("tools", "agent")

app = workflow.compile(checkpointer=memory)

# 9. 채팅 인터페이스

class ChatSession:
    def __init__(self, session_id: str = "default"):
        self.session_id = session_id
        self.config = {"configurable": {"thread_id": session_id}}
    
    def chat(self, user_input: str, verbose: bool = False, stream: bool = False):
        """사용자 입력 처리"""
        print(f"\n👤 사용자: {user_input}")
        
        try:
            messages = [HumanMessage(content=user_input)]
            
            if stream:
                # 스트리밍 모드
                print("🤖 챗봇: ", end="", flush=True)
                for chunk in app.stream(
                    {"messages": messages, "use_rag": False, "query_type": None},
                    config=self.config,
                    stream_mode="values"
                ):
                    if "messages" in chunk and chunk["messages"]:
                        last_msg = chunk["messages"][-1]
                        if isinstance(last_msg, AIMessage) and last_msg.content:
                            print(last_msg.content, end="", flush=True)
                print()
            else:
                # 일반 모드
                result = app.invoke(
                    {"messages": messages, "use_rag": False, "query_type": None},
                    config=self.config
                )
                
                # 처리 경로 표시
                if verbose and "query_type" in result:
                    route = "🔍 RAG (LCEL)" if result.get("use_rag") else "🤖 Agent+Tools"
                    print(f"[경로: {route} | 유형: {result['query_type']}]")
                
                bot_response = result["messages"][-1].content
                print(f"🤖 챗봇:\n{bot_response}")
        
        except Exception as e:
            print(f"❌ 오류 발생: {e}")
            import traceback
            traceback.print_exc()
    
    def clear_history(self):
        """대화 기록 초기화"""
        print("🔄 대화 기록이 초기화되었습니다.")

def interactive_chat():
    """대화형 인터페이스"""
    session = ChatSession()
    print("=" * 60)
    print("🚨 재난 안전 도우미 챗봇 (Hybrid RAG + Agent with LCEL)")
    print("=" * 60)
    print("명령어: '/exit', '/clear', '/verbose', '/stream'")
    print()
    
    verbose = False
    stream = False
    
    while True:
        try:
            user_input = input("👤 질문: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() == '/exit':
                print("👋 안녕히 가세요!")
                break
            elif user_input.lower() == '/clear':
                session.clear_history()
                continue
            elif user_input.lower() == '/verbose':
                verbose = not verbose
                print(f"🔄 Verbose 모드: {'ON' if verbose else 'OFF'}")
                continue
            elif user_input.lower() == '/stream':
                stream = not stream
                print(f"🔄 Stream 모드: {'ON' if stream else 'OFF'}")
                continue
            
            session.chat(user_input, verbose=verbose, stream=stream)
        
        except KeyboardInterrupt:
            print("\n👋 안녕히 가세요!")
            break

if __name__ == "__main__":
    interactive_chat()