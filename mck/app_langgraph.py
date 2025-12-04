"""
app_langgraph.py
LangGraph 기반 고급 RAG 시스템

주요 기능:
1. 상태 기반 워크플로우
2. 질의 재작성 노드
3. 검색 노드
4. 재순위 노드
5. 답변 생성 노드
6. 출력 노드
"""

from typing import Dict, List, Any, TypedDict, Annotated
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import HuggingFaceEndpoint
import operator
import os


# LangGraph import (optional)
try:
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    print("[WARNING] LangGraph를 사용할 수 없습니다. pip install langgraph 필요")


class GraphState(TypedDict):
    """그래프 상태 정의"""
    question: str  # 원본 질문
    rewritten_question: str  # 재작성된 질문
    documents: List[Document]  # 검색된 문서
    reranked_documents: List[Document]  # 재순위된 문서
    answer: str  # 최종 답변
    steps: Annotated[List[str], operator.add]  # 실행 단계 추적


class LangGraphRAG:
    """LangGraph 기반 RAG 시스템"""
    
    def __init__(
        self,
        retriever,
        huggingface_token: str = None,
        use_llm: bool = False
    ):
        """
        Args:
            retriever: 벡터 스토어 리트리버
            huggingface_token: HuggingFace API 토큰
            use_llm: LLM 사용 여부
        """
        self.retriever = retriever
        self.use_llm = use_llm
        
        print(f"[INFO] LangGraphRAG 초기화")
        print(f"  - LangGraph 사용 가능: {LANGGRAPH_AVAILABLE}")
        print(f"  - LLM 사용: {use_llm}")
        
        # LLM 초기화 (옵션)
        if use_llm:
            if huggingface_token is None:
                huggingface_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN", "")
            
            try:
                self.llm = HuggingFaceEndpoint(
                    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
                    temperature=0.0,
                    max_new_tokens=300,
                    huggingfacehub_api_token=huggingface_token
                )
                print(f"[SUCCESS] LLM 초기화 완료")
            except Exception as e:
                print(f"[WARNING] LLM 초기화 실패: {str(e)}")
                self.llm = None
                self.use_llm = False
        else:
            self.llm = None
        
        # 그래프 구성
        if LANGGRAPH_AVAILABLE:
            self.graph = self._build_graph()
            self.app = self.graph.compile()
        else:
            self.graph = None
            self.app = None
    
    def _build_graph(self):
        """LangGraph 그래프 구성"""
        print(f"[INFO] LangGraph 구성 중...")
        
        # StateGraph 생성
        workflow = StateGraph(GraphState)
        
        # 노드 추가
        workflow.add_node("rewrite", self.rewrite_node)
        workflow.add_node("retrieve", self.retrieve_node)
        workflow.add_node("rerank", self.rerank_node)
        workflow.add_node("generate", self.generate_node)
        workflow.add_node("output", self.output_node)
        
        # 엣지 추가 (워크플로우 정의)
        workflow.set_entry_point("rewrite")
        workflow.add_edge("rewrite", "retrieve")
        workflow.add_edge("retrieve", "rerank")
        workflow.add_edge("rerank", "generate")
        workflow.add_edge("generate", "output")
        workflow.add_edge("output", END)
        
        print(f"[SUCCESS] LangGraph 구성 완료")
        
        return workflow
    
    def rewrite_node(self, state: GraphState) -> GraphState:
        """
        질의 재작성 노드
        
        Args:
            state: 현재 상태
        
        Returns:
            GraphState: 업데이트된 상태
        """
        print(f"[노드] rewrite_node 실행")
        
        question = state["question"]
        
        # LLM 있으면 재작성, 없으면 원본 사용
        if self.use_llm and self.llm is not None:
            try:
                prompt = PromptTemplate(
                    template="""질문을 더 명확하고 검색하기 좋게 재작성하세요.
                    
원본 질문: {question}

재작성된 질문:""",
                    input_variables=["question"]
                )
                
                chain = prompt | self.llm | StrOutputParser()
                rewritten = chain.invoke({"question": question})
                
                print(f"  - 재작성: {question} → {rewritten}")
            except Exception as e:
                print(f"  - 재작성 실패: {str(e)}, 원본 사용")
                rewritten = question
        else:
            rewritten = question
            print(f"  - 원본 사용: {question}")
        
        return {
            **state,
            "rewritten_question": rewritten,
            "steps": ["rewrite"]
        }
    
    def retrieve_node(self, state: GraphState) -> GraphState:
        """
        검색 노드
        
        Args:
            state: 현재 상태
        
        Returns:
            GraphState: 업데이트된 상태
        """
        print(f"[노드] retrieve_node 실행")
        
        question = state.get("rewritten_question", state["question"])
        
        try:
            # 문서 검색
            documents = self.retriever.invoke(question)
            print(f"  - 검색 완료: {len(documents)}개 문서")
        except Exception as e:
            print(f"  - 검색 실패: {str(e)}")
            documents = []
        
        return {
            **state,
            "documents": documents,
            "steps": ["retrieve"]
        }
    
    def rerank_node(self, state: GraphState) -> GraphState:
        """
        재순위 노드 (간단한 버전: 상위 3개만 선택)
        
        Args:
            state: 현재 상태
        
        Returns:
            GraphState: 업데이트된 상태
        """
        print(f"[노드] rerank_node 실행")
        
        documents = state["documents"]
        
        # 간단한 재순위: 상위 3개만 선택
        reranked = documents[:3]
        
        print(f"  - 재순위 완료: {len(documents)}개 → {len(reranked)}개")
        
        return {
            **state,
            "reranked_documents": reranked,
            "steps": ["rerank"]
        }
    
    def generate_node(self, state: GraphState) -> GraphState:
        """
        답변 생성 노드
        
        Args:
            state: 현재 상태
        
        Returns:
            GraphState: 업데이트된 상태
        """
        print(f"[노드] generate_node 실행")
        
        question = state["question"]
        documents = state.get("reranked_documents", state.get("documents", []))
        
        # 컨텍스트 포맷팅
        context = self._format_documents(documents)
        
        # LLM 있으면 답변 생성, 없으면 컨텍스트만
        if self.use_llm and self.llm is not None:
            try:
                prompt = PromptTemplate(
                    template="""다음 컨텍스트를 바탕으로 질문에 답변하세요.

컨텍스트:
{context}

질문: {question}

답변:""",
                    input_variables=["context", "question"]
                )
                
                chain = prompt | self.llm | StrOutputParser()
                answer = chain.invoke({"context": context, "question": question})
                
                print(f"  - 답변 생성 완료")
            except Exception as e:
                print(f"  - 답변 생성 실패: {str(e)}")
                answer = f"[답변 생성 실패] {str(e)}"
        else:
            answer = f"[검색된 문서 수: {len(documents)}]\n\n{context}"
            print(f"  - 컨텍스트 반환")
        
        return {
            **state,
            "answer": answer,
            "steps": ["generate"]
        }
    
    def output_node(self, state: GraphState) -> GraphState:
        """
        출력 노드
        
        Args:
            state: 현재 상태
        
        Returns:
            GraphState: 최종 상태
        """
        print(f"[노드] output_node 실행")
        print(f"  - 실행된 단계: {state['steps']}")
        
        return {
            **state,
            "steps": ["output"]
        }
    
    def _format_documents(self, documents: List[Document]) -> str:
        """문서 포맷팅"""
        formatted = []
        
        for i, doc in enumerate(documents, 1):
            metadata = doc.metadata
            formatted.append(
                f"[문서 {i}] {metadata.get('book', 'N/A')} - "
                f"제{metadata.get('chapter_number', 'N/A')}장\n"
                f"{doc.page_content}\n"
            )
        
        return "\n".join(formatted)
    
    def invoke(self, question: str) -> Dict[str, Any]:
        """
        질문 처리 (동기 방식)
        
        Args:
            question: 질문
        
        Returns:
            Dict: 결과
        """
        print(f"\n[질문] {question}")
        print("="*80)
        
        if not LANGGRAPH_AVAILABLE or self.app is None:
            print("[ERROR] LangGraph를 사용할 수 없습니다.")
            return self._fallback_invoke(question)
        
        try:
            # 초기 상태
            initial_state = {
                "question": question,
                "rewritten_question": "",
                "documents": [],
                "reranked_documents": [],
                "answer": "",
                "steps": []
            }
            
            # 그래프 실행
            result = self.app.invoke(initial_state)
            
            print("="*80)
            print(f"[완료] 총 {len(result['steps'])}개 노드 실행")
            
            return result
            
        except Exception as e:
            print(f"[ERROR] 그래프 실행 실패: {str(e)}")
            return self._fallback_invoke(question)
    
    def stream(self, question: str):
        """
        질문 처리 (스트리밍 방식)
        
        Args:
            question: 질문
        
        Yields:
            Dict: 각 단계별 결과
        """
        print(f"\n[질문 (스트리밍)] {question}")
        print("="*80)
        
        if not LANGGRAPH_AVAILABLE or self.app is None:
            print("[ERROR] LangGraph를 사용할 수 없습니다.")
            yield self._fallback_invoke(question)
            return
        
        try:
            # 초기 상태
            initial_state = {
                "question": question,
                "rewritten_question": "",
                "documents": [],
                "reranked_documents": [],
                "answer": "",
                "steps": []
            }
            
            # 그래프 스트리밍 실행
            for output in self.app.stream(initial_state):
                yield output
            
            print("="*80)
            print(f"[완료] 스트리밍 종료")
            
        except Exception as e:
            print(f"[ERROR] 스트리밍 실패: {str(e)}")
            yield self._fallback_invoke(question)
    
    def _fallback_invoke(self, question: str) -> Dict[str, Any]:
        """Fallback: 단순 검색"""
        print(f"[INFO] Fallback 모드: 단순 검색만 수행")
        
        try:
            documents = self.retriever.invoke(question)
            context = self._format_documents(documents[:3])
            
            return {
                "question": question,
                "rewritten_question": question,
                "documents": documents,
                "reranked_documents": documents[:3],
                "answer": f"[검색 결과]\n\n{context}",
                "steps": ["fallback"]
            }
        except Exception as e:
            return {
                "question": question,
                "answer": f"오류 발생: {str(e)}",
                "steps": ["error"]
            }
    
    def print_result(self, result: Dict[str, Any]):
        """결과 출력"""
        print("\n" + "="*80)
        print("[LangGraph RAG 결과]")
        print("="*80)
        
        print(f"\n[질문] {result['question']}")
        
        if result.get('rewritten_question') and result['rewritten_question'] != result['question']:
            print(f"\n[재작성된 질문] {result['rewritten_question']}")
        
        print(f"\n[답변]")
        print(result['answer'])
        
        if 'reranked_documents' in result:
            docs = result['reranked_documents']
        elif 'documents' in result:
            docs = result['documents']
        else:
            docs = []
        
        print(f"\n[참고 문서] {len(docs)}개")
        for i, doc in enumerate(docs, 1):
            print(f"  [{i}] {doc.metadata.get('book')} - 제{doc.metadata.get('chapter_number')}장")
        
        print(f"\n[실행 단계] {' → '.join(result.get('steps', []))}")
        print("="*80)


def main():
    """테스트용 메인 함수"""
    from embedding_build import EmbeddingBuilder
    
    # 1. 벡터 스토어 로드
    builder = EmbeddingBuilder(
        model_name="jhgan/ko-sroberta-multitask",
        persist_directory="./chroma_hp_test",
        collection_name="harry_potter_test"
    )
    
    vectorstore = builder.load_vectorstore()
    
    if vectorstore is None:
        print("[ERROR] 먼저 벡터 스토어를 구축해주세요")
        return
    
    # 2. 리트리버 생성
    retriever = builder.create_retriever(search_type="similarity", k=5)
    
    # 3. LangGraph RAG 생성
    rag = LangGraphRAG(retriever, use_llm=False)
    
    # 4. 테스트
    test_question = "해리 포터의 친구들은 누구인가요?"
    
    print("\n[테스트 1: invoke()]")
    result = rag.invoke(test_question)
    rag.print_result(result)
    
    if LANGGRAPH_AVAILABLE and rag.app is not None:
        print("\n[테스트 2: stream()]")
        for step_output in rag.stream(test_question):
            print(f"\n[스트림 출력] {list(step_output.keys())}")


if __name__ == "__main__":
    main()
