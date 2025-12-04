"""
rag_pipeline.py
RAG LCEL 파이프라인 구축 모듈

주요 기능:
1. 질의 재작성 (Query Rewriting)
2. 컨텍스트 검색 (Retrieval)
3. LLM 기반 답변 생성
4. LCEL 체인 구성
"""

from typing import List, Dict, Any
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_community.llms import HuggingFaceEndpoint
from langchain_core.documents import Document
import os


class RAGPipeline:
    """RAG LCEL 파이프라인 클래스"""
    
    def __init__(
        self,
        retriever,
        huggingface_token: str = None,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
        temperature: float = 0.0,
        max_new_tokens: int = 300
    ):
        """
        Args:
            retriever: 벡터 스토어 리트리버
            huggingface_token: HuggingFace API 토큰
            model_name: HuggingFace 모델명
            temperature: 온도 (0=결정적)
            max_new_tokens: 최대 생성 토큰 수
        """
        self.retriever = retriever
        self.model_name = model_name
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        
        print(f"[INFO] RAGPipeline 초기화")
        print(f"  - 모델: {model_name}")
        print(f"  - Temperature: {temperature}")
        print(f"  - Max tokens: {max_new_tokens}")
        
        # HuggingFace 토큰 설정
        if huggingface_token is None:
            huggingface_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN", "")
        
        # LLM 초기화
        print(f"[INFO] LLM 초기화 중...")
        try:
            self.llm = HuggingFaceEndpoint(
                repo_id=model_name,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                huggingfacehub_api_token=huggingface_token
            )
            print(f"[SUCCESS] LLM 초기화 완료")
        except Exception as e:
            print(f"[WARNING] HuggingFace LLM 초기화 실패: {str(e)}")
            print(f"[INFO] 로컬 모드로 전환 (답변 생성 불가)")
            self.llm = None
        
        # LCEL 체인 구성
        self._build_chains()
    
    def _build_chains(self):
        """LCEL 체인 구성"""
        print(f"\n[INFO] LCEL 체인 구성 중...")
        
        # 1. 질의 재작성 프롬프트
        self.query_rewrite_prompt = PromptTemplate(
            template="""당신은 해리포터 시리즈 전문가입니다. 
사용자의 질문을 더 명확하고 검색하기 좋은 형태로 재작성하세요.

원본 질문: {question}

재작성된 질문:""",
            input_variables=["question"]
        )
        
        # 2. RAG 프롬프트
        self.rag_prompt = ChatPromptTemplate.from_template(
            """당신은 해리포터 시리즈에 대한 질문에 답변하는 AI 어시스턴트입니다.
아래 제공된 컨텍스트를 바탕으로 질문에 답변하세요.

컨텍스트:
{context}

질문: {question}

답변 규칙:
1. 컨텍스트에 있는 정보만 사용하세요
2. 정확하고 간결하게 답변하세요
3. 컨텍스트에 정보가 없으면 "제공된 정보로는 답변할 수 없습니다"라고 하세요
4. 한국어로 답변하세요

답변:"""
        )
        
        # 3. 질의 재작성 체인 (LLM 있을 때만)
        if self.llm is not None:
            self.query_rewrite_chain = (
                self.query_rewrite_prompt
                | self.llm
                | StrOutputParser()
            )
        else:
            # LLM 없으면 원본 질문 그대로 사용
            self.query_rewrite_chain = RunnablePassthrough()
        
        # 4. RAG 체인
        if self.llm is not None:
            self.rag_chain = (
                RunnableParallel({
                    "context": self.retriever | self._format_docs,
                    "question": RunnablePassthrough()
                })
                | self.rag_prompt
                | self.llm
                | StrOutputParser()
            )
        else:
            # LLM 없으면 컨텍스트만 반환
            self.rag_chain = (
                RunnableParallel({
                    "context": self.retriever | self._format_docs,
                    "question": RunnablePassthrough()
                })
            )
        
        print(f"[SUCCESS] LCEL 체인 구성 완료")
    
    def _format_docs(self, docs: List[Document]) -> str:
        """
        Document 리스트를 문자열로 포맷팅
        
        Args:
            docs: Document 리스트
        
        Returns:
            str: 포맷팅된 컨텍스트
        """
        formatted = []
        
        for i, doc in enumerate(docs, 1):
            metadata = doc.metadata
            book = metadata.get('book', 'N/A')
            chapter = metadata.get('chapter_number', 'N/A')
            chapter_title = metadata.get('chapter_title', 'N/A')
            
            formatted.append(
                f"[문서 {i}]\n"
                f"출처: {book} - 제{chapter}장 {chapter_title}\n"
                f"내용: {doc.page_content}\n"
            )
        
        return "\n".join(formatted)
    
    def invoke(self, question: str) -> Dict[str, Any]:
        """
        질문에 대한 답변 생성 (동기 방식)
        
        Args:
            question: 질문
        
        Returns:
            Dict: {"question": ..., "answer": ..., "context": ...}
        """
        print(f"\n[질문] {question}")
        print("-" * 80)
        
        try:
            # 1. 질의 재작성 (옵션)
            if self.llm is not None and hasattr(self, 'query_rewrite_chain'):
                try:
                    rewritten_question = self.query_rewrite_chain.invoke({"question": question})
                    print(f"[재작성된 질문] {rewritten_question}")
                except:
                    rewritten_question = question
            else:
                rewritten_question = question
            
            # 2. 컨텍스트 검색
            print(f"[검색 중...]")
            retrieved_docs = self.retriever.invoke(rewritten_question)
            print(f"[검색 완료] {len(retrieved_docs)}개 문서 검색됨")
            
            # 3. 답변 생성
            if self.llm is not None:
                print(f"[답변 생성 중...]")
                answer = self.rag_chain.invoke(rewritten_question)
                print(f"[답변 생성 완료]")
            else:
                answer = "[LLM 미설정] 컨텍스트만 검색되었습니다."
            
            # 4. 결과 반환
            result = {
                "question": question,
                "rewritten_question": rewritten_question,
                "answer": answer if isinstance(answer, str) else str(answer),
                "context": retrieved_docs,
                "num_docs": len(retrieved_docs)
            }
            
            return result
            
        except Exception as e:
            print(f"[ERROR] 파이프라인 실행 실패: {str(e)}")
            return {
                "question": question,
                "answer": f"오류 발생: {str(e)}",
                "context": [],
                "num_docs": 0
            }
    
    def print_result(self, result: Dict[str, Any]):
        """
        결과 출력
        
        Args:
            result: invoke() 결과
        """
        print("\n" + "="*80)
        print("[RAG 파이프라인 결과]")
        print("="*80)
        
        print(f"\n[질문] {result['question']}")
        
        if result.get('rewritten_question') != result['question']:
            print(f"\n[재작성된 질문] {result['rewritten_question']}")
        
        print(f"\n[답변]")
        print(result['answer'])
        
        print(f"\n[참고 문서] {result['num_docs']}개")
        for i, doc in enumerate(result['context'], 1):
            print(f"\n  [{i}] {doc.metadata.get('book')} - 제{doc.metadata.get('chapter_number')}장")
            print(f"      {doc.page_content[:100]}...")
        
        print("\n" + "="*80)


class SimpleRAGPipeline:
    """LLM 없이 검색만 수행하는 간단한 RAG 파이프라인"""
    
    def __init__(self, retriever, k: int = 4):
        """
        Args:
            retriever: 벡터 스토어 리트리버
            k: 검색할 문서 수
        """
        self.retriever = retriever
        self.k = k
        
        print(f"[INFO] SimpleRAGPipeline 초기화 (검색 전용)")
    
    def invoke(self, question: str) -> Dict[str, Any]:
        """
        질문에 대한 컨텍스트 검색
        
        Args:
            question: 질문
        
        Returns:
            Dict: {"question": ..., "context": ...}
        """
        print(f"\n[질문] {question}")
        
        try:
            # 컨텍스트 검색
            retrieved_docs = self.retriever.invoke(question)
            
            result = {
                "question": question,
                "context": retrieved_docs,
                "num_docs": len(retrieved_docs)
            }
            
            return result
            
        except Exception as e:
            print(f"[ERROR] 검색 실패: {str(e)}")
            return {
                "question": question,
                "context": [],
                "num_docs": 0
            }
    
    def print_result(self, result: Dict[str, Any]):
        """결과 출력"""
        print("\n" + "="*80)
        print(f"[질문] {result['question']}")
        print(f"\n[검색 결과] {result['num_docs']}개 문서")
        print("="*80)
        
        for i, doc in enumerate(result['context'], 1):
            print(f"\n[{i}] {doc.metadata.get('book')} - 제{doc.metadata.get('chapter_number')}장: {doc.metadata.get('chapter_title')}")
            print(f"    인물: {doc.metadata.get('characters', [])}")
            print(f"    장소: {doc.metadata.get('locations', [])}")
            print(f"    감정: {doc.metadata.get('sentiment', 'N/A')}")
            print(f"    내용: {doc.page_content[:200]}...")
        
        print("\n" + "="*80)


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
        print("[ERROR] 먼저 벡터 스토어를 구축해주세요 (embedding_build.py 실행)")
        return
    
    # 2. 리트리버 생성
    retriever = builder.create_retriever(search_type="similarity", k=4)
    
    # 3. RAG 파이프라인 생성 (Simple 버전 - LLM 없이)
    print("\n[INFO] Simple RAG 파이프라인 사용 (검색 전용)")
    rag = SimpleRAGPipeline(retriever, k=4)
    
    # 4. 테스트 질문
    test_questions = [
        "해리 포터가 호그와트에 처음 도착했을 때 무슨 일이 있었나요?",
        "덤블도어는 어떤 사람인가요?",
        "해리의 친구들은 누구인가요?"
    ]
    
    for question in test_questions:
        result = rag.invoke(question)
        rag.print_result(result)
        print("\n")


if __name__ == "__main__":
    main()
