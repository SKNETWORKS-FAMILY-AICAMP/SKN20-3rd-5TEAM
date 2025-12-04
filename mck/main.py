"""
main.py
해리포터 RAG QA 챗봇 시스템 메인 실행 파일

전체 파이프라인:
1. TXT 파일 로드 및 전처리
2. 장(Chapter) 자동 감지 및 분리
3. 청킹 (RecursiveCharacterTextSplitter)
4. 메타데이터 자동 태깅
5. 임베딩 및 ChromaDB 저장
6. RAG 파이프라인 구축
7. 대화형 QA 시스템

사용법:
    python main.py --build           # 벡터DB 구축
    python main.py --query           # 질의응답 모드
    python main.py --interactive     # 대화형 모드
"""

import argparse
import os
from pathlib import Path

# 프로젝트 모듈
from preprocess import TextPreprocessor
from chapter_splitter import ChapterSplitter
from metadata_tagger import MetadataTagger
from embedding_build import EmbeddingBuilder
from rag_pipeline import SimpleRAGPipeline, RAGPipeline
from app_langgraph import LangGraphRAG, LANGGRAPH_AVAILABLE


class HarryPotterRAGSystem:
    """해리포터 RAG QA 시스템 메인 클래스"""
    
    def __init__(
        self,
        data_dir: str = "./data/cleaned_data",
        persist_dir: str = "./chroma_hp",
        collection_name: str = "harry_potter_chapters",
        embedding_model: str = "jhgan/ko-sroberta-multitask",
        chunk_size: int = 800,
        chunk_overlap: int = 150
    ):
        """
        Args:
            data_dir: TXT 파일 디렉토리
            persist_dir: ChromaDB 저장 경로
            collection_name: 컬렉션 이름
            embedding_model: 임베딩 모델명
            chunk_size: 청크 크기
            chunk_overlap: 청크 오버랩
        """
        self.data_dir = data_dir
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        print("\n" + "="*80)
        print("해리포터 RAG QA 챗봇 시스템")
        print("="*80)
        print(f"데이터 디렉토리: {data_dir}")
        print(f"벡터DB 경로: {persist_dir}")
        print(f"임베딩 모델: {embedding_model}")
        print(f"청크 설정: size={chunk_size}, overlap={chunk_overlap}")
        print("="*80 + "\n")
    
    def build_vectorstore(self, force_rebuild: bool = False):
        """
        전체 파이프라인 실행: 데이터 로드 → 청킹 → 태깅 → 벡터DB 구축
        
        Args:
            force_rebuild: 기존 DB 삭제 후 재구축 여부
        """
        print("\n" + "="*80)
        print("[전체 파이프라인 실행]")
        print("="*80 + "\n")
        
        # Step 1: 전처리
        print("\n[STEP 1] TXT 파일 로드 및 전처리")
        print("-" * 80)
        preprocessor = TextPreprocessor(self.data_dir)
        processed_files = preprocessor.preprocess_all(remove_title=False)
        
        if not processed_files:
            print("[ERROR] 로드된 파일이 없습니다. 데이터 디렉토리를 확인하세요.")
            return False
        
        # Step 2: 장 분리 및 청킹
        print("\n[STEP 2] 장 분리 및 청킹")
        print("-" * 80)
        splitter = ChapterSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        
        all_documents = []
        for file_info in processed_files:
            book_title = preprocessor.extract_book_title(file_info['filename'])
            documents = splitter.process_book(file_info['text'], book_title)
            all_documents.extend(documents)
        
        print(f"\n[완료] 총 {len(all_documents)}개 청크 생성")
        
        # Step 3: 메타데이터 태깅
        print("\n[STEP 3] 메타데이터 자동 태깅")
        print("-" * 80)
        tagger = MetadataTagger(use_llm=False)
        tagged_documents = tagger.add_metadata_to_documents(all_documents)
        
        # 통계 출력
        tagger.print_metadata_stats(tagged_documents)
        
        # Step 4: 임베딩 및 벡터DB 구축
        print("\n[STEP 4] 임베딩 및 ChromaDB 구축")
        print("-" * 80)
        builder = EmbeddingBuilder(
            model_name=self.embedding_model,
            persist_directory=self.persist_dir,
            collection_name=self.collection_name
        )
        
        vectorstore = builder.build_vectorstore(
            documents=tagged_documents,
            force_rebuild=force_rebuild
        )
        
        print("\n" + "="*80)
        print("[전체 파이프라인 완료]")
        print("="*80)
        print(f"✓ 처리된 책: {len(processed_files)}권")
        print(f"✓ 생성된 청크: {len(tagged_documents):,}개")
        print(f"✓ 벡터DB 저장: {self.persist_dir}")
        print("="*80 + "\n")
        
        return True
    
    def load_system(self):
        """
        기존 벡터DB 로드 및 RAG 시스템 구성
        
        Returns:
            tuple: (builder, retriever, rag_pipeline)
        """
        print("\n[시스템 로드]")
        print("-" * 80)
        
        # 벡터 스토어 로드
        builder = EmbeddingBuilder(
            model_name=self.embedding_model,
            persist_directory=self.persist_dir,
            collection_name=self.collection_name
        )
        
        vectorstore = builder.load_vectorstore()
        
        if vectorstore is None:
            print("[ERROR] 벡터 스토어를 찾을 수 없습니다.")
            print("먼저 'python main.py --build'로 벡터DB를 구축하세요.")
            return None, None, None
        
        # 리트리버 생성
        retriever = builder.create_retriever(
            search_type="similarity",
            k=4
        )
        
        # RAG 파이프라인 생성 (Simple 버전 - LLM 없이)
        rag_pipeline = SimpleRAGPipeline(retriever, k=4)
        
        print("-" * 80)
        print("[시스템 로드 완료]\n")
        
        return builder, retriever, rag_pipeline
    
    def query_mode(self, questions: list = None):
        """
        질의응답 모드
        
        Args:
            questions: 질문 리스트 (None이면 기본 질문 사용)
        """
        # 시스템 로드
        builder, retriever, rag = self.load_system()
        
        if rag is None:
            return
        
        # 기본 질문
        if questions is None:
            questions = [
                "해리 포터가 호그와트에 처음 도착했을 때 무슨 일이 있었나요?",
                "덤블도어는 어떤 사람인가요?",
                "해리의 가장 친한 친구들은 누구인가요?",
                "볼드모트는 누구인가요?",
                "호그와트의 기숙사는 어떤 것들이 있나요?"
            ]
        
        print("\n" + "="*80)
        print("[질의응답 모드]")
        print("="*80 + "\n")
        
        for i, question in enumerate(questions, 1):
            print(f"\n{'='*80}")
            print(f"[질문 {i}/{len(questions)}]")
            print(f"{'='*80}\n")
            
            result = rag.invoke(question)
            rag.print_result(result)
            
            input("\n계속하려면 Enter를 누르세요...")
    
    def interactive_mode(self, use_langgraph: bool = False):
        """
        대화형 모드
        
        Args:
            use_langgraph: LangGraph 사용 여부
        """
        # 시스템 로드
        builder, retriever, _ = self.load_system()
        
        if retriever is None:
            return
        
        # RAG 선택
        if use_langgraph and LANGGRAPH_AVAILABLE:
            print("\n[INFO] LangGraph RAG 모드")
            rag = LangGraphRAG(retriever, use_llm=False)
        else:
            print("\n[INFO] Simple RAG 모드")
            rag = SimpleRAGPipeline(retriever, k=4)
        
        print("\n" + "="*80)
        print("[대화형 모드]")
        print("="*80)
        print("해리포터에 대해 무엇이든 물어보세요!")
        print("종료하려면 'quit', 'exit', 'q'를 입력하세요.")
        print("="*80 + "\n")
        
        while True:
            try:
                # 질문 입력
                question = input("\n질문: ").strip()
                
                # 종료 체크
                if question.lower() in ['quit', 'exit', 'q', '종료']:
                    print("\n시스템을 종료합니다.")
                    break
                
                if not question:
                    continue
                
                # 질의응답
                result = rag.invoke(question)
                rag.print_result(result)
                
            except KeyboardInterrupt:
                print("\n\n시스템을 종료합니다.")
                break
            except Exception as e:
                print(f"\n[ERROR] {str(e)}")
    
    def test_search(self, query: str, k: int = 5):
        """
        검색 테스트
        
        Args:
            query: 검색 쿼리
            k: 검색할 문서 수
        """
        # 시스템 로드
        builder, retriever, _ = self.load_system()
        
        if builder is None:
            return
        
        # 검색 테스트
        builder.test_retrieval(query, k=k)


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description="해리포터 RAG QA 챗봇 시스템"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        default="interactive",
        choices=["build", "query", "interactive", "test"],
        help="실행 모드 (build: DB구축, query: 질의응답, interactive: 대화형, test: 검색테스트)"
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        default=r"c:\Users\ansck\Desktop\Project\3rd_project\data\cleaned_data",
        help="TXT 파일 디렉토리"
    )
    
    parser.add_argument(
        "--persist-dir",
        type=str,
        default="./chroma_hp",
        help="ChromaDB 저장 경로"
    )
    
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="기존 DB 삭제 후 재구축"
    )
    
    parser.add_argument(
        "--use-langgraph",
        action="store_true",
        help="LangGraph 사용 (대화형 모드에서)"
    )
    
    parser.add_argument(
        "--test-query",
        type=str,
        default="해리 포터가 호그와트에 처음 도착했을 때",
        help="테스트 검색 쿼리"
    )
    
    args = parser.parse_args()
    
    # 시스템 초기화
    system = HarryPotterRAGSystem(
        data_dir=args.data_dir,
        persist_dir=args.persist_dir,
        collection_name="harry_potter_chapters",
        embedding_model="jhgan/ko-sroberta-multitask",
        chunk_size=800,
        chunk_overlap=150
    )
    
    # 모드별 실행
    if args.mode == "build":
        system.build_vectorstore(force_rebuild=args.force_rebuild)
    
    elif args.mode == "query":
        system.query_mode()
    
    elif args.mode == "interactive":
        system.interactive_mode(use_langgraph=args.use_langgraph)
    
    elif args.mode == "test":
        system.test_search(args.test_query, k=5)


if __name__ == "__main__":
    main()
