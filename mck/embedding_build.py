"""
embedding_build.py
임베딩 생성 및 ChromaDB 저장 모듈

주요 기능:
1. HuggingFaceEmbeddings를 사용한 임베딩 생성
2. ChromaDB 벡터 스토어 구축
3. 메타데이터와 함께 저장
4. 리트리버 생성
"""

import os
from typing import List, Optional
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.vectorstores import VectorStoreRetriever


class EmbeddingBuilder:
    """임베딩 및 벡터DB 구축 클래스"""
    
    def __init__(
        self,
        model_name: str = "jhgan/ko-sroberta-multitask",
        persist_directory: str = "./chroma_hp",
        collection_name: str = "harry_potter_chapters"
    ):
        """
        Args:
            model_name: HuggingFace 임베딩 모델명
            persist_directory: ChromaDB 저장 디렉토리
            collection_name: 컬렉션 이름
        """
        self.model_name = model_name
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.vectorstore = None
        
        print(f"[INFO] EmbeddingBuilder 초기화")
        print(f"  - 모델: {model_name}")
        print(f"  - 저장 경로: {persist_directory}")
        print(f"  - 컬렉션: {collection_name}")
        
        # 임베딩 모델 초기화
        print(f"[INFO] 임베딩 모델 로드 중...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},  # GPU 사용 시 'cuda'
            encode_kwargs={'normalize_embeddings': True}
        )
        print(f"[SUCCESS] 임베딩 모델 로드 완료")
    
    def build_vectorstore(
        self,
        documents: List[Document],
        force_rebuild: bool = False
    ) -> Chroma:
        """
        ChromaDB 벡터 스토어 구축
        
        Args:
            documents: Document 객체 리스트
            force_rebuild: 기존 DB 삭제 후 재구축 여부
        
        Returns:
            Chroma: 벡터 스토어 객체
        """
        print(f"\n[STEP 4] ChromaDB 벡터 스토어 구축 시작")
        print("="*80)
        
        # 기존 DB 삭제 (force_rebuild=True)
        if force_rebuild and os.path.exists(self.persist_directory):
            import shutil
            print(f"[INFO] 기존 DB 삭제: {self.persist_directory}")
            shutil.rmtree(self.persist_directory)
        
        # 벡터 스토어 생성
        print(f"[INFO] 벡터 스토어 생성 중... ({len(documents)}개 문서)")
        
        try:
            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=self.persist_directory,
                collection_name=self.collection_name
            )
            
            print(f"[SUCCESS] 벡터 스토어 생성 완료")
            print(f"  - 저장 위치: {self.persist_directory}")
            print(f"  - 문서 수: {len(documents)}")
            
            # 통계 출력
            self._print_vectorstore_stats()
            
        except Exception as e:
            print(f"[ERROR] 벡터 스토어 생성 실패: {str(e)}")
            raise
        
        print("="*80 + "\n")
        
        return self.vectorstore
    
    def load_vectorstore(self) -> Optional[Chroma]:
        """
        기존 ChromaDB 벡터 스토어 로드
        
        Returns:
            Optional[Chroma]: 벡터 스토어 객체 (없으면 None)
        """
        if not os.path.exists(self.persist_directory):
            print(f"[WARNING] 벡터 스토어가 존재하지 않습니다: {self.persist_directory}")
            return None
        
        try:
            print(f"[INFO] 벡터 스토어 로드 중: {self.persist_directory}")
            
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
                collection_name=self.collection_name
            )
            
            print(f"[SUCCESS] 벡터 스토어 로드 완료")
            self._print_vectorstore_stats()
            
            return self.vectorstore
            
        except Exception as e:
            print(f"[ERROR] 벡터 스토어 로드 실패: {str(e)}")
            return None
    
    def _print_vectorstore_stats(self):
        """벡터 스토어 통계 출력"""
        if self.vectorstore is None:
            return
        
        try:
            # 컬렉션 정보
            collection = self.vectorstore._collection
            count = collection.count()
            
            print(f"\n[벡터 스토어 통계]")
            print(f"  - 총 문서 수: {count:,}")
            
            # 샘플 조회
            if count > 0:
                results = collection.peek(limit=1)
                if results and 'metadatas' in results and len(results['metadatas']) > 0:
                    sample_metadata = results['metadatas'][0]
                    print(f"  - 샘플 메타데이터 키: {list(sample_metadata.keys())}")
        
        except Exception as e:
            print(f"[WARNING] 통계 조회 실패: {str(e)}")
    
    def create_retriever(
        self,
        search_type: str = "similarity",
        k: int = 4,
        score_threshold: Optional[float] = None,
        filter_dict: Optional[dict] = None
    ) -> VectorStoreRetriever:
        """
        리트리버 생성
        
        Args:
            search_type: 검색 타입 ("similarity", "mmr", "similarity_score_threshold")
            k: 반환할 문서 수
            score_threshold: 유사도 임계값 (similarity_score_threshold 사용 시)
            filter_dict: 메타데이터 필터 (예: {"book": "해리포터와 마법사의 돌"})
        
        Returns:
            VectorStoreRetriever: 리트리버 객체
        """
        if self.vectorstore is None:
            raise ValueError("벡터 스토어가 초기화되지 않았습니다.")
        
        print(f"\n[리트리버 생성]")
        print(f"  - 검색 타입: {search_type}")
        print(f"  - k: {k}")
        
        search_kwargs = {"k": k}
        
        if filter_dict:
            search_kwargs["filter"] = filter_dict
            print(f"  - 필터: {filter_dict}")
        
        if score_threshold is not None and search_type == "similarity_score_threshold":
            search_kwargs["score_threshold"] = score_threshold
            print(f"  - 임계값: {score_threshold}")
        
        retriever = self.vectorstore.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
        )
        
        print(f"[SUCCESS] 리트리버 생성 완료")
        
        return retriever
    
    def test_retrieval(self, query: str, k: int = 3):
        """
        검색 테스트
        
        Args:
            query: 검색 쿼리
            k: 반환할 문서 수
        """
        if self.vectorstore is None:
            print("[ERROR] 벡터 스토어가 초기화되지 않았습니다.")
            return
        
        print(f"\n[검색 테스트]")
        print(f"  - 쿼리: {query}")
        print(f"  - k: {k}")
        print("-" * 80)
        
        try:
            # 유사도 검색
            results = self.vectorstore.similarity_search(query, k=k)
            
            print(f"\n[검색 결과] {len(results)}개 문서")
            
            for i, doc in enumerate(results, 1):
                print(f"\n[{i}] 책: {doc.metadata.get('book', 'N/A')}")
                print(f"    장: 제{doc.metadata.get('chapter_number', 'N/A')}장 - {doc.metadata.get('chapter_title', 'N/A')}")
                print(f"    인물: {doc.metadata.get('characters', [])}")
                print(f"    장소: {doc.metadata.get('locations', [])}")
                print(f"    내용: {doc.page_content[:150]}...")
        
        except Exception as e:
            print(f"[ERROR] 검색 실패: {str(e)}")


def main():
    """테스트용 메인 함수"""
    from preprocess import TextPreprocessor
    from chapter_splitter import ChapterSplitter
    from metadata_tagger import MetadataTagger
    
    # 1. 데이터 로드 및 전처리
    data_dir = r"c:\Users\ansck\Desktop\Project\3rd_project\data\cleaned_data"
    preprocessor = TextPreprocessor(data_dir)
    processed_files = preprocessor.preprocess_all(remove_title=False)
    
    # 2. 챕터 분리 및 청킹
    splitter = ChapterSplitter(chunk_size=800, chunk_overlap=150)
    
    all_documents = []
    for file_info in processed_files[:1]:  # 테스트: 첫 번째 책만
        book_title = preprocessor.extract_book_title(file_info['filename'])
        documents = splitter.process_book(file_info['text'], book_title)
        all_documents.extend(documents)
    
    # 3. 메타데이터 태깅
    tagger = MetadataTagger(use_llm=False)
    tagged_documents = tagger.add_metadata_to_documents(all_documents)
    
    # 4. 임베딩 및 벡터DB 구축
    builder = EmbeddingBuilder(
        model_name="jhgan/ko-sroberta-multitask",
        persist_directory="./chroma_hp_test",
        collection_name="harry_potter_test"
    )
    
    vectorstore = builder.build_vectorstore(tagged_documents, force_rebuild=True)
    
    # 5. 검색 테스트
    builder.test_retrieval("해리 포터가 호그와트에 처음 도착했을 때", k=3)
    
    # 6. 리트리버 생성
    retriever = builder.create_retriever(search_type="similarity", k=4)
    print(f"\n[완료] 리트리버 생성 완료")


if __name__ == "__main__":
    main()
