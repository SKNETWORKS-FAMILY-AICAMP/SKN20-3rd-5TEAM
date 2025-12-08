"""
임베딩 및 벡터 DB 모듈
LangChain과 OpenAI를 활용한 텍스트 임베딩 생성 및 벡터 DB 구축 함수
"""
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from typing import List
from dotenv import load_dotenv
import os

load_dotenv()  # .env 파일에서 환경 변수 로드

def create_embeddings_and_vectordb(documents: List[Document]):
    """
    문서 리스트로부터 OpenAI 임베딩 객체 생성

    Args:
    - documents (List[Document]): LangChain Document 리스트

    Returns:
    - OpenAIEmbeddings: 생성된 임베딩 객체
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=os.getenv("OPENAI_API_KEY"))

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name="shelter_and_disaster_guidelines",
        persist_directory="./chroma_db"
    )

    print(f"VectorDB 생성 완료: {len(documents)}개 문서 저장")
