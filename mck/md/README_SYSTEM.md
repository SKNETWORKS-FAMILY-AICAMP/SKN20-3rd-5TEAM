# 해리포터 RAG QA 챗봇 시스템

해리포터 시리즈 TXT 파일을 기반으로 한 고급 RAG(Retrieval-Augmented Generation) 질의응답 시스템입니다.

## 📋 프로젝트 개요

이 프로젝트는 해리포터 소설 텍스트를 자동으로 처리하여 지능형 질의응답 시스템을 구축합니다.

### 주요 기능

✅ **자동 텍스트 처리**
- TXT 파일 로드 및 UTF-8 디코딩
- 불규칙한 개행/공백 자동 정리
- 첫 줄(책 제목) 자동 제거

✅ **장(Chapter) 자동 감지**
- 정규식 기반 다양한 패턴 감지
  - "제 1장", "제1장", "1장"
  - "CHAPTER ONE", "Chapter 1"
  - "1. 제목"
- 장별 자동 분리

✅ **스마트 청킹**
- RecursiveCharacterTextSplitter 사용
- chunk_size: 800, chunk_overlap: 150
- 한국어 최적화 구분자: ["\n\n", "\n", " ", "다.", "요."]

✅ **자동 메타데이터 태깅**
- 인물 자동 추출 (해리, 론, 헤르미온느 등)
- 장소 자동 추출 (호그와트, 다이애건 앨리 등)
- 감정 분석 (positive/negative/neutral)
- 키워드 추출 (빈도 기반)
- 자동 요약 생성

✅ **벡터DB 구축**
- HuggingFace Embeddings (jhgan/ko-sroberta-multitask)
- ChromaDB 영구 저장
- 메타데이터 기반 필터링 지원

✅ **RAG 파이프라인**
- LCEL(LangChain Expression Language) 기반
- 질의 재작성
- 컨텍스트 검색
- 유사도 기반 리트리버

✅ **LangGraph 지원**
- 상태 기반 워크플로우
- rewrite → retrieve → rerank → generate → output
- invoke() / stream() 지원

## 🗂️ 프로젝트 구조

```
3rd_project/
├── data/
│   └── cleaned_data/          # TXT 파일들
│       ├── cleaned_해리포터와 마법사의 돌.txt
│       ├── cleaned_해리포터와 비밀의 방.txt
│       └── ...
├── preprocess.py              # TXT 로드 및 전처리
├── chapter_splitter.py        # 장 감지 및 청킹
├── metadata_tagger.py         # 메타데이터 태깅
├── embedding_build.py         # 임베딩 및 ChromaDB
├── rag_pipeline.py            # RAG LCEL 파이프라인
├── app_langgraph.py           # LangGraph 구현
├── main.py                    # 통합 실행 파일
├── requirements.txt           # 패키지 의존성
└── README.md                  # 이 파일
```

## 🚀 설치 및 실행

### 1. 환경 설정

```bash
# Conda 환경 생성 (Python 3.12)
conda create -n 3rd_project python=3.12 -y
conda activate 3rd_project

# 패키지 설치
pip install -r requirements.txt
```

### 2. 데이터 준비

`data/cleaned_data/` 디렉토리에 해리포터 TXT 파일들을 배치합니다.

```
data/cleaned_data/
├── cleaned_해리포터와 마법사의 돌.txt
├── cleaned_해리포터와 비밀의 방.txt
├── cleaned_해리포터와 아즈카반의 죄수.txt
├── cleaned_해리포터와 불의 잔.txt
├── cleaned_해리포터와 불사조기사단.txt
├── cleaned_해리포터와 혼혈왕자.txt
└── cleaned_해리포터와 죽음의 성물.txt
```

### 3. 벡터DB 구축

```bash
# 처음 실행 시 (벡터DB 구축)
python main.py --mode build

# 기존 DB 삭제 후 재구축
python main.py --mode build --force-rebuild
```

실행 시 다음 과정이 자동으로 진행됩니다:
1. TXT 파일 로드 및 전처리
2. 장 자동 감지 및 분리
3. 청킹 (800자 단위)
4. 메타데이터 자동 태깅
5. 임베딩 생성 및 ChromaDB 저장

### 4. 질의응답 실행

#### 방법 1: 대화형 모드 (추천)

```bash
python main.py --mode interactive
```

대화형으로 질문을 입력하고 답변을 받을 수 있습니다.

```
질문: 해리 포터의 친구들은 누구인가요?
[검색 결과] ...
```

종료: `quit`, `exit`, `q` 입력

#### 방법 2: 미리 정의된 질문 모드

```bash
python main.py --mode query
```

5개의 기본 질문에 자동으로 답변합니다.

#### 방법 3: 검색 테스트

```bash
python main.py --mode test --test-query "해리가 호그와트에 처음 도착했을 때"
```

특정 쿼리로 검색 결과만 확인합니다.

### 5. LangGraph 모드 (선택)

```bash
# LangGraph 설치
pip install langgraph

# LangGraph 대화형 모드
python main.py --mode interactive --use-langgraph
```

## 📝 모듈별 실행

각 모듈을 개별적으로 테스트할 수 있습니다:

```bash
# 1. 전처리 테스트
python preprocess.py

# 2. 장 분리 테스트
python chapter_splitter.py

# 3. 메타데이터 태깅 테스트
python metadata_tagger.py

# 4. 임베딩 구축 테스트
python embedding_build.py

# 5. RAG 파이프라인 테스트
python rag_pipeline.py

# 6. LangGraph 테스트
python app_langgraph.py
```

## 🔧 커스터마이징

### 청크 크기 조정

`main.py`의 `HarryPotterRAGSystem` 초기화 부분:

```python
system = HarryPotterRAGSystem(
    chunk_size=800,      # 청크 크기
    chunk_overlap=150    # 오버랩 크기
)
```

### 검색 결과 수 조정

`main.py`의 리트리버 생성 부분:

```python
retriever = builder.create_retriever(
    search_type="similarity",
    k=4  # 검색 결과 수
)
```

### 임베딩 모델 변경

```python
system = HarryPotterRAGSystem(
    embedding_model="jhgan/ko-sroberta-multitask"  # 다른 모델로 변경 가능
)
```

추천 모델:
- `jhgan/ko-sroberta-multitask` (한국어 특화)
- `sentence-transformers/all-mpnet-base-v2` (영어)
- `intfloat/multilingual-e5-large` (다국어)

### 메타데이터 필터링

특정 책이나 장만 검색:

```python
retriever = builder.create_retriever(
    search_type="similarity",
    k=4,
    filter_dict={"book": "해리포터와 마법사의 돌"}
)
```

## 📊 시스템 출력 예시

### 벡터DB 구축 결과

```
================================================================================
[전체 파이프라인 완료]
================================================================================
✓ 처리된 책: 7권
✓ 생성된 청크: 2,847개
✓ 벡터DB 저장: ./chroma_hp
================================================================================
```

### 질의응답 결과

```
================================================================================
[질문] 해리 포터의 친구들은 누구인가요?
================================================================================

[답변]
해리 포터의 가장 친한 친구들은 론 위즐리와 헤르미온느 그레인저입니다.
론은 해리가 호그와트로 가는 기차에서 처음 만났고, 헤르미온느는 
학교에서 만나 친구가 되었습니다.

[참고 문서] 4개
  [1] 해리포터와 마법사의 돌 - 제6장
      호그와트 급행열차에서 론 위즐리를 만났다...
  [2] 해리포터와 마법사의 돌 - 제9장
      헤르미온느 그레인저가 트롤 사건 이후 친구가 되었다...
  ...
```

## 🎯 주요 특징

### 1. 완전 자동화
- 수동 작업 없이 TXT 파일만 있으면 전체 시스템 구축
- 장 감지, 메타데이터 태깅 모두 자동

### 2. 한국어 최적화
- 한국어 임베딩 모델 사용
- 한국어 텍스트 구조에 맞는 청킹
- 한국어 감정 분석 및 키워드 추출

### 3. 풍부한 메타데이터
- 책 제목, 장 번호, 장 제목
- 등장 인물, 장소
- 감정, 키워드, 요약
- 청크 인덱스

### 4. 유연한 검색
- 유사도 검색
- 메타데이터 필터링
- k 개수 조정 가능

### 5. 확장 가능
- 새로운 책 추가 용이
- 다른 임베딩 모델로 교체 가능
- LLM 연동 가능 (HuggingFace API)

## 🛠️ 트러블슈팅

### 문제: "벡터 스토어를 찾을 수 없습니다"

**해결:** 먼저 벡터DB를 구축하세요.

```bash
python main.py --mode build
```

### 문제: UTF-8 디코딩 에러

**해결:** `preprocess.py`가 자동으로 cp949 인코딩도 시도합니다. 
파일이 다른 인코딩이라면 UTF-8로 변환하세요.

### 문제: 메모리 부족

**해결:** 청크 크기를 줄이거나 한 번에 처리할 책 수를 제한하세요.

```python
# chapter_splitter.py의 main() 함수에서
for file_info in processed_files[:3]:  # 처음 3권만
```

### 문제: LangGraph ImportError

**해결:** LangGraph는 선택사항입니다. 설치 없이도 기본 RAG 사용 가능.

```bash
# LangGraph 설치 (선택)
pip install langgraph
```

## 📚 패키지 요구사항

주요 패키지:
- `langchain`: RAG 파이프라인
- `langchain-community`: 커뮤니티 통합
- `chromadb`: 벡터 데이터베이스
- `sentence-transformers`: 임베딩
- `huggingface-hub`: HuggingFace 모델
- `langgraph`: 상태 기반 워크플로우 (선택)

전체 목록은 `requirements.txt` 참조

## 🔮 향후 개선 사항

- [ ] HuggingFace LLM 연동 (답변 생성)
- [ ] Transformer 기반 NER 모델 추가
- [ ] BM25 + Dense Hybrid 검색
- [ ] 웹 UI (Streamlit/Gradio)
- [ ] 대화 히스토리 관리
- [ ] 멀티턴 대화 지원

## 📄 라이선스

이 프로젝트는 교육 목적으로 제작되었습니다.

## 👥 기여

버그 리포트나 기능 제안은 이슈로 등록해주세요.

---

**제작:** SKNETWORKS-FAMILY-AICAMP 5기 3차 프로젝트 5팀  
**날짜:** 2025년 12월
