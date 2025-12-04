# 해리포터 RAG QA 시스템 - 사용 가이드

## 빠른 시작 (Quick Start)

### 1단계: 환경 설정
```bash
# 프로젝트 디렉토리로 이동
cd c:\Users\ansck\Desktop\Project\3rd_project

# Python 환경 활성화 (이미 설정되어 있음)
conda activate 3rd_project

# 추가 패키지 설치 (필요시)
pip install langchain langchain-community chromadb sentence-transformers
```

### 2단계: 벡터DB 구축 (최초 1회)
```bash
python main.py --mode build
```

예상 소요 시간: 5-10분 (7권 기준)

출력 예시:
```
================================================================================
해리포터 RAG QA 챗봇 시스템
================================================================================
데이터 디렉토리: c:\Users\ansck\Desktop\Project\3rd_project\data\cleaned_data
벡터DB 경로: ./chroma_hp
...
[완료] 총 2,847개 청크 메타데이터 태깅 완료
================================================================================
```

### 3단계: 질의응답 시작
```bash
python main.py --mode interactive
```

질문 예시:
- 해리 포터가 호그와트에 처음 도착했을 때 무슨 일이 있었나요?
- 덤블도어는 어떤 사람인가요?
- 해리의 친구들은 누구인가요?
- 호그와트의 기숙사는 어떤 것들이 있나요?
- 마법사의 돌은 무엇인가요?

종료: `quit` 또는 `q` 입력

## 상세 사용법

### 모드별 실행

#### 1. build 모드 (벡터DB 구축)
```bash
# 기본 구축
python main.py --mode build

# 강제 재구축 (기존 DB 삭제)
python main.py --mode build --force-rebuild

# 커스텀 경로
python main.py --mode build --data-dir "./data/cleaned_data" --persist-dir "./my_chroma"
```

#### 2. query 모드 (미리 정의된 질문)
```bash
python main.py --mode query
```

5개의 기본 질문에 대해 자동으로 답변합니다:
1. 해리 포터가 호그와트에 처음 도착했을 때
2. 덤블도어는 어떤 사람
3. 해리의 가장 친한 친구들
4. 볼드모트는 누구
5. 호그와트의 기숙사

#### 3. interactive 모드 (대화형)
```bash
# 기본 대화형 모드
python main.py --mode interactive

# LangGraph 사용 (설치 필요: pip install langgraph)
python main.py --mode interactive --use-langgraph
```

#### 4. test 모드 (검색 테스트)
```bash
# 기본 쿼리
python main.py --mode test

# 커스텀 쿼리
python main.py --mode test --test-query "해리가 덤블도어를 처음 만났을 때"
```

## 개별 모듈 테스트

### 1. 전처리 테스트
```bash
python preprocess.py
```

**확인 사항:**
- 7개 TXT 파일 로드 성공
- 각 파일의 글자 수, 줄 수 출력
- UTF-8 디코딩 성공

### 2. 장 분리 테스트
```bash
python chapter_splitter.py
```

**확인 사항:**
- 장 자동 감지 (예: 제1장, 제2장...)
- 청크 생성 (chunk_size=800, overlap=150)
- Document 객체 생성

예상 출력:
```
[INFO] 감지된 장 수: 17
  - 1. 제1장: 살아남은 아이
  - 2. 제2장: 사라진 유리
  ...
[INFO] 총 청크 수: 415
```

### 3. 메타데이터 태깅 테스트
```bash
python metadata_tagger.py
```

**확인 사항:**
- 인물 추출 (해리, 론, 헤르미온느...)
- 장소 추출 (호그와트, 다이애건 앨리...)
- 감정 분석 (positive/negative/neutral)
- 키워드 추출

예상 출력:
```
[메타데이터 통계]
[인물 Top 10]
  - 해리: 287회
  - 론: 145회
  - 헤르미온느: 132회
  ...
```

### 4. 임베딩 구축 테스트
```bash
python embedding_build.py
```

**확인 사항:**
- 임베딩 모델 로드 (jhgan/ko-sroberta-multitask)
- ChromaDB 생성
- 검색 테스트 성공

### 5. RAG 파이프라인 테스트
```bash
python rag_pipeline.py
```

**확인 사항:**
- 리트리버 생성
- 질문 → 검색 → 결과 반환

### 6. LangGraph 테스트
```bash
python app_langgraph.py
```

**확인 사항:**
- 그래프 구성 (rewrite → retrieve → rerank → generate → output)
- invoke() / stream() 동작

## 파일 추가 방법

### 새로운 해리포터 책 추가

1. TXT 파일을 `data/cleaned_data/` 디렉토리에 복사
   ```
   data/cleaned_data/cleaned_해리포터와_새책.txt
   ```

2. 벡터DB 재구축
   ```bash
   python main.py --mode build --force-rebuild
   ```

3. 시스템 자동으로 새 책 인식 및 처리

### 다른 시리즈 책 추가

1. 새 데이터 디렉토리 생성
   ```
   data/나니아연대기/
   ```

2. 커스텀 경로로 구축
   ```bash
   python main.py --mode build \
       --data-dir "./data/나니아연대기" \
       --persist-dir "./chroma_narnia"
   ```

## 커스터마이징 가이드

### 1. 청크 크기 조정

`main.py` 수정:
```python
system = HarryPotterRAGSystem(
    chunk_size=500,      # 기본 800 → 500으로 변경
    chunk_overlap=100    # 기본 150 → 100으로 변경
)
```

**효과:**
- 작은 청크: 더 정확한 검색, 많은 청크 수
- 큰 청크: 더 많은 컨텍스트, 적은 청크 수

### 2. 검색 결과 수 조정

`embedding_build.py` 또는 `main.py`:
```python
retriever = builder.create_retriever(
    search_type="similarity",
    k=3  # 기본 4 → 3으로 변경
)
```

### 3. 임베딩 모델 변경

`main.py`:
```python
system = HarryPotterRAGSystem(
    embedding_model="sentence-transformers/all-mpnet-base-v2"  # 영어용
)
```

추천 모델:
- `jhgan/ko-sroberta-multitask`: 한국어 최적 (기본값)
- `sentence-transformers/all-mpnet-base-v2`: 영어
- `intfloat/multilingual-e5-large`: 다국어

**주의:** 모델 변경 시 벡터DB 재구축 필요

### 4. 메타데이터 사전 확장

`metadata_tagger.py`의 `__init__` 메서드:
```python
self.known_characters = {
    "해리", "론", "헤르미온느",
    # 새 인물 추가
    "도비", "윙가르디움 레비오사",
    ...
}

self.known_locations = {
    "호그와트", "다이애건 앨리",
    # 새 장소 추가
    "9와 3/4 승강장", "그리몰드 플레이스",
    ...
}
```

### 5. 검색 필터 사용

특정 책만 검색:
```python
retriever = builder.create_retriever(
    search_type="similarity",
    k=4,
    filter_dict={"book": "해리포터와 마법사의 돌"}
)
```

특정 감정만 검색:
```python
filter_dict={"sentiment": "positive"}
```

## 고급 기능

### HuggingFace LLM 연동 (옵션)

1. HuggingFace 토큰 발급
   - https://huggingface.co/settings/tokens

2. 환경 변수 설정
   ```bash
   # Windows PowerShell
   $env:HUGGINGFACEHUB_API_TOKEN="your_token_here"
   
   # 또는 .env 파일 생성
   echo HUGGINGFACEHUB_API_TOKEN=your_token_here > .env
   ```

3. LLM 사용
   ```python
   from rag_pipeline import RAGPipeline
   
   rag = RAGPipeline(
       retriever=retriever,
       huggingface_token="your_token",
       model_name="mistralai/Mistral-7B-Instruct-v0.2"
   )
   ```

### LangGraph 워크플로우

```bash
pip install langgraph

python main.py --mode interactive --use-langgraph
```

LangGraph는 다음 노드를 순차 실행:
1. **rewrite_node**: 질문 재작성
2. **retrieve_node**: 문서 검색
3. **rerank_node**: 재순위 (상위 3개)
4. **generate_node**: 답변 생성
5. **output_node**: 최종 출력

## 성능 최적화

### 메모리 사용량 줄이기

1. **배치 처리**: 한 번에 1-2권씩 처리
   ```python
   # chapter_splitter.py
   for file_info in processed_files[:2]:  # 2권씩
   ```

2. **청크 크기 증가**: 청크 수 감소
   ```python
   chunk_size=1000  # 800 → 1000
   ```

### 검색 속도 향상

1. **k 값 감소**: 검색 문서 수 줄이기
   ```python
   k=3  # 4 → 3
   ```

2. **인덱스 최적화**: ChromaDB 설정
   ```python
   # embedding_build.py
   collection_metadata={"hnsw:space": "cosine"}
   ```

## 문제 해결

### Q: "벡터 스토어를 찾을 수 없습니다"
**A:** 
```bash
python main.py --mode build
```

### Q: UTF-8 디코딩 에러
**A:** 파일을 UTF-8로 변환하거나, `preprocess.py`가 자동으로 cp949도 시도합니다.

### Q: 메모리 부족
**A:** 청크 크기 증가 또는 배치 처리

### Q: 검색 결과가 부정확
**A:** 
- k 값 증가 (더 많은 문서 검색)
- chunk_size 조정
- 임베딩 모델 변경

### Q: LangGraph ImportError
**A:** 
```bash
pip install langgraph
```
또는 LangGraph 없이 기본 RAG 사용

## 출력 예시

### 벡터DB 구축
```
================================================================================
[전체 파이프라인 실행]
================================================================================

[STEP 1] TXT 파일 로드 및 전처리
--------------------------------------------------------------------------------
[SUCCESS] 파일 로드 완료: cleaned_해리포터와 마법사의 돌.txt (314,523 글자)
...
[완료] 총 7개 파일 전처리 완료

[STEP 2] 장 분리 및 청킹
--------------------------------------------------------------------------------
[처리 중] 해리포터와 마법사의 돌
[INFO] 감지된 장 수: 17
[완료] 해리포터와 마법사의 돌: 총 17개 장, 415개 청크

[STEP 3] 메타데이터 자동 태깅
--------------------------------------------------------------------------------
[진행] 415/415 (100.0%) 완료
[완료] 총 415개 청크 메타데이터 태깅 완료

[STEP 4] 임베딩 및 ChromaDB 구축
--------------------------------------------------------------------------------
[INFO] 벡터 스토어 생성 중... (415개 문서)
[SUCCESS] 벡터 스토어 생성 완료

[전체 파이프라인 완료]
✓ 처리된 책: 7권
✓ 생성된 청크: 2,847개
✓ 벡터DB 저장: ./chroma_hp
```

### 대화형 질의응답
```
================================================================================
[대화형 모드]
================================================================================
해리포터에 대해 무엇이든 물어보세요!
종료하려면 'quit', 'exit', 'q'를 입력하세요.
================================================================================

질문: 해리의 친구들은 누구인가요?

================================================================================
[질문] 해리의 친구들은 누구인가요?

[검색 결과] 4개 문서
================================================================================

[1] 해리포터와 마법사의 돌 - 제6장: 9와 3/4 승강장으로 가는 여행
    인물: ['해리', '론', '론 위즐리']
    장소: ['호그와트']
    감정: positive
    내용: 호그와트 급행열차에서 론 위즐리를 만났다. 론은 빨간 머리에...

[2] 해리포터와 마법사의 돌 - 제10장: 할로윈
    인물: ['해리', '론', '헤르미온느', '헤르미온느 그레인저']
    장소: ['호그와트']
    감정: positive
    내용: 트롤 사건 이후 헤르미온느 그레인저가 친구가 되었다...
```

## 다음 단계

1. ✅ 시스템 설치 및 구축 완료
2. ✅ 기본 질의응답 테스트
3. 🔄 커스터마이징 (청크 크기, 모델 등)
4. 🔄 HuggingFace LLM 연동 (선택)
5. 🔄 웹 UI 구축 (Streamlit/Gradio)

## 추가 리소스

- [LangChain 문서](https://python.langchain.com/)
- [ChromaDB 문서](https://docs.trychroma.com/)
- [HuggingFace 모델 허브](https://huggingface.co/models)
- [LangGraph 문서](https://langchain-ai.github.io/langgraph/)

---

**문의사항이 있으시면 이슈를 등록해주세요!**
