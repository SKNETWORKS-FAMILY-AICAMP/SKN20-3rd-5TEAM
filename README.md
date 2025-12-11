<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=wave&height=200&color=0:FFD86F,50:FFAA28,100:FF6B00&text=SKN-3RD-5TEAM&fontSize=47&fontColor=FFFFFF&section=header&desc=RAG%20Pipeline%20for%20Shelter%20and%20Disaster%20Guidelines&descSize=15" />
</p>

<p align="center">
  📌 <strong>민방위 대피시설 + 재난 행동요령 데이터 기반 RAG & 챗봇 시스템</strong><br>
  <sub>SK Networks Family 20기 — 3RD 5TEAM</sub>
</p>

<p align="center">
  🔍 <strong>CSV + JSON → LangChain Document → Embedding → ChromaDB → LangGraph Agent → FastAPI + Web UI</strong>
</p>

---

## 👥 팀구성

| <img src="./image/쿼카.jpeg" width="150"> <br> 문창교 |  <img src="./image/dak.jpeg" width="150"> <br> 권규리 |  <img src="./image/rich.jpeg" width="150"> <br> 김황현 |  <img src="./image/loopy.jpeg" width="150"> <br> 김효빈 |  <img src="./image/ham.jpeg" width="150"> <br> 이승규 |
|:------:|:------:|:------:|:------:|:------:|
| <a href="https://github.com/mck1902"><img src="https://img.shields.io/badge/GitHub-mck1902-blue?logo=github"></a> | <a href="https://github.com/gyur1eek"><img src="https://img.shields.io/badge/GitHub-gyur1eek-yellow?logo=github"></a> | <a href="https://github.com/hyun2kim"><img src="https://img.shields.io/badge/GitHub-khyun2kim-green?logo=github"></a> | <a href="https://github.com/kimobi"><img src="https://img.shields.io/badge/GitHub-kimobi-pink?logo=github"></a> | <a href="https://github.com/neobeauxarts"><img src="https://img.shields.io/badge/GitHub-neobeauxarts-lightblue?logo=github"></a> |

---

## 💻 기술 스택

| 분류 | 사용 기술 |
|------|-----------|
| 언어 | ![Python](https://img.shields.io/badge/Python-3776AB.svg?style=flat&logo=python&logoColor=white) Python |
| 프론트엔드 | ![HTML5](https://img.shields.io/badge/HTML5-E34F26.svg?style=flat&logo=html5&logoColor=white) HTML<br>![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E.svg?style=flat&logo=javascript&logoColor=black) JavaScript<br>![TailwindCSS](https://img.shields.io/badge/Tailwind-06B6D4.svg?style=flat&logo=tailwindcss&logoColor=white) TailwindCSS |
| 백엔드 | ![FastAPI](https://img.shields.io/badge/FastAPI-009688.svg?style=flat&logo=fastapi&logoColor=white) FastAPI |
| AI 프레임워크 | ![LangChain](https://img.shields.io/badge/LangChain-1E8C7E.svg?style=flat&logo=chainlink&logoColor=white) LangChain<br>![LangGraph](https://img.shields.io/badge/LangGraph-1E8C7E.svg?style=flat&logo=chainlink&logoColor=white) LangGraph |
| LLM | ![OpenAI](https://img.shields.io/badge/OpenAI%20GPT--4o--mini-000000.svg?style=flat&logo=openai&logoColor=white) GPT-4o-mini |
| 임베딩 모델 | ![OpenAI](https://img.shields.io/badge/OpenAI%20Embedding-000000.svg?style=flat&logo=openai&logoColor=white) text-embedding-3-small |
| 벡터 DB | ![ChromaDB](https://img.shields.io/badge/ChromaDB-16C47F.svg?style=flat&logo=databricks&logoColor=white) ChromaDB |
| 검색 알고리즘 | Dense Retrieval (Vector Search)<br>BM25 (Sparse Retrieval)<br>Ensemble Hybrid Search |
| 외부 API | Kakao Local API (위치 검색)<br>Naver Maps API (지도 시각화)<br>Naver Panorama API (거리뷰) |
| 데이터 전처리 | ![Pandas](https://img.shields.io/badge/Pandas-150458.svg?style=flat&logo=pandas&logoColor=white) pandas |
| 환경 변수 관리 | ![Dotenv](https://img.shields.io/badge/Dotenv-9ACD32.svg?style=flat&logo=dotenv&logoColor=white) python-dotenv (.env) |

---

## 📌 프로젝트 개요

본 프로젝트는 **민방위 대피 시설(CSV)** 과 **재난 행동요령(JSON)** 데이터를 통합하여  
**LangGraph Agent 기반 RAG(Retrieval-Augmented Generation)** 시스템과 **FastAPI + 웹 인터페이스** 기반 실시간 챗봇을 구축하는 프로젝트입니다.

사용자는 **HTML/JavaScript 기반 웹 인터페이스**에서  
재난 상황에 대한 질문 또는 대피소 위치 관련 질의를 입력하면  
**하이브리드 벡터 검색(Dense + BM25)** 기반으로 **근거 기반 답변**을 제공하고, **Naver 지도에 대피소 위치를 시각화**합니다.

### 🎯 핵심 기능

1. **LangGraph Agent 아키텍처**
   - 의도 분류 (Intent Classification)
   - 질문 재정의 (Query Rewriting)
   - 도구 기반 멀티 턴 대화 (Tool-based Multi-turn)
   - 메모리 기반 대화 컨텍스트 유지

2. **하이브리드 검색 시스템**
   - Dense Retrieval: OpenAI 임베딩 기반 의미 검색
   - BM25: 키워드 기반 정확도 검색
   - Ensemble Weighting: 두 방식의 가중 평균

3. **복합 질문 처리**
   - "설악산 근처인데 산사태 발생 시" → 위치 검색 + 재난 행동요령
   - "동대문맨션 수용인원" → 시설명 직접 검색
   - "강남역 근처 대피소" → GPS 기반 거리 계산

4. **실시간 지도 시각화**
   - Kakao Local API: 지명 → 좌표 변환
   - Naver Maps API: 대피소 마커 표시
   - Naver Panorama API: 거리뷰 제공
   - GPS 기반 현위치 검색

---

## 📁 프로젝트 구조

```
SKN20-3RD-5TEAM/
├── data/                           # 데이터 디렉토리
│   ├── shelter.csv                 # 민방위 대피시설 데이터
│   ├── natural_disaster/*.json     # 자연재난 행동요령
│   └── social_disaster/*.json      # 사회재난 행동요령
│
├── chroma_db/                      # ChromaDB 영구 저장소
│
├── shelter_chatbot/
│   └── cert/                       # SSL 인증서 디렉토리
│       ├── cert.pem
│       └── key.pem
│
├── data_loaders.py                 # 데이터 로드 모듈
├── documents.py                    # Document 변환 모듈
├── embedding_and_vectordb.py       # 임베딩 및 VectorDB 생성
├── from_DataLoad_to_VectorDB.py    # 전체 파이프라인 실행 (초기화)
├── langgraph_agent.py              # LangGraph Agent 및 Tools 정의
├── main.py                         # FastAPI 서버
├── shelter_1.0.html                # 웹 프론트엔드
└── generate_ssl_cert.py            # SSL 인증서 자동 생성
```

---

## 🚀 설치 & 실행

### 0️⃣ 패키지 설치

```bash
pip install -r requirements.txt
```

**주요 패키지:**

| 카테고리 | 패키지 | 역할 |
|---------|--------|------|
| **웹 프레임워크** | `fastapi` | REST API 서버 구축 |
| | `uvicorn` | ASGI 서버 실행 |
| | `pydantic` | 데이터 검증 및 직렬화 |
| **AI/LLM** | `langchain` | LLM 프레임워크 핵심 |
| | `langchain-openai` | OpenAI API 통합 |
| | `langchain-community` | 커뮤니티 도구 (BM25 등) |
| | `langchain-chroma` | ChromaDB 연동 |
| | `langchain-core` | LangChain 핵심 모듈 |
| | `langgraph` | Agent 상태 기반 그래프 |
| | `openai` | OpenAI Python SDK |
| **벡터 DB** | `chromadb` | 벡터 데이터베이스 |
| **데이터 처리** | `pandas` | 테이블 데이터 처리 |
| | `numpy` | 수치 연산 |
| **네트워크** | `requests` | HTTP 요청 (Kakao API) |
| | `httpx` | 비동기 HTTP 클라이언트 |
| **환경 관리** | `python-dotenv` | .env 파일 로드 |
| **보안** | `cryptography` | SSL 인증서 생성 |
---

### 1️⃣ 환경변수 설정 (.env)
```bash
OPENAI_API_KEY=your_openai_api_key_here
KAKAO_REST_API_KEY=your_kakao_rest_api_key_here
```

---

### 2️⃣ VectorDB 초기화 (최초 1회만 실행)

#### 📄 `from_DataLoad_to_VectorDB.py` (데이터 파이프라인 통합 실행)

**목적:** 원본 데이터를 VectorDB로 변환하는 전체 파이프라인을 한 번에 실행

**실행:**
```bash
python from_DataLoad_to_VectorDB.py
```

**내부 동작 순서:**

1. **`data_loaders.py` 호출 - 데이터 로드**
   - `load_shelter_csv(csv_file, data_dir)`: CSV 파일을 pandas DataFrame으로 로드
   - `load_disaster_json(path)`: 단일 JSON 파일을 dict로 로드
   - `load_all_disaster_jsons(json_files, data_dir)`: 여러 JSON 파일을 일괄 로드

2. **`documents.py` 호출 - Document 변환**
   - `csv_to_documents(shelter_data)`: 
     - 대피소 DataFrame을 LangChain Document 리스트로 변환
     - 각 행을 자연어 텍스트(`page_content`)로 변환
     - 메타데이터 추출 (시설명, 주소, 좌표, 수용인원 등)
   
   - `json_to_documents(disaster_datas)`:
     - 재난 행동요령 JSON을 LangChain Document 리스트로 변환
     - `parse_node()` 함수로 계층적 JSON 구조를 재귀 탐색
     - 세부사항, 주의사항, 행동요령 등을 포함한 Document 생성

3. **`embedding_and_vectordb.py` 호출 - 임베딩 및 저장**
   - `create_embeddings_and_vectordb(documents)`:
     - OpenAI `text-embedding-3-small` 모델로 임베딩 생성
     - ChromaDB에 Document + 임베딩 저장
     - 컬렉션명: `shelter_and_disaster_guidelines`
     - 저장 경로: `./chroma_db/`

**데이터 변환 과정:**
```
[원본 데이터]
├── shelter.csv (DataFrame)
│   └── csv_to_documents()
│       └── List[Document(page_content, metadata)] ---- ['shelter' Documents]
│
└── *.json (dict)
    └── json_to_documents()
        └── List[Document(page_content, metadata)] ---- ['guidline' Documents]

↓ (통합)

['shelter' Documents] + ['guidline' Documents] → [All Documents]

↓ (임베딩)

[OpenAI Embeddings] → Vector[1536차원]

↓ (저장)

[ChromaDB] → ./chroma_db/
```

---

### 3️⃣ FastAPI 서버 실행

#### 📄 `main.py` (API 서버 및 요청 처리)

**목적:** 사용자 요청을 받아 LangGraph Agent에 전달하고 결과를 반환하는 FastAPI 서버

**실행:**
```bash
python main.py
```

**핵심 구성요소:**

1. **Lifespan 초기화 (`@asynccontextmanager async def lifespan`)**
   - 서버 시작 시 실행:
     - OpenAI 임베딩 모델 초기화
     - ChromaDB 로드 (`./chroma_db/`)
     - 대피소 CSV 데이터 로드
     - `langgraph_agent.py`의 `create_langgraph_app()` 호출하여 Agent 초기화

2. **주요 API 엔드포인트:**

   - `POST /api/location/extract`:
     - 사용자 쿼리를 받아 LangGraph Agent 실행
     - Agent의 `structured_data` 추출 (지도 표시용)
     - 응답 반환 (`LocationExtractResponse`)
   
   - `GET /api/shelters/nearest`:
     - GPS 좌표 기반 가장 가까운 대피소 검색
     - Haversine 공식으로 거리 계산
     - ChromaDB 메타데이터 필터링
   
   - `GET /`: 
     - `shelter_1.0.html` 파일 제공 (웹 UI)
   
   - `GET /api/health`, `GET /api/status`:
     - 서버 상태 확인

3. **CORS 미들웨어:**
   - 모든 도메인에서의 API 호출 허용

---

#### 📄 `langgraph_agent.py` (Agent 로직 및 도구 정의)

**목적:** 순수하게 "생각하는 로직"만 담당 - 의도 분류, 질문 재정의, 도구 선택 및 실행

**핵심 구성요소:**

1. **`create_hybrid_retrievers(vectorstore)` - 하이브리드 검색기 생성**
   - Vector Retriever: 의미 기반 검색 (OpenAI 임베딩)
   - BM25 Retriever: 키워드 기반 검색
   - `EnsembleRetriever`: 두 방식을 가중 평균으로 결합 (Vector 60-70%, BM25 30-40%)

2. **`create_langgraph_app(vectorstore)` - LangGraph Agent 생성**

   **LLM 초기화:**
   - `llm`: GPT-4o-mini (temperature=0, 정확한 분류용)
   - `llm_creative`: GPT-4o-mini (temperature=0.7, 일반 지식 답변용)

   **Chain 정의:**
   - `intent_classification_prompt + llm`: 사용자 질문을 8개 카테고리로 분류
     - `hybrid_location_disaster`, `shelter_info`, `shelter_search`, `shelter_count`, 
     - `shelter_capacity`, `disaster_guideline`, `general_knowledge`, `general_chat`
   
   - `query_rewrite_prompt + llm`: BM25 검색 최적화를 위한 쿼리 재작성
     - 핵심 키워드 추출, 동의어 추가, 지역명 확장

   **Tools (7가지 도구) 정의:**

   - **`@tool search_shelter_by_location(query)`**
     - Kakao Local API로 지명 → 좌표 변환
     - ChromaDB에서 모든 대피소 메타데이터 가져오기
     - Haversine 공식으로 거리 계산
     - 가까운 5곳 반환 + `structured_data` (지도 표시용)

   - **`@tool search_shelter_by_name(query)`** 
     - 시설명 부분 일치 검색
     - 메타데이터에서 `facility_name` 필터링
     - 수용인원, 주소 등 상세 정보 반환

   - **`@tool search_location_with_disaster(query)`** 
     - 복합 질문 처리 (위치 + 재난)
     - 재난 키워드 추출 → 위치 검색 → 대피소 검색 + 행동요령 검색
     - 통합 결과 반환

   - **`@tool count_shelters(query)`**
     - 조건에 맞는 대피소 개수 세기
     - Hybrid Retriever 사용

   - **`@tool search_shelter_by_capacity(query)`**
     - 수용인원 기준 검색
     - 정규표현식으로 숫자 추출
     - 메타데이터 필터링

   - **`@tool search_disaster_guideline(query)`**
     - 재난 행동요령 검색
     - Hybrid Retriever 사용
     - 상위 3개 결과 통합

   - **`@tool answer_general_knowledge(query)`**
     - 일반 지식 질문 답변
     - LLM 사전 학습 지식 활용 (VectorDB 미사용)

   **State 정의:**
   ```python
   class AgentState(TypedDict):
       messages: List[BaseMessage]
       intent: str
       rewritten_query: str
       structured_data: Optional[dict]  # 지도 표시용
   ```

   **Node 함수:**
   - `intent_classifier_node`: 의도 분류
   - `query_rewrite_node`: 질문 재정의
   - `agent_node`: 도구 선택 및 실행 계획
   - `tools_node_with_structured_data`: 도구 실행 및 구조화된 데이터 추출

   **Graph 구조:**
   ```
   START → intent_classifier → query_rewrite → agent ⇄ tools → END
   ```

   **Memory:**
   - `MemorySaver`: 대화 컨텍스트 유지 (세션 ID 기반)

---

#### 📄 `shelter_1.0.html` (프론트엔드 - 사용자 인터페이스)

**목적:** 사용자가 질문을 입력하고 결과를 시각적으로 확인하는 웹 페이지

**핵심 기능:**

1. **API 호출:**
   - `POST /api/location/extract`: 사용자 쿼리 전송
   - `GET /api/shelters/nearest`: GPS 좌표 기반 검색

2. **Naver Maps API 연동:**
   - 대피소 마커 표시
   - 사용자 위치 마커 표시
   - InfoWindow로 상세 정보 표시

3. **Naver Panorama API 연동:** 
   - 지도 클릭 시 해당 위치의 거리뷰 표시
   - 지도 영역과 파노라마 영역을 50%/50%로 분할
   - 마커 클릭 시 해당 위치의 파노라마 자동 표시

4. **GPS 위치 정보:**
   - `navigator.geolocation.getCurrentPosition()` 사용
   - 실시간 현위치 검색
   - 페이지 로드 시 자동으로 현위치 표시 

5. **현위치 기반 기능 강화:** 
   - `currentUserPosition` 전역 변수로 현위치 저장
   - `resetMapToCurrentLocation()` 함수로 지도 정보 없을 때 현위치로 리셋
   - 대피소 정보 없는 응답 시 자동으로 현위치 지도 표시

---

### 4️⃣ SSL 인증서 생성 (선택사항 - HTTPS 필요 시)

#### 📄 `generate_ssl_cert.py` (SSL 인증서 자동 생성)

**목적:** HTTPS 서버 구동을 위한 자가 서명 인증서 생성

**실행:**
```bash
python generate_ssl_cert.py
```

**주요 기능:**
- `cryptography` 라이브러리로 RSA 4096-bit 개인키 생성
- X.509 인증서 생성 (유효기간 365일)
- Subject Alternative Name (SAN) 설정
- `./shelter_chatbot/cert/` 디렉토리에 저장

---

## 🔄 복합 질문 처리 과정 (예시)

### 질문: "지금 내가 플레이데이터 서초캠퍼스에 있는데 땅이 흔들리고 있어. 지금 내가 어떻게 행동해야하는지 또 어디로 대피해야하는지 알려줘"

#### 🎬 전체 처리 흐름 [[📊 인터랙티브 플로우차트 보기](./RAG%20챗봇%20처리%20과정.html)]

```
[1단계: 프론트엔드 (shelter_1.0.html)]
│
├── 사용자가 텍스트 입력
├── JavaScript: fetch(`POST /api/location/extract`, {query: "..."})
│
↓
│
[2단계: FastAPI 서버 (main.py)]
│
├── `@app.post("/api/location/extract")` 엔드포인트 실행
├── Request Body 파싱: `LocationExtractRequest(query="...")`
├── LangGraph Agent 호출:
│   └── langgraph_app.invoke({"messages": [HumanMessage(content=query)]})
│
↓
│
[3단계: LangGraph Agent (langgraph_agent.py)]
│
├── 🧠 Node 1: intent_classifier_node
│   ├── `intent_classification_prompt + llm` 실행
│   ├── 질문 분석:
│   │   - "플레이데이터 서초캠퍼스" → 위치 키워드 감지
│   │   - "땅이 흔들리고" → 지진 키워드 감지
│   │   - "행동해야하는지" + "대피해야하는지" → 복합 의도 감지
│   └── 의도 분류 결과: "hybrid_location_disaster" (위치 + 재난)
│
├── 🔄 Node 2: query_rewrite_node
│   ├── `query_rewrite_prompt + llm` 실행
│   ├── BM25 최적화를 위한 쿼리 재작성:
│   │   └── "플레이데이터 서초캠퍼스 지진 대피소 행동요령"
│   │       (불필요한 조사 제거, 핵심 키워드 추출)
│
├── 🤖 Node 3: agent_node
│   ├── 시스템 프롬프트 + 사용자 메시지 결합
│   ├── LLM이 적절한 도구 선택:
│   │   └── `search_location_with_disaster(query)` 선택
│   │       (이유: 위치 + 재난이 함께 있는 복합 질문)
│
├── 🛠️ Node 4: tools_node (search_location_with_disaster 실행)
│   │
│   ├── [4-1] 재난 키워드 추출
│   │   ├── "땅이 흔들리고" → "지진" 감지
│   │   └── detected_disaster = "지진"
│   │
│   ├── [4-2] 위치 키워드 추출
│   │   ├── 질문에서 재난 키워드 제거: "플레이데이터 서초캠퍼스"
│   │   └── location_query = "플레이데이터 서초캠퍼스"
│   │
│   ├── [4-3] Kakao Local API 호출
│   │   ├── GET https://dapi.kakao.com/v2/local/search/keyword.json
│   │   ├── params: {"query": "플레이데이터 서초캠퍼스"}
│   │   └── 응답:
│   │       ├── place_name: "플레이데이터"
│   │       ├── y (위도): 37.4979
│   │       └── x (경도): 127.0276
│   │
│   ├── [4-4] 대피소 검색 (ChromaDB)
│   │   ├── vectorstore.get(where={"type": "shelter"})
│   │   ├── 모든 대피소 메타데이터 가져오기
│   │   ├── Haversine 거리 계산:
│   │   │   └── distance = haversine(37.4979, 127.0276, shelter.lat, shelter.lon)
│   │   ├── 거리순 정렬
│   │   └── 가장 가까운 3곳 선택:
│   │       ├── 1. 서초구민회관 (0.5km)
│   │       ├── 2. 강남역지하상가 (1.2km)
│   │       └── 3. 교보타워 (1.5km)
│   │
│   ├── [4-5] 재난 행동요령 검색 (Hybrid Retriever)
│   │   ├── guideline_hybrid_retriever.invoke("지진")
│   │   ├── Vector Search (60%):
│   │   │   ├── 쿼리 임베딩: "지진" → Vector[1536]
│   │   │   └── 유사도 계산 → 관련 Document 반환
│   │   ├── BM25 Search (40%):
│   │   │   ├── 토큰화: ["지진"]
│   │   │   └── 키워드 매칭 → 관련 Document 반환
│   │   └── Ensemble 결과 (상위 2개):
│   │       ├── Document 1: "지진 > 발생 시 > 실내\n- 책상 아래로 대피\n- 문 열기\n..."
│   │       └── Document 2: "지진 > 발생 직후\n- 엘리베이터 사용 금지\n..."
│   │
│   └── [4-6] 통합 결과 생성
│       └── return {
│           "text": """
│               🚨 플레이데이터 근처 지진 발생 시 대응 가이드
│               
│               📍 가장 가까운 대피소 3곳
│               1. 서초구민회관 (0.5km)
│               2. 강남역지하상가 (1.2km)
│               3. 교보타워 (1.5km)
│               
│               🚨 지진 행동요령
│               - 책상 아래로 대피하세요
│               - 문을 열어 출구를 확보하세요
│               - 엘리베이터 사용을 금지합니다
│               ...
│               """,
│           "structured_data": {
│               "location": "플레이데이터",
│               "coordinates": (37.4979, 127.0276),
│               "shelters": [
│                   {"name": "서초구민회관", "lat": 37.5000, "lon": 127.0300, "distance": 0.5, ...},
│                   ...
│               ]
│           }
│       }
│
↓
│
[4단계: FastAPI 서버 (main.py) - 응답 반환]
│
├── Agent 결과에서 `structured_data` 추출
├── `LocationExtractResponse` 생성:
│   ├── success: True
│   ├── location: "플레이데이터"
│   ├── coordinates: (37.4979, 127.0276)
│   ├── shelters: [...]
│   └── message: "🚨 플레이데이터 근처 지진 발생 시 대응 가이드\n..."
│
└── JSON 응답 반환
│
↓
│
[5단계: 프론트엔드 (shelter_1.0.html) - UI 업데이트]
│
├── JavaScript: response.json() 파싱
├── 채팅창에 메시지 출력:
│   └── addMessage("bot", response.message)
│
└── Naver Maps API 호출:
    ├── map.setCenter(new naver.maps.LatLng(37.4979, 127.0276))
    ├── 사용자 위치 마커 생성 (파란색)
    └── 대피소 마커 생성 (빨간색 - 가장 가까운 곳)
        ├── 서초구민회관
        ├── 강남역지하상가
        └── 교보타워
```
---

## 📊 데이터 변환 과정 상세

### CSV 데이터 변환 흐름

```
[원본: shelter.csv]
├── 컬럼: 시설명, 도로명전체주소, 최대수용인원, 위도, 경도, ...
│
↓ load_shelter_csv()
│
[pandas DataFrame]
├── 각 행이 대피소 1곳
│
↓ csv_to_documents()
│
[LangChain Document]
├── page_content: "민방위 대피 시설 XX은 YY에 위치해 있으며, ..."
├── metadata:
│   ├── type: "shelter"
│   ├── facility_name: "동대문맨션"
│   ├── address: "서울특별시 중구 ..."
│   ├── lat: 37.5665
│   ├── lon: 126.9780
│   ├── capacity: 1500
│   └── ...
│
↓ OpenAI Embeddings
│
[Vector: 1536차원 부동소수점 배열]
│
↓ ChromaDB.add()
│
[ChromaDB 저장]
└── 인덱스 생성, 검색 가능 상태
```

### JSON 데이터 변환 흐름

```
[원본: *.json]
├── 계층 구조:
│   ├── 재난명
│   ├── 행동요령
│   │   ├── 발생 전
│   │   │   ├── 세부사항: [...]
│   │   │   └── 주의사항: [...]
│   │   ├── 발생 시
│   │   └── 발생 후
│
↓ load_disaster_json()
│
[Python dict]
│
↓ json_to_documents() + parse_node() (재귀 탐색)
│
[LangChain Document]
├── page_content: "지진 > 발생 시 > 실내\n세부사항:\n- 책상 아래로 대피\n..."
├── metadata:
│   ├── type: "disaster_guideline"
│   ├── category: "자연재난"
│   ├── keyword: "지진"
│   ├── situation: "발생 시"
│   └── path: "지진 > 발생 시 > 실내"
│
↓ OpenAI Embeddings
│
[Vector: 1536차원 부동소수점 배열]
│
↓ ChromaDB.add()
│
[ChromaDB 저장]
└── 의미 기반 검색 가능
```

---

### 📝 각 파일의 역할 요약

| 파일 | 역할 | 비유 |
|------|------|------|
| `data_loaders.py` | 원본 데이터 읽기 | 📂 창고에서 재료 꺼내기 |
| `documents.py` | 데이터를 Document로 변환 | 🍳 재료를 요리 재료로 손질 |
| `embedding_and_vectordb.py` | 임베딩 생성 및 저장 | 📦 요리를 포장해서 냉장고에 보관 |
| `from_DataLoad_to_VectorDB.py` | 위 3개를 통합 실행 | 🚀 한 번에 준비 완료 |
| `langgraph_agent.py` | 생각하는 로직 (두뇌) | 🧠 셰프의 레시피와 조리 기술 |
| `main.py` | API 서버 (얼굴) | 👨‍🍳 손님 응대 및 주문 받기 |
| `shelter_1.0.html` | 웹 UI (손님) | 🍽️ 레스토랑 테이블 |

**분리 이유:**
- **로직(langgraph_agent.py)**과 **UI(main.py)**를 분리해야:
  - 디버깅이 쉬움 (로직 오류 vs UI 오류 구분)
  - 확장이 편함 (카카오톡 봇, 텔레그램 봇 등으로 UI 교체 가능)
  - 재사용 가능 (같은 Agent를 다른 프로젝트에서 사용)

---

## ⚙ 시스템 아키텍처

### LangGraph Agent 구조

```python
[사용자 쿼리]
    ↓
[Intent Classifier Node]  # GPT-4o-mini 기반 의도 분류
    ↓
[Query Rewrite Node]  # BM25 최적화를 위한 쿼리 재작성
    ↓
[Agent Node]  # 도구 선택 및 실행 계획
    ↓
[Tools Execution]  # 7가지 도구 중 선택
    ├─ search_shelter_by_location
    ├─ search_shelter_by_name
    ├─ search_location_with_disaster
    ├─ count_shelters
    ├─ search_shelter_by_capacity
    ├─ search_disaster_guideline
    └─ answer_general_knowledge
    ↓
[Response Generation]  # 구조화된 데이터 반환
    ↓
[FastAPI Response]  # JSON 형식
    ↓
[HTML UI + Naver Maps]  # 시각화
```

### 하이브리드 검색 구조

```python
[사용자 쿼리]
    ↓
[Ensemble Retriever]
    ├─ [Vector Retriever] (Weight: 0.6~0.7)
    │   └─ OpenAI Embeddings + ChromaDB
    │
    └─ [BM25 Retriever] (Weight: 0.3~0.4)
        └─ 토큰 기반 키워드 매칭
    ↓
[결과 통합 및 중복 제거]
    ↓
[Top-K 문서 반환]
```

---

## 🔧 API 엔드포인트

### 1. POST `/api/location/extract`
**사용자 질의를 LangGraph Agent가 처리**

**Request:**
```json
{
  "query": "강남역 근처 대피소"
}
```

**Response:**
```json
{
  "success": true,
  "location": "강남역",
  "coordinates": [37.4979, 127.0276],
  "shelters": [
    {
      "name": "강남역지하상가",
      "address": "서울 강남구 강남대로 지하396",
      "lat": 37.4979,
      "lon": 127.0276,
      "distance": 0.12,
      "capacity": 5000
    }
  ],
  "total_count": 5,
  "message": "강남역 근처 대피소 5곳을 찾았습니다."
}
```

### 2. GET `/api/shelters/nearest`
**현위치 기준 가장 가까운 대피소 검색**

**Query Parameters:**
- `lat`: 위도 (float)
- `lon`: 경도 (float)
- `k`: 반환 개수 (int, default=5)

### 3. GET `/api/health`
**서버 상태 확인**

### 4. GET `/api/status`
**상세 시스템 상태 확인**

---


## 🔗 웹 챗봇 서비스

| 서비스 | 접속 링크 |
|--------|-----------|
| 민방위 대피시설 · 재난 행동요령 실시간 질의응답 챗봇 | 🔗 **https://183.98.34.111:8443/** |


## 🎯 제공 기능

### 1. 위치 기반 대피소 검색
- **GPS 현위치 검색**: "현위치" 또는 GPS 버튼 클릭
- **지명 검색**: "강남역 근처 대피소"
- **주소 검색**: "서울시 중구 명동"
- **건물명 검색**: "롯데월드 주변"

### 2. 시설명 직접 검색
- "동대문맨션 수용인원"
- "서울역 대피소 정보"

### 3. 복합 질문 처리 (위치 + 재난)
- "설악산 근처인데 산사태 발생 시"
- "강남역에서 지진 나면 어떻게 해야 해?"
- "명동 화재 났을 때 대피소"

### 4. 재난 행동요령
- "지진 발생 시 행동요령"
- "화재 신고 방법"
- "홍수 대비 방법"

### 5. 통계 및 분석
- "서울 지하 대피소 몇 개?"
- "1000명 이상 수용 가능한 대피소"

### 6. 실시간 지도 시각화
- Naver Maps API를 통한 대피소 마커 표시
- 사용자 위치 표시
- 거리 기반 정렬 (가장 가까운 대피소 강조)
- InfoWindow를 통한 상세 정보 표시

---

## 📊 데이터 구조

### 대피소 데이터 (CSV)
```python
{
    "type": "shelter",
    "facility_name": "동대문맨션",
    "address": "서울특별시 중구 을지로 281",
    "lat": 37.5665,
    "lon": 126.9780,
    "capacity": 1500,
    "shelter_type": "지하",
    "facility_type": "민방위"
}
```

### 재난 행동요령 (JSON)
```python
{
    "type": "disaster_guideline",
    "category": "자연재난",
    "keyword": "지진",
    "situation": "발생 직후",
    "path": "지진 > 발생 직후 > 실내",
    "세부사항": ["책상 아래로 대피", "문 열기", ...],
    "주의사항": ["엘리베이터 사용 금지", ...]
}
```

---

## 📦 산출물 정리

| 요구 산출물 | 제공 항목 |
|------------|-----------|
| 수집된 데이터 및 전처리 | `data/` 폴더 및 Document 변환 모듈 |
| 시스템 아키텍처 | LangGraph Agent + Hybrid Search 다이어그램 |
| 개발된 소프트웨어 | Python 스크립트 8종 + HTML 웹 UI |
| 배포 환경 | FastAPI HTTPS 서버 + 외부 접속 URL |
| VectorDB | ChromaDB 영구 저장소 |
| API 문서 | FastAPI 자동 생성 (Swagger UI) |

---

## 🌱 향후 개선 계획

- [ ] 사용자 위치 기반 대피소 추천 강화
- [ ] 카테고리별 대피소 필터링 (장애인, 유아, 반려동물 등)
- [ ] Retrieval Routing (의도별 검색 전략 최적화)
- [ ] RAG 평가 지표 적용 (RAGAS)
- [ ] 웹 검색 API 통합 (Tavily)
- [ ] 실시간 재난 알림 시스템
- [ ] 다국어 지원 (영어, 중국어, 일본어)

---

## 📝 라이선스

이 프로젝트는 SK Networks Family 20기 AI 교육 과정의 일환로 제작되었습니다.

---

## 🙏 소감

- **김황현** - 이번에도 주제 선정에 어려움이 있었지만 주제 선정 후 다들 맡은 업무를 열심히 하여 고생은 했지만 재밌게 할 수 있었습니다. 고생하셨습니다.
- **문창교** - 모두가 열심히 해준 덕분에 나름 만족할만한 결과가 나온 것 같습니다. 모두 고생하셨습니다. 감사합니다.
- **김효빈** - 프로젝트 과정 동안 정말 많이 배우고 LLM의 전체적인 과정을 이해할 수 있는 유익한 시간이었습니다. 감사합니다 !
- **권규리** - 막연헀던 llm과 rag의 연관성을 관계를 이론적으로만 듣다가 직접 팀원들이 설명해주는 것 들으면서 이해할 수 있게 되었습니다. 덕분에 많이 배울 수 있었습니다.
- **이승규** - 수업시간에 이해하지 못했던 것들이 차근차근 이해가 되면서 전체 파이프라인과 시스템 아키텍처에 대한 깊이있는 이해가 가능했습니다. 너무 재밋어요!

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=wave&height=100&color=0:FFD86F,50:FFAA28,100:FF6B00&section=footer" />
</p>
