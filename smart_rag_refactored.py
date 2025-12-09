"""
Smart RAG 시스템 (하이브리드 리트리버)

Dense + BM25 결합 검색과 쿼리 분석 기반 라우팅을 수행하는 RAG 시스템.
ChromaDB 기반 벡터스토어와 호환되며, 대피소 정보 및 재난 행동요령 검색을 지원.

메타데이터 구조:
    - 대피소: type="shelter", facility_name, capacity, shelter_type, address 등
    - 행동요령: type="disaster_guideline", keyword, situation, category 등

Usage:
    rag = SmartRAG(config=RAGConfig(db_path="./chroma_db"))
    response = rag.invoke("지진 발생 시 행동요령")
"""

from __future__ import annotations

import json
import re
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

warnings.filterwarnings("ignore")
load_dotenv()


# =============================================================================
# 설정 (Configuration)
# =============================================================================


@dataclass
class RAGConfig:
    """RAG 시스템 설정."""

    db_path: str = r"C:\3project\chroma_db"
    collection_name: str = "shelter_and_disaster_guidelines"
    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    retriever_k: int = 5
    dense_weight: float = 0.5
    bm25_weight: float = 0.5
    default_display_limit: int = 15


# =============================================================================
# 열거형 (Enums)
# =============================================================================


class QueryType(str, Enum):
    """질문 유형 분류."""

    AGGREGATION = "AGGREGATION"  # 개수, 통계 질문
    FILTER = "FILTER"  # 조건 기반 목록 조회
    SHELTER_INFO = "SHELTER_INFO"  # 특정 대피소 정보
    GUIDELINE = "GUIDELINE"  # 재난 행동요령
    GENERAL = "GENERAL"  # 기타


class SourceType(str, Enum):
    """데이터 소스 유형."""

    SHELTER = "shelter"
    GUIDELINE = "disaster_guideline"


class FilterOperator(str, Enum):
    """필터 연산자."""

    GT = "gt"
    GTE = "gte"
    LT = "lt"
    LTE = "lte"
    EQ = "eq"
    NE = "ne"
    CONTAINS = "contains"


# =============================================================================
# 데이터 클래스 (Data Classes)
# =============================================================================


@dataclass
class FilterCondition:
    """필터 조건."""

    field: str
    operator: str
    value: Any


@dataclass
class QueryAnalysis:
    """쿼리 분석 결과."""

    conditions: list[FilterCondition] = field(default_factory=list)
    sort_by: str | None = None
    sort_order: str = "desc"
    limit: int | None = None


@dataclass
class FilterResult:
    """필터 실행 결과."""

    ids: list[str]
    metadatas: list[dict[str, Any]]
    documents: list[str]

    @property
    def count(self) -> int:
        return len(self.ids)

    @property
    def is_empty(self) -> bool:
        return self.count == 0


@dataclass
class ShelterStats:
    """대피소 통계."""

    capacity_min: int
    capacity_max: int
    capacity_avg: float
    capacity_total: int
    shelter_types: dict[str, int]
    top_districts: dict[str, int]


@dataclass
class GuidelineStats:
    """행동요령 통계."""

    disaster_types: dict[str, int]


@dataclass
class DBStats:
    """DB 통계."""

    total_documents: int
    shelter_count: int
    guideline_count: int
    shelter_stats: ShelterStats | None = None
    guideline_stats: GuidelineStats | None = None


# =============================================================================
# 유틸리티 (Utilities)
# =============================================================================


def normalize_korean_numbers(text: str) -> str:
    """
    한국어 숫자 표현을 정규화.

    Examples:
        "1만" -> "10000"
        "1.5만" -> "15000"
        "1,000" -> "1000"
    """
    patterns: list[tuple[str, Callable[[re.Match], str]]] = [
        (r"(\d+)\s*만", lambda m: str(int(m.group(1)) * 10000)),
        (
            r"(\d+)\.(\d+)\s*만",
            lambda m: str(int(float(f"{m.group(1)}.{m.group(2)}") * 10000)),
        ),
    ]

    for pattern, replacement in patterns:
        text = re.sub(pattern, replacement, text)

    # 천 단위 콤마 제거
    text = re.sub(r"(\d+),(\d+)", r"\1\2", text)
    return text


def extract_district(address: str | None) -> str | None:
    """주소에서 구/군 정보 추출."""
    if not address:
        return None
    match = re.search(r"([가-힣]+[구군])", address)
    return match.group(1) if match else None


def format_documents(docs: list[Document]) -> str:
    """문서 리스트를 문자열로 포맷."""
    if not docs:
        return ""
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


def get_metadata_value(metadata: dict[str, Any], key: str, default: Any = None) -> Any:
    """메타데이터에서 값을 안전하게 추출."""
    return metadata.get(key, default)


# =============================================================================
# 재난 유형 매핑
# =============================================================================

DISASTER_KEYWORDS: dict[str, str] = {
    "지진": "지진",
    "earthquake": "지진",
    "화재": "화재",
    "불": "화재",
    "fire": "화재",
    "태풍": "태풍",
    "typhoon": "태풍",
    "폭풍": "폭풍",
    "storm": "폭풍",
    "홍수": "홍수",
    "침수": "홍수",
    "flood": "홍수",
    "산사태": "산사태",
    "landslide": "산사태",
    "해일": "해일",
    "쓰나미": "쓰나미",
    "tsunami": "쓰나미",
    "화산": "화산",
    "volcanic": "화산",
    "가스": "가스",
    "gas": "가스",
    "산불": "산불",
    "wildfire": "산불",
    "방사능": "방사능",
    "방사선": "방사능",
    "radiation": "방사능",
    "댐": "댐",
    "dam": "댐",
}


# =============================================================================
# 하이브리드 리트리버 (Hybrid Retriever)
# =============================================================================


class HybridRetriever:
    """
    Dense(의미) + BM25(키워드) 결합 리트리버.

    RRF(Reciprocal Rank Fusion)를 사용하여 두 검색 결과를 통합.
    """

    def __init__(
        self,
        vectorstore: Chroma,
        documents: list[Document],
        k: int = 5,
        dense_weight: float = 0.5,
        bm25_weight: float = 0.5,
    ) -> None:
        self.vectorstore = vectorstore
        self.k = k
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight

        # 내부 검색 시 더 많은 후보를 가져옴
        self._internal_k = k * 2

        # Dense 리트리버
        self.dense_retriever = vectorstore.as_retriever(
            search_kwargs={"k": self._internal_k}
        )

        # BM25 리트리버
        self.bm25_retriever = BM25Retriever.from_documents(documents)
        self.bm25_retriever.k = self._internal_k

    def invoke(
        self, query: str, source_filter: SourceType | None = None
    ) -> list[Document]:
        """
        하이브리드 검색 실행.

        Args:
            query: 검색 쿼리
            source_filter: 소스 타입 필터

        Returns:
            검색된 문서 리스트
        """
        normalized_query = normalize_korean_numbers(query)

        dense_docs = self._search_dense(normalized_query, source_filter)
        bm25_docs = self._search_bm25(normalized_query, source_filter)

        return self._fuse_results(dense_docs, bm25_docs)

    def _search_dense(
        self, query: str, source_filter: SourceType | None
    ) -> list[Document]:
        """Dense 검색 수행."""
        if source_filter:
            return self.vectorstore.similarity_search(
                query, k=self._internal_k, filter={"type": source_filter.value}
            )
        return self.dense_retriever.invoke(query)

    def _search_bm25(
        self, query: str, source_filter: SourceType | None
    ) -> list[Document]:
        """BM25 검색 수행."""
        docs = self.bm25_retriever.invoke(query)

        if source_filter:
            docs = [
                doc
                for doc in docs
                if doc.metadata.get("type") == source_filter.value
            ]
        return docs

    def _fuse_results(
        self, dense_docs: list[Document], bm25_docs: list[Document]
    ) -> list[Document]:
        """RRF로 검색 결과 통합."""
        scores: dict[str, float] = {}
        doc_map: dict[str, Document] = {}

        # Dense 결과에 RRF 스코어 부여
        for rank, doc in enumerate(dense_docs):
            key = self._get_doc_key(doc)
            scores[key] = scores.get(key, 0) + self.dense_weight / (rank + 1)
            doc_map[key] = doc

        # BM25 결과에 RRF 스코어 부여
        for rank, doc in enumerate(bm25_docs):
            key = self._get_doc_key(doc)
            scores[key] = scores.get(key, 0) + self.bm25_weight / (rank + 1)
            doc_map[key] = doc

        # 스코어 기준 정렬 후 상위 k개 반환
        ranked_keys = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
        return [doc_map[k] for k in ranked_keys[: self.k]]

    @staticmethod
    def _get_doc_key(doc: Document) -> str:
        """문서 식별용 키 생성."""
        return doc.page_content[:200]


# =============================================================================
# 쿼리 분석기 (Query Analyzer)
# =============================================================================


class QueryAnalyzer:
    """LLM 기반 질문 유형 분류 및 조건 추출."""

    def __init__(self, llm: ChatOpenAI) -> None:
        self.llm = llm

    def classify(self, query: str) -> QueryType:
        """질문 유형 분류."""
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """질문을 분류하라. 한 단어로만 답하라.

분류 기준:
- AGGREGATION: 개수, 통계 질문 (몇 곳, 몇 개, 총, 평균, 최대, 최소 등)
- FILTER: 조건에 맞는 목록 조회 (N명 이상인 곳, 지하 대피소 목록 등)
- SHELTER_INFO: 특정 대피소 하나의 정보 (OO시설 주소 등)
- GUIDELINE: 재난 대응/행동요령 질문 (지진 발생 시, 화재 대피 방법 등)
- GENERAL: 위 분류에 해당하지 않는 질문

답변: AGGREGATION, FILTER, SHELTER_INFO, GUIDELINE, GENERAL 중 하나""",
                ),
                ("human", "{query}"),
            ]
        )

        result = (prompt | self.llm).invoke({"query": query})
        response = result.content.strip().upper()

        for query_type in QueryType:
            if query_type.value in response:
                return query_type
        return QueryType.GENERAL

    def extract_conditions(self, query: str) -> QueryAnalysis:
        """질문에서 필터 조건 추출."""
        normalized_query = normalize_korean_numbers(query)

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """질문에서 대피소 검색 조건을 추출하여 JSON으로 반환하라.

가능한 조건 필드:
1. capacity: {{"field": "capacity", "operator": "gt/gte/lt/lte/eq", "value": 숫자}}
2. shelter_type: {{"field": "shelter_type", "operator": "eq", "value": "지상/지하"}}
3. facility_type: {{"field": "facility_type", "operator": "eq", "value": "문자열"}}
4. address: {{"field": "address", "operator": "contains", "value": "문자열"}}
5. facility_name: {{"field": "facility_name", "operator": "contains", "value": "문자열"}}

연산자:
- "초과", "넘는" → "gt"
- "이상" → "gte"
- "미만" → "lt"
- "이하" → "lte"
- "인", "같은" → "eq"
- "포함", "있는" → "contains"

반환 형식:
{{"conditions": [...], "sort_by": "필드명 또는 null", "sort_order": "desc/asc 또는 null", "limit": 숫자 또는 null}}

JSON만 출력하라.""",
                ),
                ("human", "{query}"),
            ]
        )

        result = (prompt | self.llm).invoke({"query": normalized_query})
        return self._parse_conditions(result.content)

    def extract_disaster_type(self, query: str) -> str | None:
        """질문에서 재난 유형 추출."""
        query_lower = query.lower()

        for keyword, disaster in DISASTER_KEYWORDS.items():
            if keyword in query_lower:
                return disaster
        return None

    @staticmethod
    def _parse_conditions(content: str) -> QueryAnalysis:
        """LLM 응답에서 조건 파싱."""
        try:
            cleaned = content.strip()
            cleaned = re.sub(r"```json?\n?", "", cleaned)
            cleaned = re.sub(r"```", "", cleaned)

            data = json.loads(cleaned)

            conditions = [
                FilterCondition(
                    field=c["field"], operator=c["operator"], value=c["value"]
                )
                for c in data.get("conditions", [])
            ]

            return QueryAnalysis(
                conditions=conditions,
                sort_by=data.get("sort_by"),
                sort_order=data.get("sort_order", "desc"),
                limit=data.get("limit"),
            )
        except (json.JSONDecodeError, KeyError):
            return QueryAnalysis()


# =============================================================================
# 메타데이터 필터 (Metadata Filter)
# =============================================================================


class MetadataFilter:
    """ChromaDB 메타데이터 필터링."""

    CHROMA_OPERATOR_MAP: dict[str, str] = {
        FilterOperator.GT.value: "$gt",
        FilterOperator.GTE.value: "$gte",
        FilterOperator.LT.value: "$lt",
        FilterOperator.LTE.value: "$lte",
        FilterOperator.EQ.value: "$eq",
        FilterOperator.NE.value: "$ne",
    }

    def __init__(self, vectorstore: Chroma) -> None:
        self.vectorstore = vectorstore

    def execute(
        self, conditions: list[FilterCondition], source_type: SourceType
    ) -> FilterResult:
        """조건에 맞는 문서 필터링."""
        # 소스 타입 조건 추가
        all_conditions = [
            FilterCondition(
                field="type", operator=FilterOperator.EQ.value, value=source_type.value
            ),
            *conditions,
        ]

        # contains 연산자가 있으면 수동 필터링
        if any(c.operator == FilterOperator.CONTAINS.value for c in conditions):
            return self._manual_filter(all_conditions)

        try:
            return self._chroma_filter(all_conditions)
        except Exception as e:
            print(f"Chroma 필터 오류, 수동 필터링으로 전환: {e}")
            return self._manual_filter(all_conditions)

    def _chroma_filter(self, conditions: list[FilterCondition]) -> FilterResult:
        """Chroma 네이티브 필터링."""
        where_clauses = [self._build_where_clause(c) for c in conditions]

        where = where_clauses[0] if len(where_clauses) == 1 else {"$and": where_clauses}

        results = self.vectorstore.get(where=where, include=["metadatas", "documents"])

        return FilterResult(
            ids=results.get("ids", []),
            metadatas=results.get("metadatas", []),
            documents=results.get("documents", []),
        )

    def _build_where_clause(self, condition: FilterCondition) -> dict[str, Any]:
        """단일 조건을 Chroma where 절로 변환."""
        chroma_op = self.CHROMA_OPERATOR_MAP.get(condition.operator, "$eq")
        return {condition.field: {chroma_op: condition.value}}

    def _manual_filter(self, conditions: list[FilterCondition]) -> FilterResult:
        """수동 필터링 (contains 등 Chroma 미지원 연산자용)."""
        all_data = self.vectorstore.get(include=["metadatas", "documents"])

        filtered_ids = []
        filtered_metadatas = []
        filtered_documents = []

        for i, metadata in enumerate(all_data["metadatas"]):
            if self._matches_conditions(metadata, conditions):
                filtered_ids.append(all_data["ids"][i])
                filtered_metadatas.append(metadata)
                filtered_documents.append(all_data["documents"][i])

        return FilterResult(
            ids=filtered_ids,
            metadatas=filtered_metadatas,
            documents=filtered_documents,
        )

    @staticmethod
    def _matches_conditions(
        metadata: dict[str, Any], conditions: list[FilterCondition]
    ) -> bool:
        """메타데이터가 모든 조건을 만족하는지 확인."""
        for cond in conditions:
            field_value = metadata.get(cond.field)

            if cond.operator == FilterOperator.CONTAINS.value:
                if not field_value or cond.value.lower() not in str(field_value).lower():
                    return False
            elif cond.operator == FilterOperator.GT.value:
                if not (field_value and field_value > cond.value):
                    return False
            elif cond.operator == FilterOperator.GTE.value:
                if not (field_value and field_value >= cond.value):
                    return False
            elif cond.operator == FilterOperator.LT.value:
                if not (field_value and field_value < cond.value):
                    return False
            elif cond.operator == FilterOperator.LTE.value:
                if not (field_value and field_value <= cond.value):
                    return False
            elif cond.operator == FilterOperator.EQ.value:
                if str(field_value) != str(cond.value):
                    return False

        return True


# =============================================================================
# 결과 포매터 (Result Formatter)
# =============================================================================


class ResultFormatter:
    """검색 결과 포맷팅."""

    @staticmethod
    def format_aggregation(result: FilterResult) -> str:
        """집계 결과 포맷."""
        if result.is_empty:
            return "해당 조건에 맞는 대피소가 없습니다."

        response = f"총 **{result.count}곳**입니다."

        items = result.metadatas
        if result.count <= 10:
            response += "\n\n**목록:**"
            for meta in items:
                response += ResultFormatter._format_shelter_brief(meta)
        else:
            response += "\n\n**상위 10개 (수용인원 기준):**"
            sorted_items = sorted(
                items, key=lambda x: x.get("capacity", 0), reverse=True
            )[:10]
            for meta in sorted_items:
                name = meta.get("facility_name", "알 수 없음")
                capacity = meta.get("capacity", 0)
                response += f"\n- {name} (수용인원: {capacity:,}명)"

        return response

    @staticmethod
    def format_filter(
        result: FilterResult,
        sort_by: str | None = None,
        sort_order: str = "desc",
        limit: int = 15,
    ) -> str:
        """필터 결과 포맷."""
        if result.is_empty:
            return "해당 조건에 맞는 대피소가 없습니다."

        items = list(zip(result.metadatas, result.documents))

        # 정렬
        if sort_by and items and sort_by in items[0][0]:
            items.sort(key=lambda x: x[0].get(sort_by, 0), reverse=(sort_order == "desc"))

        response = f"총 **{result.count}곳**이 검색되었습니다.\n"

        for meta, _ in items[:limit]:
            response += ResultFormatter._format_shelter_detail(meta)

        if result.count > limit:
            response += f"\n... 외 {result.count - limit}곳"

        return response

    @staticmethod
    def format_shelter_info(docs: list[Document]) -> str:
        """특정 대피소 정보 포맷."""
        if not docs:
            return "해당 시설 정보를 찾을 수 없습니다."

        meta = docs[0].metadata

        return (
            f"**{meta.get('facility_name', '알 수 없음')}**\n\n"
            f"- **시설구분**: {meta.get('facility_type', '-')}\n"
            f"- **위치**: {meta.get('shelter_type', '-')}\n"
            f"- **주소**: {meta.get('address', '-')}\n"
            f"- **수용인원**: {meta.get('capacity', 0):,}명\n"
            f"- **운영상태**: {meta.get('operating_status', '-')}"
        )

    @staticmethod
    def _format_shelter_brief(meta: dict[str, Any]) -> str:
        """대피소 간략 정보 포맷."""
        name = meta.get("facility_name", "알 수 없음")
        capacity = meta.get("capacity", 0)
        shelter_type = meta.get("shelter_type", "")
        return f"\n- {name} ({shelter_type}, 수용인원: {capacity:,}명)"

    @staticmethod
    def _format_shelter_detail(meta: dict[str, Any]) -> str:
        """대피소 상세 정보 포맷."""
        name = meta.get("facility_name", "알 수 없음")
        capacity = meta.get("capacity", 0)
        shelter_type = meta.get("shelter_type", "")
        address = meta.get("address", "")

        return (
            f"\n**{name}**"
            f"\n  - 위치: {shelter_type} | 수용인원: {capacity:,}명"
            f"\n  - 주소: {address}\n"
        )


# =============================================================================
# Smart RAG 메인 클래스
# =============================================================================


class SmartRAG:
    """질문 유형별 최적 처리를 수행하는 통합 RAG 시스템."""

    RAG_SYSTEM_PROMPT = """너는 민방위 대피소 및 재난 행동요령 안내 Assistant다.

규칙:
1. 반드시 문맥(context)에 포함된 정보만 사용해서 답한다.
2. 문맥이 비어있거나 관련 정보가 없으면: "제공된 정보에서 해당 내용을 찾을 수 없습니다."
3. 새로운 지명, 수치, 예시를 만들어내지 않는다.
4. 답변은 간결하고 정확하게 한다.
5. 행동요령은 단계별로 명확하게 안내한다."""

    def __init__(self, config: RAGConfig | None = None) -> None:
        self.config = config or RAGConfig()
        print("SmartRAG 초기화 중...")

        self._init_models()
        self._init_vectorstore()
        self._init_documents()
        self._init_components()

        print("SmartRAG 초기화 완료!\n")

    def _init_models(self) -> None:
        """LLM 및 임베딩 모델 초기화."""
        self.llm = ChatOpenAI(model=self.config.llm_model, temperature=0)
        self.embeddings = OpenAIEmbeddings(model=self.config.embedding_model)

    def _init_vectorstore(self) -> None:
        """Vectorstore 초기화."""
        self.vectorstore = Chroma(
            persist_directory=self.config.db_path,
            embedding_function=self.embeddings,
            collection_name=self.config.collection_name,
        )

    def _init_documents(self) -> None:
        """문서 로드 및 분류."""
        raw = self.vectorstore.get(include=["documents", "metadatas"])

        self.all_documents = [
            Document(page_content=normalize_korean_numbers(text), metadata=meta)
            for text, meta in zip(raw["documents"], raw["metadatas"])
        ]

        self.shelter_docs = [
            doc
            for doc in self.all_documents
            if doc.metadata.get("type") == SourceType.SHELTER.value
        ]
        self.guideline_docs = [
            doc
            for doc in self.all_documents
            if doc.metadata.get("type") == SourceType.GUIDELINE.value
        ]

        print(f"  - 총 문서: {len(self.all_documents)}개")
        print(f"  - 대피소: {len(self.shelter_docs)}개")
        print(f"  - 행동요령: {len(self.guideline_docs)}개")

        if not self.all_documents:
            self._warn_empty_db()

    def _warn_empty_db(self) -> None:
        """빈 DB 경고 출력."""
        print("\n⚠️  경고: DB에 문서가 없습니다!")
        print("   from_DataLoad_to_VectorDB.py를 먼저 실행하여 DB를 생성하세요.")
        print(f"   현재 DB 경로: {self.config.db_path}")
        print(f"   컬렉션 이름: {self.config.collection_name}\n")

    def _init_components(self) -> None:
        """컴포넌트 초기화."""
        self.hybrid_retriever = (
            HybridRetriever(
                self.vectorstore,
                self.all_documents,
                k=self.config.retriever_k,
                dense_weight=self.config.dense_weight,
                bm25_weight=self.config.bm25_weight,
            )
            if self.all_documents
            else None
        )

        self.query_analyzer = QueryAnalyzer(self.llm)
        self.metadata_filter = MetadataFilter(self.vectorstore)
        self.formatter = ResultFormatter()

        self.rag_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.RAG_SYSTEM_PROMPT),
                (
                    "human",
                    """문맥:
{context}

질문: {question}

답변:""",
                ),
            ]
        )

    # -------------------------------------------------------------------------
    # 공개 메서드 (Public Methods)
    # -------------------------------------------------------------------------

    def invoke(self, query: str, verbose: bool = True) -> str:
        """
        질문 처리 메인 함수.

        Args:
            query: 사용자 질문
            verbose: 디버그 정보 출력 여부

        Returns:
            생성된 응답
        """
        if not self.all_documents:
            return (
                "DB에 문서가 없습니다. "
                "from_DataLoad_to_VectorDB.py를 먼저 실행하여 데이터를 임베딩하세요."
            )

        query_type = self.query_analyzer.classify(query)
        if verbose:
            print(f"  [유형: {query_type.value}]")

        handlers = {
            QueryType.AGGREGATION: self._handle_aggregation,
            QueryType.FILTER: self._handle_filter,
            QueryType.SHELTER_INFO: self._handle_shelter_info,
            QueryType.GUIDELINE: self._handle_guideline,
            QueryType.GENERAL: self._handle_general,
        }

        handler = handlers.get(query_type, self._handle_general)
        return handler(query, verbose)

    def search(
        self, query: str, k: int = 5, source_filter: SourceType | None = None
    ) -> list[Document]:
        """하이브리드 검색 (디버깅용)."""
        if not self.hybrid_retriever:
            return []
        self.hybrid_retriever.k = k
        return self.hybrid_retriever.invoke(query, source_filter)

    def get_stats(self) -> DBStats:
        """DB 통계 조회."""
        shelter_result = self.metadata_filter.execute([], SourceType.SHELTER)
        guideline_result = self.metadata_filter.execute([], SourceType.GUIDELINE)

        return DBStats(
            total_documents=len(self.all_documents),
            shelter_count=shelter_result.count,
            guideline_count=guideline_result.count,
            shelter_stats=self._compute_shelter_stats(shelter_result),
            guideline_stats=self._compute_guideline_stats(guideline_result),
        )

    # -------------------------------------------------------------------------
    # 핸들러 메서드 (Handler Methods)
    # -------------------------------------------------------------------------

    def _handle_aggregation(self, query: str, verbose: bool) -> str:
        """집계 쿼리 처리."""
        analysis = self.query_analyzer.extract_conditions(query)

        if verbose:
            print(f"  [조건: {[c.__dict__ for c in analysis.conditions]}]")

        result = self.metadata_filter.execute(analysis.conditions, SourceType.SHELTER)
        return self.formatter.format_aggregation(result)

    def _handle_filter(self, query: str, verbose: bool) -> str:
        """필터 쿼리 처리."""
        analysis = self.query_analyzer.extract_conditions(query)

        if verbose:
            print(f"  [조건: {[c.__dict__ for c in analysis.conditions]}]")

        if analysis.conditions:
            result = self.metadata_filter.execute(analysis.conditions, SourceType.SHELTER)
            return self.formatter.format_filter(
                result,
                analysis.sort_by,
                analysis.sort_order,
                analysis.limit or self.config.default_display_limit,
            )

        if verbose:
            print("  [조건 추출 실패, 검색 폴백]")
        docs = self.hybrid_retriever.invoke(query, SourceType.SHELTER)
        return self._generate_response(query, docs)

    def _handle_shelter_info(self, query: str, verbose: bool) -> str:
        """특정 대피소 정보 조회."""
        docs = self.hybrid_retriever.invoke(query, SourceType.SHELTER)
        return self.formatter.format_shelter_info(docs) if docs else "해당 시설 정보를 찾을 수 없습니다."

    def _handle_guideline(self, query: str, verbose: bool) -> str:
        """재난 행동요령 질문 처리."""
        disaster_type = self.query_analyzer.extract_disaster_type(query)
        if verbose and disaster_type:
            print(f"  [재난유형: {disaster_type}]")

        docs = self.hybrid_retriever.invoke(query, SourceType.GUIDELINE)
        return self._generate_response(query, docs)

    def _handle_general(self, query: str, verbose: bool) -> str:
        """일반 질문 처리."""
        docs = self.hybrid_retriever.invoke(query)
        return self._generate_response(query, docs)

    # -------------------------------------------------------------------------
    # 내부 메서드 (Private Methods)
    # -------------------------------------------------------------------------

    def _generate_response(self, query: str, docs: list[Document]) -> str:
        """LLM 응답 생성."""
        context = format_documents(docs)
        chain = self.rag_prompt | self.llm | StrOutputParser()
        return chain.invoke({"context": context, "question": query})

    @staticmethod
    def _compute_shelter_stats(result: FilterResult) -> ShelterStats | None:
        """대피소 통계 계산."""
        if result.is_empty:
            return None

        capacities = [m.get("capacity", 0) for m in result.metadatas]
        shelter_types: dict[str, int] = {}
        districts: dict[str, int] = {}

        for meta in result.metadatas:
            # 대피소 유형 집계
            shelter_type = meta.get("shelter_type", "기타")
            shelter_types[shelter_type] = shelter_types.get(shelter_type, 0) + 1

            # 지역구 집계
            district = extract_district(meta.get("address")) or "기타"
            districts[district] = districts.get(district, 0) + 1

        return ShelterStats(
            capacity_min=min(capacities) if capacities else 0,
            capacity_max=max(capacities) if capacities else 0,
            capacity_avg=sum(capacities) / len(capacities) if capacities else 0,
            capacity_total=sum(capacities),
            shelter_types=shelter_types,
            top_districts=dict(
                sorted(districts.items(), key=lambda x: x[1], reverse=True)[:10]
            ),
        )

    @staticmethod
    def _compute_guideline_stats(result: FilterResult) -> GuidelineStats | None:
        """행동요령 통계 계산."""
        if result.is_empty:
            return None

        disaster_types: dict[str, int] = {}
        for meta in result.metadatas:
            disaster = meta.get("keyword", "기타")
            disaster_types[disaster] = disaster_types.get(disaster, 0) + 1

        return GuidelineStats(disaster_types=disaster_types)


# =============================================================================
# 실행 (Main)
# =============================================================================


def print_stats(stats: DBStats) -> None:
    """통계 출력."""
    print("=" * 70)
    print("DB 통계:")
    print(f"  총 문서: {stats.total_documents}")
    print(f"  대피소: {stats.shelter_count}")
    print(f"  행동요령: {stats.guideline_count}")

    if stats.shelter_stats:
        ss = stats.shelter_stats
        print(f"\n  [대피소 통계]")
        print(
            f"  수용인원: 최소 {ss.capacity_min:,} / "
            f"최대 {ss.capacity_max:,} / 평균 {ss.capacity_avg:,.0f}"
        )
        print(f"  총 수용가능: {ss.capacity_total:,}명")
        print(f"  위치유형: {ss.shelter_types}")

    if stats.guideline_stats:
        gs = stats.guideline_stats
        print(f"\n  [행동요령 통계]")
        print(f"  재난유형: {gs.disaster_types}")

    print("=" * 70)


def run_test_queries(rag: SmartRAG) -> None:
    """테스트 쿼리 실행."""
    test_questions = [
        # SHELTER_INFO
        "대동세무고등학교 대피소 정보 알려줘",
        "동대문맨션의 최대수용인원은?",
        # AGGREGATION
        "최대 수용 인원이 1만 명 넘는 대피소가 몇 곳있어?",
        "지하에 있는 대피소는 총 몇 곳이야?",
        # FILTER
        "수용인원 500명 이하인 곳은?",
        "종로구에 있는 대피소 목록",
        # GUIDELINE
        "지진 발생 시 행동요령 알려줘",
        "화재가 났을 때 어떻게 대피해야 해?",
    ]

    for question in test_questions:
        print(f"\n{'=' * 70}")
        print(f"질문: {question}")
        print("-" * 70)
        answer = rag.invoke(question)
        print(f"답변:\n{answer}")


def main() -> None:
    """메인 함수."""
    config = RAGConfig()
    rag = SmartRAG(config)

    print_stats(rag.get_stats())
    run_test_queries(rag)


if __name__ == "__main__":
    main()
