"""
metadata_tagger.py
자동 메타데이터 태깅 모듈

주요 기능:
1. HuggingFace NER 모델을 사용한 인물/장소 추출
2. LLM 기반 감정 분석
3. 키워드 추출
4. 요약 생성
5. Document 객체에 메타데이터 추가
"""

import re
from typing import List, Dict, Set
from langchain_core.documents import Document
from collections import Counter


class MetadataTagger:
    """메타데이터 자동 태깅 클래스"""
    
    def __init__(self, use_llm: bool = False):
        """
        Args:
            use_llm: LLM 기반 태깅 사용 여부 (False면 규칙 기반)
        """
        self.use_llm = use_llm
        
        # 해리포터 주요 인물 사전 (규칙 기반)
        self.known_characters = {
            "해리", "해리 포터", "론", "론 위즐리", "헤르미온느", "헤르미온느 그레인저",
            "덤블도어", "스네이프", "볼드모트", "시리우스", "해그리드", "맥고나걸",
            "드레이코", "드레이코 말포이", "지니", "프레드", "조지", "퍼시",
            "빌", "찰리", "아서", "몰리", "네빌", "루나", "초", "세드릭",
            "더들리", "버논", "페투니아", "두들리", "더즐리", "제임스", "릴리",
            "리무스", "루핀", "통스", "무디", "킹슬리", "위즐리", "말포이",
            "크라우치", "베라트릭스", "루시우스", "나르시사", "벨라트릭스",
            "도비", "크리처", "윈키", "호그와트", "그린델왈드"
        }
        
        # 주요 장소 사전
        self.known_locations = {
            "호그와트", "그리핀도르", "슬리데린", "래번클로", "후플푸프",
            "다이애건 앨리", "호그스미드", "금단의 숲", "프리벳가", "버로우",
            "아즈카반", "마법부", "고드릭 골짜기", "스핀너스 엔드",
            "마법사의 돌", "비밀의 방", "필요의 방", "호그와트 급행열차"
        }
        
        print(f"[INFO] MetadataTagger 초기화 (use_llm: {use_llm})")
    
    def extract_characters_rule_based(self, text: str) -> List[str]:
        """
        규칙 기반 인물 추출
        
        Args:
            text: 텍스트
        
        Returns:
            List[str]: 인물 리스트
        """
        characters = set()
        
        text_lower = text
        
        for char in self.known_characters:
            # 정확한 매칭
            if char in text_lower:
                characters.add(char)
        
        return sorted(list(characters))
    
    def extract_locations_rule_based(self, text: str) -> List[str]:
        """
        규칙 기반 장소 추출
        
        Args:
            text: 텍스트
        
        Returns:
            List[str]: 장소 리스트
        """
        locations = set()
        
        for loc in self.known_locations:
            if loc in text:
                locations.add(loc)
        
        return sorted(list(locations))
    
    def analyze_sentiment_rule_based(self, text: str) -> str:
        """
        규칙 기반 감정 분석
        
        Args:
            text: 텍스트
        
        Returns:
            str: 감정 (positive/negative/neutral)
        """
        # 긍정/부정 키워드
        positive_words = [
            "웃", "행복", "기쁨", "즐거", "사랑", "좋", "훌륭", "멋진", "아름다",
            "친구", "승리", "성공", "희망", "빛", "따뜻"
        ]
        
        negative_words = [
            "두려", "무서", "슬프", "아픔", "고통", "죽음", "어둠", "악", "증오",
            "분노", "절망", "비명", "공포", "위험", "저주", "싸움", "전쟁"
        ]
        
        positive_count = sum(1 for word in positive_words if word in text)
        negative_count = sum(1 for word in negative_words if word in text)
        
        if positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "negative"
        else:
            return "neutral"
    
    def extract_keywords(self, text: str, top_k: int = 5) -> List[str]:
        """
        키워드 추출 (빈도 기반)
        
        Args:
            text: 텍스트
            top_k: 추출할 키워드 수
        
        Returns:
            List[str]: 키워드 리스트
        """
        # 명사 추출 (간단한 규칙: 2글자 이상 한글)
        words = re.findall(r'[가-힣]{2,}', text)
        
        # 불용어 제거
        stopwords = {
            "그는", "그녀", "그들", "우리", "저는", "나는", "있는", "없는", 
            "하는", "되는", "이다", "것이", "수가", "때문", "그것", "이것",
            "저것", "어떤", "무엇", "어디", "언제", "누구", "어떻게"
        }
        
        words = [w for w in words if w not in stopwords and len(w) >= 2]
        
        # 빈도 계산
        word_counts = Counter(words)
        
        # 상위 키워드
        top_keywords = [word for word, count in word_counts.most_common(top_k)]
        
        return top_keywords
    
    def generate_summary(self, text: str, max_length: int = 100) -> str:
        """
        간단한 요약 생성 (첫 문장들)
        
        Args:
            text: 텍스트
            max_length: 최대 길이
        
        Returns:
            str: 요약
        """
        # 첫 몇 문장 추출
        sentences = re.split(r'[.!?]\s+', text)
        
        summary = ""
        for sent in sentences:
            if len(summary) + len(sent) <= max_length:
                summary += sent + ". "
            else:
                break
        
        return summary.strip()
    
    def add_metadata_to_documents(self, documents: List[Document]) -> List[Document]:
        """
        Document 객체에 메타데이터 추가
        
        Args:
            documents: Document 객체 리스트
        
        Returns:
            List[Document]: 메타데이터가 추가된 Document 리스트
        """
        print(f"\n[STEP 3] 메타데이터 태깅 시작 ({len(documents)}개 청크)")
        print("="*80)
        
        tagged_documents = []
        
        for i, doc in enumerate(documents):
            text = doc.page_content
            
            # 기존 메타데이터 복사
            metadata = doc.metadata.copy()
            
            # 인물 추출
            characters = self.extract_characters_rule_based(text)
            metadata["characters"] = characters
            
            # 장소 추출
            locations = self.extract_locations_rule_based(text)
            metadata["locations"] = locations
            
            # 감정 분석
            sentiment = self.analyze_sentiment_rule_based(text)
            metadata["sentiment"] = sentiment
            
            # 키워드 추출
            keywords = self.extract_keywords(text, top_k=5)
            metadata["keywords"] = keywords
            
            # 요약 생성
            summary = self.generate_summary(text, max_length=80)
            metadata["summary"] = summary
            
            # 새 Document 생성
            tagged_doc = Document(
                page_content=text,
                metadata=metadata
            )
            
            tagged_documents.append(tagged_doc)
            
            # 진행상황 출력 (10%마다)
            if (i + 1) % max(1, len(documents) // 10) == 0:
                progress = (i + 1) / len(documents) * 100
                print(f"[진행] {i+1}/{len(documents)} ({progress:.1f}%) 완료")
        
        print(f"[완료] 총 {len(tagged_documents)}개 청크 메타데이터 태깅 완료")
        print("="*80 + "\n")
        
        return tagged_documents
    
    def print_metadata_stats(self, documents: List[Document]):
        """
        메타데이터 통계 출력
        
        Args:
            documents: Document 객체 리스트
        """
        print("\n[메타데이터 통계]")
        print("-" * 80)
        
        # 인물 통계
        all_characters = []
        for doc in documents:
            all_characters.extend(doc.metadata.get("characters", []))
        
        char_counts = Counter(all_characters)
        print(f"\n[인물 Top 10]")
        for char, count in char_counts.most_common(10):
            print(f"  - {char}: {count}회")
        
        # 장소 통계
        all_locations = []
        for doc in documents:
            all_locations.extend(doc.metadata.get("locations", []))
        
        loc_counts = Counter(all_locations)
        print(f"\n[장소 Top 10]")
        for loc, count in loc_counts.most_common(10):
            print(f"  - {loc}: {count}회")
        
        # 감정 통계
        sentiments = [doc.metadata.get("sentiment", "neutral") for doc in documents]
        sentiment_counts = Counter(sentiments)
        print(f"\n[감정 분포]")
        for sentiment, count in sentiment_counts.items():
            print(f"  - {sentiment}: {count}개 ({count/len(documents)*100:.1f}%)")
        
        print("-" * 80)


def main():
    """테스트용 메인 함수"""
    from preprocess import TextPreprocessor
    from chapter_splitter import ChapterSplitter
    
    # 데이터 로드 및 전처리
    data_dir = r"c:\Users\ansck\Desktop\Project\3rd_project\data\cleaned_data"
    preprocessor = TextPreprocessor(data_dir)
    processed_files = preprocessor.preprocess_all(remove_title=False)
    
    # 챕터 분리 및 청킹
    splitter = ChapterSplitter(chunk_size=800, chunk_overlap=150)
    
    all_documents = []
    for file_info in processed_files[:1]:  # 테스트: 첫 번째 책만
        book_title = preprocessor.extract_book_title(file_info['filename'])
        documents = splitter.process_book(file_info['text'], book_title)
        all_documents.extend(documents)
    
    # 메타데이터 태깅
    tagger = MetadataTagger(use_llm=False)
    tagged_documents = tagger.add_metadata_to_documents(all_documents)
    
    # 통계 출력
    tagger.print_metadata_stats(tagged_documents)
    
    # 샘플 출력
    print("\n[샘플 Document]")
    if tagged_documents:
        sample_doc = tagged_documents[10]
        print(f"책: {sample_doc.metadata['book']}")
        print(f"장: 제{sample_doc.metadata['chapter_number']}장 - {sample_doc.metadata['chapter_title']}")
        print(f"인물: {sample_doc.metadata['characters']}")
        print(f"장소: {sample_doc.metadata['locations']}")
        print(f"감정: {sample_doc.metadata['sentiment']}")
        print(f"키워드: {sample_doc.metadata['keywords']}")
        print(f"요약: {sample_doc.metadata['summary']}")


if __name__ == "__main__":
    main()
