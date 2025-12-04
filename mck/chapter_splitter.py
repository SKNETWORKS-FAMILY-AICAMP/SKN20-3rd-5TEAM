"""
chapter_splitter.py
장(Chapter) 자동 감지 및 분리 모듈

주요 기능:
1. 정규식을 사용한 다양한 장 패턴 감지
2. 장별 텍스트 분리
3. RecursiveCharacterTextSplitter를 사용한 청킹
4. Document 객체 생성
"""

import re
from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class ChapterSplitter:
    """장 감지 및 청킹 클래스"""
    
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150):
        """
        Args:
            chunk_size: 청크 크기
            chunk_overlap: 청크 중첩 크기
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # RecursiveCharacterTextSplitter 초기화
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", "다.", "요.", ""],
            length_function=len,
        )
        
        print(f"[INFO] ChapterSplitter 초기화")
        print(f"  - chunk_size: {chunk_size}")
        print(f"  - chunk_overlap: {chunk_overlap}")
    
    def detect_chapters(self, text: str) -> List[Dict[str, any]]:
        """
        정규식을 사용하여 장 감지
        
        패턴:
        - "CHAPTER ONE", "CHAPTER 1"
        - "제 1 장", "제1장", "1장"
        - "1. 제목"
        
        Args:
            text: 전체 텍스트
        
        Returns:
            List[Dict]: [{"chapter_number": ..., "chapter_title": ..., "start_pos": ...}, ...]
        """
        # 다양한 장 패턴 정의
        patterns = [
            # 한글: "제 1장", "제1장", "제 1 장"
            r'제\s*(\d+)\s*장\s*[\.\:]?\s*([^\n]+)',
            # 영어: "CHAPTER ONE", "Chapter 1"
            r'CHAPTER\s+(ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE|TEN|ELEVEN|TWELVE|THIRTEEN|FOURTEEN|FIFTEEN|SIXTEEN|SEVENTEEN|EIGHTEEN|NINETEEN|TWENTY|\d+)[\.\:]?\s*([^\n]*)',
            # 숫자: "1장", "1. 제목"
            r'^(\d+)\s*장\s*[\.\:]?\s*([^\n]+)',
            r'^(\d+)[\.\:]\s*([^\n]+)',
        ]
        
        chapters = []
        
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.MULTILINE | re.IGNORECASE)
            
            for match in matches:
                start_pos = match.start()
                
                # 장 번호 추출
                if len(match.groups()) >= 2:
                    chapter_num_str = match.group(1)
                    chapter_title = match.group(2).strip()
                    
                    # 영어 숫자를 아라비아 숫자로 변환
                    chapter_num = self._convert_number(chapter_num_str)
                    
                    # 중복 제거 (같은 위치)
                    if not any(c['start_pos'] == start_pos for c in chapters):
                        chapters.append({
                            "chapter_number": chapter_num,
                            "chapter_title": chapter_title,
                            "start_pos": start_pos,
                            "matched_text": match.group(0)
                        })
        
        # 위치순 정렬
        chapters = sorted(chapters, key=lambda x: x['start_pos'])
        
        # 중복 제거 (비슷한 위치의 매칭)
        filtered_chapters = []
        for i, chapter in enumerate(chapters):
            # 이전 장과 위치가 너무 가까우면 스킵
            if i > 0 and chapter['start_pos'] - filtered_chapters[-1]['start_pos'] < 10:
                continue
            filtered_chapters.append(chapter)
        
        print(f"[INFO] 감지된 장 수: {len(filtered_chapters)}")
        for i, ch in enumerate(filtered_chapters[:5]):  # 처음 5개만 출력
            print(f"  - {i+1}. 제{ch['chapter_number']}장: {ch['chapter_title'][:30]}...")
        
        return filtered_chapters
    
    def _convert_number(self, num_str: str) -> int:
        """
        영어 숫자를 아라비아 숫자로 변환
        
        Args:
            num_str: 숫자 문자열
        
        Returns:
            int: 아라비아 숫자
        """
        english_numbers = {
            'ONE': 1, 'TWO': 2, 'THREE': 3, 'FOUR': 4, 'FIVE': 5,
            'SIX': 6, 'SEVEN': 7, 'EIGHT': 8, 'NINE': 9, 'TEN': 10,
            'ELEVEN': 11, 'TWELVE': 12, 'THIRTEEN': 13, 'FOURTEEN': 14, 'FIFTEEN': 15,
            'SIXTEEN': 16, 'SEVENTEEN': 17, 'EIGHTEEN': 18, 'NINETEEN': 19, 'TWENTY': 20
        }
        
        num_str_upper = num_str.upper().strip()
        
        if num_str_upper in english_numbers:
            return english_numbers[num_str_upper]
        
        try:
            return int(num_str)
        except:
            return 0
    
    def split_by_chapters(self, text: str, book_title: str) -> List[Dict[str, any]]:
        """
        텍스트를 장별로 분리
        
        Args:
            text: 전체 텍스트
            book_title: 책 제목
        
        Returns:
            List[Dict]: [{"chapter_number": ..., "chapter_title": ..., "chapter_text": ...}, ...]
        """
        # 장 감지
        chapters_info = self.detect_chapters(text)
        
        if not chapters_info:
            print(f"[WARNING] 장을 감지하지 못했습니다. 전체를 하나의 장으로 처리합니다.")
            return [{
                "book": book_title,
                "chapter_number": 1,
                "chapter_title": "전체",
                "chapter_text": text
            }]
        
        # 장별 텍스트 추출
        chapters = []
        
        for i, chapter_info in enumerate(chapters_info):
            start_pos = chapter_info['start_pos']
            
            # 다음 장의 시작 위치 (마지막 장이면 텍스트 끝)
            if i < len(chapters_info) - 1:
                end_pos = chapters_info[i + 1]['start_pos']
            else:
                end_pos = len(text)
            
            # 장 텍스트 추출
            chapter_text = text[start_pos:end_pos].strip()
            
            chapters.append({
                "book": book_title,
                "chapter_number": chapter_info['chapter_number'],
                "chapter_title": chapter_info['chapter_title'],
                "chapter_text": chapter_text
            })
            
            print(f"[INFO] 제{chapter_info['chapter_number']}장 추출: {len(chapter_text)}자")
        
        return chapters
    
    def chunk_chapters(self, chapters: List[Dict[str, any]]) -> List[Document]:
        """
        장별 텍스트를 청크로 분할하고 Document 객체 생성
        
        Args:
            chapters: 장 정보 리스트
        
        Returns:
            List[Document]: Document 객체 리스트
        """
        all_documents = []
        total_chunks = 0
        
        for chapter in chapters:
            book = chapter['book']
            chapter_number = chapter['chapter_number']
            chapter_title = chapter['chapter_title']
            chapter_text = chapter['chapter_text']
            
            # 텍스트 청킹
            chunks = self.text_splitter.split_text(chapter_text)
            
            print(f"[INFO] 제{chapter_number}장 청킹: {len(chunks)}개 청크 생성")
            
            # Document 객체 생성
            for i, chunk in enumerate(chunks):
                metadata = {
                    "book": book,
                    "chapter_number": chapter_number,
                    "chapter_title": chapter_title,
                    "chunk_index": i,
                    "total_chunks_in_chapter": len(chunks)
                }
                
                doc = Document(
                    page_content=chunk,
                    metadata=metadata
                )
                
                all_documents.append(doc)
            
            total_chunks += len(chunks)
        
        print(f"[INFO] 총 청크 수: {total_chunks}")
        return all_documents
    
    def process_book(self, text: str, book_title: str) -> List[Document]:
        """
        책 전체 처리 (장 분리 + 청킹)
        
        Args:
            text: 전체 텍스트
            book_title: 책 제목
        
        Returns:
            List[Document]: Document 객체 리스트
        """
        print(f"\n[처리 중] {book_title}")
        print("-" * 80)
        
        # 장 분리
        chapters = self.split_by_chapters(text, book_title)
        
        # 청킹
        documents = self.chunk_chapters(chapters)
        
        print(f"[완료] {book_title}: 총 {len(chapters)}개 장, {len(documents)}개 청크")
        print("-" * 80)
        
        return documents


def main():
    """테스트용 메인 함수"""
    from preprocess import TextPreprocessor
    
    # 데이터 로드
    data_dir = r"c:\Users\ansck\Desktop\Project\3rd_project\data\cleaned_data"
    preprocessor = TextPreprocessor(data_dir)
    processed_files = preprocessor.preprocess_all(remove_title=False)
    
    # 챕터 분리기 생성
    splitter = ChapterSplitter(chunk_size=800, chunk_overlap=150)
    
    # 각 책 처리
    all_documents = []
    
    for file_info in processed_files[:1]:  # 테스트: 첫 번째 책만
        book_title = preprocessor.extract_book_title(file_info['filename'])
        documents = splitter.process_book(file_info['text'], book_title)
        all_documents.extend(documents)
    
    # 결과 출력
    print(f"\n[총 결과]")
    print(f"  - 총 Document 수: {len(all_documents)}")
    print(f"\n[샘플 Document]")
    if all_documents:
        sample_doc = all_documents[0]
        print(f"  - 내용: {sample_doc.page_content[:100]}...")
        print(f"  - 메타데이터: {sample_doc.metadata}")


if __name__ == "__main__":
    main()
