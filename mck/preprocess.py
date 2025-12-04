"""
preprocess.py
TXT 파일 로드 및 전처리 모듈

주요 기능:
1. 디렉토리 내 모든 TXT 파일 로드
2. UTF-8 디코딩 오류 처리
3. 불규칙한 개행/공백 정리
4. 첫 줄(책 제목) 제거
"""

import os
import re
from typing import List, Dict, Tuple
from pathlib import Path


class TextPreprocessor:
    """TXT 파일 로드 및 전처리 클래스"""
    
    def __init__(self, data_dir: str):
        """
        Args:
            data_dir: TXT 파일이 있는 디렉토리 경로
        """
        self.data_dir = Path(data_dir)
        print(f"[INFO] 전처리기 초기화: {self.data_dir}")
    
    def load_txt_files(self) -> List[Tuple[str, str]]:
        """
        디렉토리 내 모든 TXT 파일 로드
        
        Returns:
            List[Tuple[str, str]]: (파일명, 원본 텍스트) 튜플 리스트
        """
        txt_files = []
        
        if not self.data_dir.exists():
            print(f"[ERROR] 디렉토리가 존재하지 않습니다: {self.data_dir}")
            return txt_files
        
        # TXT 파일 찾기
        for file_path in self.data_dir.glob("*.txt"):
            try:
                # UTF-8 디코딩 시도
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    txt_files.append((file_path.name, content))
                    print(f"[SUCCESS] 파일 로드 완료: {file_path.name} ({len(content)} 글자)")
            
            except UnicodeDecodeError:
                # UTF-8 실패 시 다른 인코딩 시도
                try:
                    with open(file_path, 'r', encoding='cp949') as f:
                        content = f.read()
                        txt_files.append((file_path.name, content))
                        print(f"[SUCCESS] 파일 로드 완료 (cp949): {file_path.name}")
                except Exception as e:
                    print(f"[ERROR] 파일 로드 실패: {file_path.name} - {str(e)}")
            
            except Exception as e:
                print(f"[ERROR] 파일 로드 실패: {file_path.name} - {str(e)}")
        
        print(f"[INFO] 총 {len(txt_files)}개 파일 로드 완료")
        return txt_files
    
    def remove_first_line(self, text: str) -> str:
        """
        첫 줄(책 제목) 제거
        
        Args:
            text: 원본 텍스트
        
        Returns:
            str: 첫 줄이 제거된 텍스트
        """
        lines = text.split('\n')
        if len(lines) > 1:
            # 첫 줄이 책 제목일 가능성이 높음
            first_line = lines[0].strip()
            if first_line and len(first_line) < 100:  # 제목은 보통 짧음
                print(f"[INFO] 첫 줄 제거: {first_line[:50]}...")
                return '\n'.join(lines[1:])
        return text
    
    def clean_whitespace(self, text: str) -> str:
        """
        불규칙한 개행/공백 정리
        
        Args:
            text: 원본 텍스트
        
        Returns:
            str: 정리된 텍스트
        """
        # 연속된 공백을 하나로
        text = re.sub(r' +', ' ', text)
        
        # 연속된 개행을 최대 2개로 제한 (문단 구분 유지)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # 각 줄의 앞뒤 공백 제거
        lines = text.split('\n')
        cleaned_lines = [line.strip() for line in lines]
        text = '\n'.join(cleaned_lines)
        
        # 빈 줄 연속 제거
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
        
        return text.strip()
    
    def preprocess_text(self, text: str, remove_title: bool = True) -> str:
        """
        텍스트 전처리 (첫 줄 제거 + 공백 정리)
        
        Args:
            text: 원본 텍스트
            remove_title: 첫 줄(제목) 제거 여부
        
        Returns:
            str: 전처리된 텍스트
        """
        # 첫 줄 제거
        if remove_title:
            text = self.remove_first_line(text)
        
        # 공백 정리
        text = self.clean_whitespace(text)
        
        return text
    
    def preprocess_all(self, remove_title: bool = True) -> List[Dict[str, str]]:
        """
        모든 TXT 파일 로드 및 전처리
        
        Args:
            remove_title: 첫 줄(제목) 제거 여부
        
        Returns:
            List[Dict]: [{"filename": ..., "text": ...}, ...]
        """
        print("\n" + "="*80)
        print("[STEP 1] TXT 파일 로드 및 전처리 시작")
        print("="*80 + "\n")
        
        # 파일 로드
        raw_files = self.load_txt_files()
        
        # 전처리
        processed_files = []
        for filename, raw_text in raw_files:
            print(f"\n[처리 중] {filename}")
            cleaned_text = self.preprocess_text(raw_text, remove_title=remove_title)
            
            processed_files.append({
                "filename": filename,
                "text": cleaned_text,
                "char_count": len(cleaned_text),
                "line_count": len(cleaned_text.split('\n'))
            })
            
            print(f"  - 원본 글자 수: {len(raw_text)}")
            print(f"  - 처리 후 글자 수: {len(cleaned_text)}")
            print(f"  - 줄 수: {len(cleaned_text.split('\n'))}")
        
        print("\n" + "="*80)
        print(f"[완료] 총 {len(processed_files)}개 파일 전처리 완료")
        print("="*80 + "\n")
        
        return processed_files
    
    def extract_book_title(self, filename: str) -> str:
        """
        파일명에서 책 제목 추출
        
        Args:
            filename: 파일명
        
        Returns:
            str: 책 제목
        """
        # "cleaned_" 제거
        title = filename.replace("cleaned_", "")
        # ".txt" 제거
        title = title.replace(".txt", "")
        return title


def main():
    """테스트용 메인 함수"""
    # 데이터 디렉토리 설정
    data_dir = r"c:\Users\ansck\Desktop\Project\3rd_project\data\cleaned_data"
    
    # 전처리기 생성
    preprocessor = TextPreprocessor(data_dir)
    
    # 전처리 실행
    processed_files = preprocessor.preprocess_all(remove_title=True)
    
    # 결과 출력
    print("\n[전처리 결과 요약]")
    for file_info in processed_files:
        book_title = preprocessor.extract_book_title(file_info['filename'])
        print(f"  - {book_title}: {file_info['char_count']:,}자, {file_info['line_count']:,}줄")


if __name__ == "__main__":
    main()
