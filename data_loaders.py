"""
데이터 로더 모듈
CSV(민방위 대피시설)와 JSON(재난 행동요령)을 로드하는 함수
"""

# 필수 라이브러리 임포트
import pandas as pd
import json
from typing import List

# CSV 파일 로드 함수
def load_shelter_csv(csv_file: str, data_dir: str = "data") -> pd.DataFrame:
    """
    CSV 파일을 로드하는 함수

    Args
    - file_path: CSV 파일 경로
    
    Returns
    - pandas DataFrame
    """
    file_path = f"{data_dir}/{csv_file}"
    shelter_data = pd.read_csv(file_path, encoding="utf-8")
    return shelter_data

# json 파일 로드 함수
def load_disaster_json(path: str) -> dict:
    """
    재난 행동 요령 JSON 파일을 로드

    Args
    - path: json 파일 경로

    Returns
    - json 데이터 딕셔너리
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# 모든 재난 JSON 파일 로드
def load_all_disaster_jsons(json_files: List, data_dir: str = "data") -> dict:
    """
    모든 재난 행동요령 JSON 파일을 로드
    
    Args
    - 모든 재난 행동요령 JSON 파일 리스트

    Returns:
        파일명을 키로, JSON 데이터를 값으로 하는 딕셔너리
    """
    
    disaster_datas = {}

    for filename in json_files:
        file_path = f"{data_dir}/{filename}"
        disaster_datas[filename] = load_disaster_json(file_path)
        print(f"✓ {filename} 로드 완료")
    
    return disaster_datas


