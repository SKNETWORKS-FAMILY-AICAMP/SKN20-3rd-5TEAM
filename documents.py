"""
Document 생성 및 청킹 모듈
CSV/JSON 데이터를 LangChain Document로 변환하고 청킹하는 함수
"""

# 필수 라이브러리 임포트
from langchain_core.documents import Document
import pandas as pd
from typing import List, Any

# CSV 파일 Document 변환 및 청킹 함수
def csv_to_documents(shelter_data: pd.DataFrame) -> List[Document]:
    """
        민방위 대피 시설 csv를 LangChain Document로 변환

        Args:
        - shelter_data (pd.DataFrame): 민방위 대피 시설 데이터프레임

        Returns:
        - List[Document]: 변환된 Document 리스트
        """
    
    documents = []

    for _, row in shelter_data.iterrows():
        page_content = (
            f"민방위 대피 시설 {row['시설명']}은 {row['도로명전체주소']}에 위치해 있으며, "
            f"{row['시설구분']} 시설입니다. 위치는 {row['시설위치(지상/지하)']}이고, "
            f"시설 면적은 {row['시설면적(㎡)']}이며, 최대 {row['최대수용인원']}명을 수용할 수 있습니다."
        )

        metadata = {
            "type": "shelter",
            "source": "shelter.csv",
            "management_code": str(row["관리번호"]),
            "operating_status": str(row["운영상태"]),
            "facility_name": str(row["시설명"]),
            "facility_type": str(row["시설구분"]),
            "address": str(row["도로명전체주소"]),
            "postal_code": int(row["도로명우편번호"]),
            "shelter_type": str(row["시설위치(지상/지하)"]),
            "capacity": int(row["최대수용인원"]),
            "lat": float(row["위도(EPSG4326)"]),
            "lon": float (row["경도(EPSG4326)"])
        }

        documents.append(Document(page_content=page_content, metadata=metadata))

    print(f"\n대피소: 총 {len(documents)}개 Document 생성 완료")
    return documents    

# JSON 파일 Document 변환 및 청킹 함수
def parse_node(
    node: Any,
    path: List[str],
    disaster_type: str,
    disaster_name: str,
    situation: str,
    source_file: str,
    documents: List[Document]
) -> None:
    """재귀적으로 JSON 노드를 탐색하여 Document 생성"""
    
    # 딕셔너리인 경우
    if isinstance(node, dict):
        # Leaf Node 판단: '세부사항', '내용', '이유' 등이 있으면 Document 생성
        has_content = any(key in node for key in ['세부사항', '내용', '주의사항', '이유', '신고처'])
        
        if has_content:
            # page_content 생성
            breadcrumbs = " > ".join(path)
            content_parts = [f"{breadcrumbs}\n{'-' * 50}"]
            
            # 세부사항 처리
            if '세부사항' in node:
                details = node['세부사항']
                if isinstance(details, list):
                    content_parts.append("\n세부사항:")
                    content_parts.extend([f"- {item}" for item in details])
                else:
                    content_parts.append(f"\n세부사항:\n{details}")
            
            # 주의사항 처리
            if '주의사항' in node:
                caution = node['주의사항']
                if isinstance(caution, list):
                    content_parts.append("\n\n주의사항:")
                    content_parts.extend([f"- {item}" for item in caution])
                else:
                    content_parts.append(f"\n\n주의사항:\n{caution}")
            
            # 내용 처리
            if '내용' in node:
                content_parts.append(f"\n\n내용:\n{node['내용']}")
            
            # 이유 처리
            if '이유' in node:
                reasons = node['이유']
                if isinstance(reasons, list):
                    content_parts.append("\n\n이유:")
                    content_parts.extend([f"- {item}" for item in reasons])
            
            # 신고처 처리 (특수 케이스)
            if '신고처' in node:
                contacts = node['신고처']
                if isinstance(contacts, list):
                    content_parts.append("\n\n신고처:")
                    for contact in contacts:
                        if isinstance(contact, dict):
                            contact_info = f"- {contact.get('기관', '')}"
                            if contact.get('연락처'):
                                contact_info += f": {contact['연락처']}"
                            if contact.get('방법'):
                                contact_info += f" ({contact['방법']})"
                            content_parts.append(contact_info)
            
            # 보호자 행동요령 처리
            if '보호자_행동요령' in node:
                guardian_actions = node['보호자_행동요령']
                if isinstance(guardian_actions, list):
                    content_parts.append("\n\n보호자 행동요령:")
                    content_parts.extend([f"- {item}" for item in guardian_actions])
            
            # 평소 준비사항 처리
            if '평소_준비사항' in node:
                prep = node['평소_준비사항']
                if isinstance(prep, list):
                    content_parts.append("\n\n평소 준비사항:")
                    content_parts.extend([f"- {item}" for item in prep])
            
            # 행동요령 처리
            if '행동요령' in node and isinstance(node['행동요령'], list):
                content_parts.append("\n\n행동요령:")
                content_parts.extend([f"- {item}" for item in node['행동요령']])
            
            page_content = "\n".join(content_parts)
            
            # metadata 생성
            metadata = {
                "type": "disaster_guideline",
                "source": source_file,
                "category": disaster_type,
                "keyword": disaster_name,
                "situation": situation,
                "path": " > ".join(path)
            }
            
            # 제목, 번호 등 추가 메타데이터
            if '제목' in node:
                metadata['title'] = node['제목']
            if '번호' in node:
                metadata['number'] = node['번호']
            
            # Document 생성
            documents.append(Document(
                page_content=page_content,
                metadata=metadata
            ))
        
        # 재귀 탐색 (세부사항 등의 키는 제외)
        for key, value in node.items():
            if key not in ['세부사항', '내용', '주의사항', '이유', '번호', '제목', '신고처', 
                          '보호자_행동요령', '평소_준비사항', '행동요령']:
                new_path = path + [node.get('제목', key)]
                parse_node(value, new_path, disaster_type, disaster_name, situation, source_file, documents)
    
    # 리스트인 경우
    elif isinstance(node, list):
        for item in node:
            parse_node(item, path, disaster_type, disaster_name, situation, source_file, documents)


def json_to_documents(disaster_datas: dict) -> List[Document]:
    """
    재난 행동요령 JSON 데이터를 LangChain Document로 변환
    
    Args:
    - disaster_datas (dict): load_all_disaster_jsons()에서 반환된 딕셔너리
                             {파일명: JSON 데이터} 형태
    
    Returns:
    - List[Document]: 변환된 Document 리스트
    """
    all_documents = []
    
    for filename, data in disaster_datas.items():
        if not data:
            print(f"⚠️ {filename}: 데이터가 비어있습니다.")
            continue
        
        print(f"\n📄 처리 중: {filename}")
        documents = []
        
        # 공통 메타 정보 추출
        disaster_type = data.get('재난유형', '')
        disaster_name = data.get('재난명', '')
        source_file = filename
        
        # 행동요령 섹션 처리
        action_guidelines = data.get('행동요령', {})
        
        # 1차 단계(situation) 순회
        for situation_key, situation_data in action_guidelines.items():
            situation_title = situation_data.get('제목', situation_key)
            initial_path = [disaster_name, situation_title]
            
            parse_node(
                node=situation_data,
                path=initial_path,
                disaster_type=disaster_type,
                disaster_name=disaster_name,
                situation=situation_key,
                source_file=source_file,
                documents=documents
            )
        
        all_documents.extend(documents)
        print(f"총 {len(documents)}개 Document 생성")
    
    print(f"\n대피요령: 총 {len(all_documents)}개 Document 생성 완료")
    return all_documents