from data_loaders import load_all_disaster_jsons, load_shelter_csv
from documents import json_to_documents, csv_to_documents

# CSV 로드 (data_loaders.py 사용)
shelter_data = load_shelter_csv("shelter.csv")
# JSON 로드 (data_loaders.py 사용)
json_files = [
    'natural_disaster/typoon.json',
    'natural_disaster/flood.json',
    'natural_disaster/Tsunami.json',
    'natural_disaster/landslide.json',
    'natural_disaster/storm.json',
    'natural_disaster/volcanic ash.json',
    'natural_disaster/Volcanic Eruption.json',
    'natural_disaster/earthquake.json',
    'social_disaster/fire.json',
    'social_disaster/dam.json',
    'social_disaster/gas.json',
    'social_disaster/radiation.json',
    'social_disaster/wildfire.json'
    ]
disaster_datas = load_all_disaster_jsons(json_files)

# Document 변환 (documents.py 사용)
disaster_documents = json_to_documents(disaster_datas)
shelter_documents = csv_to_documents(shelter_data)

print(f"\n전체 Document 개수: {len(shelter_documents) + len(disaster_documents)}개")

# 임베딩 및 벡터 DB 생성 (embedding_and_vectordb.py 사용)
from embedding_and_vectordb import create_embeddings_and_vectordb
all_documents = shelter_documents + disaster_documents
create_embeddings_and_vectordb(all_documents)