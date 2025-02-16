import os
import numpy as np
import pickle
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 🔹 Seed 고정 (TF-IDF 벡터 생성 일관성 유지)
def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)

set_seed()  # 🔥 실행 시 seed 고정

class Recommendation:
    def __init__(self, db_handler, client):
        self.VECTOR_FILE = "data/gpt_embeddings.npz"
        self.db_handler = db_handler
        self.client = client

    # 🔹 벡터를 미리 저장하는 함수 (강의명 + 수업 목표 포함)
    def preprocess_and_save(self):
        classes_info = self.db_handler.fetch_filtered_classes()
        
        # 🔹 GPT 임베딩 생성 (모든 강의)
        embeddings = np.array([self.get_gpt_embedding(item["full_text"]) for item in classes_info])
        
        # 🔹 벡터 저장
        class_ids = np.array([item["id"] for item in classes_info])  # "id" 값 리스트로 변환
        np.savez(self.VECTOR_FILE, embeddings=embeddings, class_ids=class_ids)

        print("✅ GPT 임베딩 벡터 저장 완료!")
        
    # 🔹 GPT-4o 임베딩 생성 함수
    def get_gpt_embedding(self, text):
        response = self.client.embeddings.create(
            model="text-embedding-3-large",
            input=text
        )
        return np.array(response.data[0].embedding)  # (1536,) 차원의 벡터 반환


    # 🔹 사용자 입력을 기반으로 추천하는 함수 (학년/학기별 최대 5개)
    def recommend_classes(self, user_query, top_n=5):
        # 저장된 GPT 임베딩 불러오기
        if not os.path.exists(self.VECTOR_FILE):
            print("⚠️ 벡터 파일이 없습니다. 먼저 'preprocess_and_save()'를 실행하세요.")
            return
        
        data = np.load(self.VECTOR_FILE)  # 저장된 벡터 로드
        embeddings = data["embeddings"]  # 강의 벡터 (예상 shape: [N, 1536])
        class_ids = data["class_ids"]  # 강의 ID
        
        # 🔹 사용자 쿼리를 GPT 임베딩으로 변환
        query_vector = self.get_gpt_embedding(user_query).reshape(1, -1)  # (1, 1536)

        # 🔹 코사인 유사도 계산
        similarities = cosine_similarity(query_vector, embeddings).flatten()

        # 🔹 강의 데이터 다시 불러오기 (list of dictionaries)
        classes_info = self.db_handler.fetch_filtered_classes()

        # 🔹 similarity 값을 리스트 형태로 추가
        for idx, item in enumerate(classes_info):
            item["similarity"] = similarities[idx]  # 각 강의에 해당하는 유사도 값 추가

        # 🔹 유사도 0.1 이상인 강의만 필터링
        filtered_classes = [item for item in classes_info if item["similarity"] >= 0.1]

        # # 🔹 학년/학기별로 최대 5개씩 추천(2학년 1학기 ~ 4학년 2학기)
        # recommended = []
        # for year in range(2, 5):  # 2학년 ~ 4학년
        #     for semester in range(1, 3):  # 1학기, 2학기
        #         subset = [item for item in filtered_classes if item["student_grade"] == year and item["semester"] == semester]
                
        #         # 🔹 유사도 기준 정렬
        #         subset = sorted(subset, key=lambda x: x["similarity"], reverse=True)

        #         # 🔹 최대 `top_n`개 추천
        #         recommended.extend(subset[:top_n])
#                     recommended.extend([
#     {key: item[key] for key in ["id", "name", "student_grade", "semester", "department_name", "similarity"]}
#     for item in subset[:top_n]
# ])

                
        # 🔹 학년/학기별로 최대 5개씩 추천(1학년 2학기 ~ 4학년 2학기)
        recommended = []
        for year in range(1, 5):  # 1학년 ~ 4학년
            for semester in range(1, 3):  # 1학기, 2학기
                # if year == 1 and semester == 1:
                #     continue  # 1학년 1학기는 건너뜀

                subset = [item for item in filtered_classes if item["student_grade"] == year and item["semester"] == semester]
                
                # 🔹 유사도 기준 정렬
                subset = sorted(subset, key=lambda x: x["similarity"], reverse=True)

                # 🔹 최대 `top_n`개 추천
                # recommended.extend(subset[:top_n])
                recommended.extend([
    {key: item[key] for key in ["id", "name", "student_grade", "semester", "department_name", "similarity"]}
    for item in subset[:top_n]
])

        return recommended
    
    # 학과 주어졌을 때, 교과목 추천
    def recommend_classes_dept(self, user_query, department_names, top_n=5):
        """
        특정 학과(들)에 대해 학년/학기별로 최대 5개씩 강의를 추천하는 함수.
        """
        
        # 🔥 department_names가 문자열이면 리스트로 변환
        if isinstance(department_names, str):
            department_names = [dept.strip() for dept in department_names.split(",")]
            
        # 저장된 GPT 임베딩 불러오기
        if not os.path.exists(self.VECTOR_FILE):
            print("⚠️ 벡터 파일이 없습니다. 먼저 'preprocess_and_save()'를 실행하세요.")
            return
        
        data = np.load(self.VECTOR_FILE)  # 저장된 벡터 로드
        embeddings = data["embeddings"]  # 강의 벡터 (예상 shape: [N, 1536])
        class_ids = data["class_ids"]  # 강의 ID
            
        # 🔹 주어진 학과에 해당하는 강의만 필터링
        classes_info = self.db_handler.fetch_filtered_classes_dept(department_names)

        # 🔹 학과 필터링된 강의 ID 추출
        filtered_class_ids = [item["id"] for item in classes_info]  # ✅ 리스트에서 "class_id" 값만 추출
        mask = np.isin(class_ids, filtered_class_ids)  # 학과 내 강의 ID 필터링
        
        filtered_embeddings = embeddings[mask]
        filtered_class_ids = class_ids[mask]
        
        # 🔹 사용자 쿼리를 GPT 임베딩으로 변환
        query_vector = self.get_gpt_embedding(user_query).reshape(1, -1)  # (1, 1536)

        # 🔹 코사인 유사도 계산
        similarities = cosine_similarity(query_vector, filtered_embeddings).flatten()

        # 🔹 유사도 값을 학과 강의 데이터에 적용
        for idx, item in enumerate(classes_info):  # ✅ 리스트에서 직접 추가
            item["similarity"] = similarities[idx]

        # 🔹 1. 유사도 0.1 이상인 강의만 필터링
        classes_info = [item for item in classes_info if item["similarity"] >= 0.1]  # ✅ 리스트 컴프리헨션 사용

        # # 🔹 2. 학년/학기별로 최대 5개씩 추천(2학년 1학기 ~ 4학년 2학기)
        # recommended = []
        # for year in range(2, 5):  # 2학년 ~ 4학년
        #     for semester in range(1, 3):  # 1학기, 2학기
        #         subset = [item for item in classes_info if item["student_grade"] == year and item["semester"] == semester]
                
        #         # 🔹 유사도 기준 정렬
        #         subset = sorted(subset, key=lambda x: x["similarity"], reverse=True)

        #         # 🔹 최대 `top_n`개 추천
        #         #recommended.extend(subset[:top_n])
#                 recommended.extend([
#     {key: item[key] for key in ["id", "name", "student_grade", "semester", "department_name", "similarity"]}
#     for item in subset[:top_n]
# ])

                
                
        # 🔹 학년/학기별로 최대 5개씩 추천(1학년 1학기 ~ 4학년 2학기)
        recommended = []
        for year in range(1, 5):  # 1학년 ~ 4학년
            for semester in range(1, 3):  # 1학기, 2학기
                # if year == 1 and semester == 1:
                #     continue  # 1학년 1학기는 건너뜀

                subset = [item for item in classes_info if item["student_grade"] == year and item["semester"] == semester]
                
                # 🔹 유사도 기준 정렬
                subset = sorted(subset, key=lambda x: x["similarity"], reverse=True)

                # 🔹 최대 `top_n`개 추천
                # recommended.extend(subset[:top_n])
                recommended.extend([
    {key: item[key] for key in ["id", "name", "student_grade", "semester", "department_name", "similarity"]}
    for item in subset[:top_n]
])

        return recommended