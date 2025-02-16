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
    def __init__(self, db_handler):
        self.VECTOR_FILE = "data/tfidf_vectors.npz"
        self.PICKLE_FILE = "data/course_vectorizer.pkl"
        self.db_handler = db_handler

    # 🔹 벡터를 미리 저장하는 함수 (강의명 + 수업 목표 포함)
    def preprocess_and_save(self):
        classes_info = self.db_handler.fetch_filtered_classes()
        vectorizer = TfidfVectorizer(max_features=5000, stop_words=None)

        # 리스트의 딕셔너리에서 "full_text" 값만 추출하여 리스트로 변환
        full_texts = [item["full_text"] for item in classes_info]

        # TF-IDF 벡터화 수행
        tfidf_matrix = vectorizer.fit_transform(full_texts)

        # 벡터 저장
        class_ids = [item["id"] for item in classes_info]  # "id" 값만 리스트로 추출
        np.savez(self.VECTOR_FILE, tfidf_matrix=tfidf_matrix.toarray(), class_ids=np.array(class_ids))

        # 벡터라이저 저장
        with open(self.PICKLE_FILE, "wb") as f:
            pickle.dump(vectorizer, f)

        print("✅ TF-IDF 벡터 및 모델 저장 완료!")


    # 🔹 사용자 입력을 기반으로 추천하는 함수 (학년/학기별 최대 5개)
    def recommend_classes(self, user_query, top_n=5):
        # 저장된 TF-IDF 불러오기
        if not os.path.exists(self.VECTOR_FILE) or not os.path.exists(self.PICKLE_FILE):
            print("⚠️ 벡터 파일이 없습니다. 먼저 'preprocess_and_save()'를 실행하세요.")
            return
        
        data = np.load(self.VECTOR_FILE)  # 저장된 벡터 로드
        tfidf_matrix = data["tfidf_matrix"]  # 강의 벡터
        class_ids = data["class_ids"]  # 강의 ID
        
        with open(self.PICKLE_FILE, "rb") as f:
            vectorizer = pickle.load(f)  # 저장된 TF-IDF 벡터라이저 로드

        # 사용자 쿼리를 벡터화
        query_vector = vectorizer.transform([user_query]).toarray()

        # 코사인 유사도 계산
        similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()

        # 🔹 강의 데이터 다시 불러오기 (list of dictionaries)
        classes_info = self.db_handler.fetch_filtered_classes()

        # 🔹 similarity 값을 리스트 형태로 추가
        for idx, item in enumerate(classes_info):
            item["similarity"] = similarities[idx]  # 각 강의에 해당하는 유사도 값 추가

        # 🔹 유사도 0.1 이상인 강의만 필터링
        filtered_classes = [item for item in classes_info if item["similarity"] >= 0.1]

        # # 🔹 학년/학기별로 최대 5개씩 추천 (2학년 1학기 ~ 4학년 2학기)
        # recommended = []
        # for year in range(2, 5):  # 2학년 ~ 4학년
        #     for semester in range(1, 3):  # 1학기, 2학기
        #         subset = [item for item in filtered_classes if item["student_grade"] == year and item["semester"] == semester]
                
        #         # 🔹 유사도 기준 정렬
        #         subset = sorted(subset, key=lambda x: x["similarity"], reverse=True)

        #         # 🔹 최대 `top_n`개 추천
                # recommended.extend(subset[:top_n])
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
            
        print(f"학과 이름들 ::: {department_names}\n\n")
            
        # 저장된 TF-IDF 불러오기
        if not os.path.exists(self.VECTOR_FILE) or not os.path.exists(self.PICKLE_FILE):
            print("⚠️ 벡터 파일이 없습니다. 먼저 'preprocess_and_save()'를 실행하세요.")
            return
        
        data = np.load(self.VECTOR_FILE)  # 저장된 벡터 로드
        tfidf_matrix = data["tfidf_matrix"]  # 강의 벡터
        class_ids = data["class_ids"]  # 강의 ID
        
        with open(self.PICKLE_FILE, "rb") as f:
            vectorizer = pickle.load(f)  # 저장된 TF-IDF 벡터라이저 로드
            
        # 🔹 주어진 학과에 해당하는 강의만 필터링
        classes_info = self.db_handler.fetch_filtered_classes_dept(department_names)

        # 🔹 학과 필터링된 강의 ID 추출
        filtered_class_ids = [item["id"] for item in classes_info]  # ✅ 리스트에서 "class_id" 값만 추출
        
        mask = np.isin(class_ids, filtered_class_ids)  # 학과 내 강의 ID 필터링
        
        # 🔹 필터링된 TF-IDF 벡터만 유지
        filtered_tfidf_matrix = tfidf_matrix[mask]
        
        # 🔹 사용자 쿼리를 벡터화
        query_vector = vectorizer.transform([user_query]).toarray()

        # 🔹 코사인 유사도 계산
        similarities = cosine_similarity(query_vector, filtered_tfidf_matrix).flatten()
        
        # 🔹 유사도 값을 학과 강의 데이터에 적용
        for idx, item in enumerate(classes_info):  # ✅ 리스트에서 직접 추가
            item["similarity"] = similarities[idx]

        # 🔹 1. 유사도 0.1 이상인 강의만 필터링
        classes_info = [item for item in classes_info if item["similarity"] >= 0.1]  # ✅ 리스트 컴프리헨션 사용

        # # 🔹 2. 학년/학기별로 최대 5개씩 추천 (2학년 1학기 ~ 4학년 2학기)
        # recommended = []
        # for year in range(2, 5):  # 2학년 ~ 4학년
        #     for semester in range(1, 3):  # 1학기, 2학기
        #         subset = [item for item in classes_info if item["student_grade"] == year and item["semester"] == semester]
                
        #         # 🔹 유사도 기준 정렬
        #         subset = sorted(subset, key=lambda x: x["similarity"], reverse=True)

        #         # 🔹 최대 `top_n`개 추천
        #         #recommended.extend(subset[:top_n])
#                    recommended.extend([
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