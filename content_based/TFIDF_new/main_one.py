import os
import argparse
import numpy as np
import random
import json
from openai import OpenAI
from dotenv import load_dotenv
from search.db_search import DatabaseHandler
from search.dense_retriever import DenseRetriever
from search.dataset_json import goalDatasetjson
from recomm.recomm import Recommendation

# 환경 변수 로드
load_dotenv()
DB_PWD = os.getenv('DB_PWD')
OPENAI_API_KEY = os.getenv('PF_OPEN_AI_KEY')

# 🔹 추천 결과 일관성 유지 (TF-IDF 기반 추천 시)
def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)

def get_args():
    parser = argparse.ArgumentParser(description="강의 추천 시스템(TF-IDF 기반)")
    parser.add_argument("--query", type=str, required=True, help="사용자 입력 쿼리")
    parser.add_argument("--expanded_query", type=str, default="", help="확장된 텍스트 (선택 사항)")
    parser.add_argument("--dept", type=str, choices=["yes", "no"], required=True, default="no", help="학과 지정할지 여부 (yes / no)")
    parser.add_argument("--departments", type=str, default="", help="학과 리스트 (쉼표로 구분)")
    
    parser.add_argument("--department_path", type=str, default="/root/ai_curri/content_based/TFIDF_new/data/depart_info.json", help="학과 소개글 임베딩 할 텍스트")
    parser.add_argument("--batch_size", type=int, default=10, help="검색에 사용되는 batch size")

    return parser.parse_args()

def main():
    args = get_args()

    set_seed()
    
    db_config = {
        "host": "210.117.181.113",
        "port": 3311,
        "user": "root",
        "password": DB_PWD,
        "database": "nll_third",
        "charset": "utf8mb4"
    }
    
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    db_handler = DatabaseHandler(
        host=db_config["host"],
        port=db_config["port"],
        user=db_config["user"],
        password=db_config["password"],
        database=db_config["database"],
        charset=db_config["charset"]
    )
    
    db_handler.connect()
    print("Fetching department data...")
    
    # 🔹 학과 벡터 저장 (최초 1회 실행)
    if not os.path.exists("data/dept_vectorizer.pkl"):
        dept_info = goalDatasetjson(json.load(open(args.department_path)))
        retriever = DenseRetriever(client, args, dept_info)
        retriever.doc_embedding()
    else:
        retriever = DenseRetriever(client, args)
        retriever.doc_embedding()
    
    recomm = Recommendation(db_handler)

    # 🔹 교과목 벡터 저장 (최초 1회 실행)
    if not os.path.exists("data/tfidf_vectors.npz") or not os.path.exists("data/course_vectorizer.pkl"):
        recomm.preprocess_and_save()
        
    # 🔹 쿼리 설정
    query = args.query
    expanded_query = args.expanded_query if args.expanded_query else query  # 확장된 쿼리 없으면 원본 사용
    query_info = query if args.expanded_query == "" else expanded_query

    # 🔹 학과 정보 설정
    if args.dept == "no":
        query_embedding = retriever.query_embedding(query_info)
        department_list = retriever.retrieve(query_embedding)
        department_list = [dept["department_name"] for dept in department_list]
        print(f"학과 검색 결과 ::: {department_list} \n\n")
    else:
        department_list = [dept.strip() for dept in args.departments.split(",")]

    # 🔹 강의 추천 실행
    recommendations = recomm.recommend_classes_dept(query_info, department_list)

    # 🔹 JSON 저장 형식 구성
    result_data = {
        "meta_info": {
            "user_query": query,
            "expanded_query": expanded_query if args.expanded_query else None,
            "given_departments": department_list if args.dept == "yes" else None,
            "selected_departments": department_list if args.dept == "no" else None
        },
        "recommendations": recommendations  # (최종 결과)
    }

    # 🔹 결과 저장 경로 설정
    save_folder_dept = "original" if args.dept == "no" else "given_dept"
    output_dir = f'/root/ai_curri/content_based/TFIDF_new/output'
    os.makedirs(output_dir, exist_ok=True)
    json_file = f'{output_dir}/스마트글로벌무역_recommendations_{("yes" if args.expanded_query else "no")}.json'
    
    # 🔹 JSON 파일로 저장 (학과 정보 포함)
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(result_data, f, ensure_ascii=False, indent=4)

    print(f"✅ 추천 결과가 {json_file} 파일로 저장됨!")

    # 🔹 추천 결과 출력
    print("\n🔹 추천 강의 목록:")
    for rec in recommendations:
        print(f"📌 [{rec['student_grade']}학년 {rec['semester']}학기] {rec['name']} "
              f"(유사도: {rec['similarity']:.2f}) - {rec['department_name']}")

if __name__ == "__main__":
    main()