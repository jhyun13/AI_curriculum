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
    parser.add_argument("--query_path", type=str, required=True, help="사용자 입력 쿼리 파일 경로")
    parser.add_argument("--expand", type=str, choices=["yes", "no"], required=True, default="no", help="텍스트를 확장할지 여부 (yes / no)")
    parser.add_argument("--dept", type=str, choices=["yes", "no", "auto"], required=True, default="no", help="학과 지정할지 여부 (yes / no)")
    # parser.add_argument("--save_path", type=str, required=True, help="결과 저장할 json 파일 경로")
    
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
        
    with open(args.query_path, "r", encoding="utf-8") as file:
        input_query = json.load(file)

    recomm = Recommendation(db_handler)

    # 🔹 교과목 벡터 저장 (최초 1회 실행)
    if not os.path.exists("data/tfidf_vectors.npz") or not os.path.exists("data/course_vectorizer.pkl"):
        recomm.preprocess_and_save()
        
    # 각 전공별로 query와 expanded_query 처리
    for major, details in input_query.items():
        print(f"Processing Major: {major}")
        departments = details.get("departments", "")
        
        for idx, query_data in enumerate(details.get("query_list", [])):
            query = query_data.get("query", "")
            expanded_query = query_data.get("expanded_query", "")
            
            # 쿼리 확장 옵션 선택
            query_info = query if args.expand == "no" else expanded_query
            query_embbeding = retriever.query_embedding(query_info)
            
            # 학과 주어지는 옵션 선택
            if args.dept == "auto":
                department_list = retriever.retrieve(query_embbeding)
                department_list = [dept["department_name"] for dept in department_list]
                print(f"학과 검색 결과 ::: {department_list} \n\n")
                recommendations = recomm.recommend_classes_dept(query if args.expand == "no" else expanded_query, department_list)
                
            elif args.dept == "yes":
                department_list = [dept.strip() for dept in departments.split(",")]
                recommendations = recomm.recommend_classes_dept(query if args.expand == "no" else expanded_query, department_list)
                
            else:
                recommendations = recomm.recommend_classes(query if args.expand == "no" else expanded_query)


            # 🔹 JSON 저장 형식 구성 (메타정보 포함)
            result_data = {
                "meta_info": {
                    "user_query": query,
                    "expanded_query": expanded_query if args.expand == "yes" else None,
                    "given_departments": department_list if args.dept == "yes" else None,
                    "selected_departments": department_list if args.dept == "auto" else None
                },
                "recommendations": recommendations # (final result)
            }
            
            if args.dept == "auto":
                save_folder_dept = "original"
            elif args.dept == "yes":
                save_folder_dept = "given_dept"
            else:
                save_folder_dept = "no_dept"
            
            output_dir = f'/root/ai_curri/content_based/TFIDF_new/output/new/{save_folder_dept}/query_exp_{args.expand.upper()}'
            os.makedirs(output_dir, exist_ok=True)
            json_file = f'{output_dir}/{major}({idx+1})_recommendations_{args.expand}.json'
                
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