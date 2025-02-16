import os 
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pickle
import faiss
import tiktoken
from .data_collector import collect_goal


class DenseRetriever:
    def __init__(self, client, args, dept_info=None):
        self.client = client
        self.dept_info = dept_info
        self.args = args
        self.index = None
        self.lookup_index = None
        self.save_path = "data"
        self.DEPT_VECTOR_FILE = "data/dept_vectorizer.pkl"


    def split_text_to_chunks(self, text, max_tokens, model="text-embedding-3-large"):
    
        encoder = tiktoken.encoding_for_model(model)
        tokens = encoder.encode(text)  # 텍스트를 토큰으로 변환

        chunks = []
        current_chunk = []
        current_length = 0

        for token in tokens:
            if current_length + 1 > max_tokens:
                # 현재 청크를 문자열로 변환 후 저장
                chunks.append(encoder.decode(current_chunk))
                current_chunk = [token]  # 새로운 청크 시작
                current_length = 1
            else:
                current_chunk.append(token)
                current_length += 1

        # 마지막 청크 추가
        if current_chunk:
            chunks.append(encoder.decode(current_chunk))

        return chunks


    def get_gpt_embedding(self, text):

        max_tokens = 8000  # 모델의 최대 허용 토큰
        chunks = self.split_text_to_chunks(text, max_tokens)  # 토큰 기준으로 텍스트 분할
        chunk_embeddings = []


        for chunk in chunks:
            response = self.client.embeddings.create(input=chunk, model='text-embedding-3-large')
            chunk_embeddings.append(response.data[0].embedding)

        # Combine embeddings of chunks by averaging
        return np.mean(chunk_embeddings, axis=0)

    
    def doc_embedding(self):
        """ 전체 embedding 처리 및 FAISS index 생성 """
        # 1️⃣ 저장된 임베딩 로드 (이미 존재하면 로드 후 종료)
        if self.load_existing_embeddings():
            return self.index, self.lookup_index
        
        # 2️⃣ 학과 데이터 임베딩 생성
        if self.dept_info is not None:
            embeddings_array, lookup_index = self.generate_department_embeddings()
            
            # 저장
            self.save_embeddings(embeddings_array, lookup_index)
        
        # 3️⃣ FAISS 인덱스 생성
        self.create_faiss_index(embeddings_array)

        return self.index, self.lookup_index
    
    
    def load_existing_embeddings(self):
        """ 기존에 저장된 임베딩 데이터를 로드 """
        save_file = self.DEPT_VECTOR_FILE
        if os.path.exists(save_file):
            with open(save_file, "rb") as f:
                data = pickle.load(f)
                embeddings = np.array(data["embeddings"], dtype=np.float32)
                self.lookup_index = data["lookup_index"]
                self.index = faiss.IndexFlatIP(embeddings.shape[1])
                self.index.add(embeddings)
            return True  # 로드 성공 시 True 반환
        return False  # 로드 실패 시 False 반환

    def generate_department_embeddings(self):
        """ 학과 데이터셋을 기반으로 임베딩 생성 """
        embeddings = []
        lookup_index = []

        dataloader = DataLoader(
            self.dept_info,
            batch_size=self.args.batch_size,
            shuffle=False,
            collate_fn=collect_goal
        )

        for batch in tqdm(dataloader, desc="Generating embeddings"):
            batch_embeddings = [self.get_gpt_embedding(text) for text in batch["text"]]
            
            embeddings.extend(batch_embeddings)
            lookup_index.extend([
                {'department_id': dept_id, 'department_name': dept_name, 'text': text}
                for dept_id, dept_name, text in zip(batch['department_id'], batch['department_name'], batch["text"])
            ])

        embeddings_array = np.array(embeddings, dtype=np.float32)
        self.lookup_index = lookup_index

        return embeddings_array, lookup_index


    def save_embeddings(self, embeddings_array, lookup_index):
        """ 생성된 임베딩을 TSV 및 Pickle 파일로 저장 """
        os.makedirs(self.save_path, exist_ok=True)

        embeddings_files = os.path.join(self.save_path, 'dept_embeddings.tsv')
        metadata_files = os.path.join(self.save_path, 'dept_metadata.tsv')

        np.savetxt(embeddings_files, embeddings_array, delimiter='\t')

        with open(metadata_files, 'w') as f:
            for item in lookup_index:
                f.write(f'{item["department_name"]}\n')

        save_file = self.DEPT_VECTOR_FILE
        with open(save_file, "wb") as f:
            pickle.dump({"embeddings": embeddings_array, "lookup_index": lookup_index}, f)


    def create_faiss_index(self, embeddings_array):
        """ FAISS 인덱스를 생성 및 정규화 """
        embeddings_array /= np.linalg.norm(embeddings_array, axis=1, keepdims=True)  # Normalize
        dim = embeddings_array.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings_array)


    def query_embedding(self, query):
        return self.get_gpt_embedding(query)
    
    
    def retrieve(self, query, top_k=5, threshold_diff=0.015):
        # Generate query embedding
        query_emb = np.array(query, dtype=np.float32).reshape(1, -1)
        query_emb /= np.linalg.norm(query_emb, axis=1, keepdims=True)  # cosine similarity normalization

        # Perform FAISS search
        similarities, indices = self.index.search(query_emb, k=top_k)

        # 첫 번째 결과는 무조건 포함
        selected_results = []
        prev_score = None

        for i, (idx, score) in enumerate(zip(indices[0], similarities[0])):
            if i >= 2 and prev_score is not None and abs(prev_score - score) >= threshold_diff:
                break  # top 2까지는 선택하고 이후에는 threshold_diff 기준으로 중단

            doc_info = self.lookup_index[idx]
            selected_results.append({
                "department_id": doc_info["department_id"],
                "department_name": doc_info.get("department_name", "Unknown"),
                "score": float(score)
            })
                
            prev_score = score       
        return selected_results