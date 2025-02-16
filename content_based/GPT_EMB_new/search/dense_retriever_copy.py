import os 
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pickle
import faiss
import tiktoken
from data_collector import collect_goal


class DenseRetriever:
    def __init__(self, client, args, dept_dataset=None):
        self.client = client
        self.dept_dataset = dept_dataset
        self.args = args
        self.index = None
        self.lookup_index = None
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
        
        if self.args.goal_index_path is not None:
            # Load from saved file
            save_file = os.path.join(self.args.save_path, "goal_Dataset.pkl")
            with open(save_file, "rb") as f:
                data = pickle.load(f)
                embeddings = data["embeddings"]
                embeddings_array = np.array(embeddings, dtype=np.float32)
                self.lookup_index = data["lookup_index"]
    

        os.makedirs(self.args.save_path, exist_ok=True)

        if self.dept_dataset is not None: 
            embeddings = []
            lookup_index = []

            # Create DataLoader
            dataloader = DataLoader(
                self.dept_dataset,
                batch_size=self.args.batch_size,
                shuffle=False,
                collate_fn=collect_goal
            )

            # Generate embeddings
            for batch in tqdm(dataloader, desc="Generating embeddings"):            
                batch_embeddings = []
                for text in batch["text"]:
                    # print(text)
                    embedding = self.get_gpt_embedding(text)
                    batch_embeddings.append(embedding)
                    
                embeddings.extend(batch_embeddings)
                embeddings_array = np.array(embeddings, dtype=np.float32)
                lookup_index.extend(
                    [{'department_id':dept_id, 'department_name':dept_name, 'text':text}
                    for dept_id, dept_name in zip(batch['department_id'], batch['department_name'])]
                )
                self.lookup_index = lookup_index
                
            
            #save embeddings as TSV
            embeddings_files = os.path.join(self.args.save_path, 'goal_embeddings.tsv')
            metadata_files = os.path.join(self.args.save_path, 'goal_metadata.tsv')
            
            np.savetxt(embeddings_files, embeddings_array, delimiter='\t')
            with open(metadata_files, 'w') as f:
                for item in lookup_index:
                    f.write(f'{item["department_name"]}\n')
            
            # Save embeddings
            save_file = os.path.join(self.args.save_path, "goal_Dataset.pkl")
            with open(save_file, "wb") as f:
                pickle.dump({"embeddings": embeddings_array, "lookup_index": lookup_index}, f)
                    

        # # Create FAISS index
        embeddings_array /= np.linalg.norm(embeddings_array, axis=1, keepdims=True)  # Normalize
        
        # IVF 인덱스 생성 (nlist: 클러스터 수, 기본값 100)
        # dim = embeddings_array.shape[1]
        # nlist = getattr(self.args, "nlist", 5)
        # quantizer = faiss.IndexFlatIP(dim)
        # self.index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        # if not self.index.is_trained:
        #     self.index.train(embeddings_array)
        # self.index.add(embeddings_array)
            
        dim = embeddings_array.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings_array)
        
        # # dim = embeddings_array.shape[1]
        # # self.index = faiss.IndexFlatL2(dim)
        # # self.index.add(embeddings_array)

        return self.index, self.lookup_index 


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