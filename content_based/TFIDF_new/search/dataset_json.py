from torch.utils.data import Dataset

class goalDatasetjson(Dataset):
    def __init__(self, data, group_by="id"):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        doc = self.data[index]
        
        department_id = doc["department_id"]
        department_name = doc.get("학과", "Unknown")
        description = doc.get("학과설명", "")

        return {
            "department_id": department_id,
            "department_name": department_name,
            "text": description,
        }    