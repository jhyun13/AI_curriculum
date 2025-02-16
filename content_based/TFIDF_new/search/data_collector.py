def collect_goal(batch):
    batch_dict ={
        'text' : [item['text'] for item in batch],
        'department_id' : [item['department_id'] for item in batch],
        'department_name' : [item['department_name'] for item in batch],  
    }
    return batch_dict