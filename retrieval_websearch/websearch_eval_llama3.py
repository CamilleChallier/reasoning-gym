from utils import  generate_search_query
import json


with open("qa_dataset_query.json", "r") as f:
    dataset = json.load(f)

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

dataset = generate_search_query(dataset, model_id, True)
        
        
with open("qa_dataset_llama3.json", "w") as f:
    json.dump(dataset, f, indent=2)
