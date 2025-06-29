from utils import  generate_search_query
import json

with open("qa_dataset_query.json", "r") as f:
    dataset = json.load(f)

MODEL_PATH = "/capstor/store/cscs/swissai/infra01/swiss-alignment/checkpoints/Apertus3-8B_iter_1678000-tulu3-sft/checkpoint-13446"

dataset = generate_search_query(dataset, MODEL_PATH, False)
    
with open("qa_dataset_apertus8.json", "w") as f:
    json.dump(dataset, f, indent=2)
