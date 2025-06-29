from utils import  generate_search_query
import json

with open("qa_dataset_query.json", "r") as f:
    dataset = json.load(f)

MODEL_PATH = "/capstor/store/cscs/swissai/infra01/swiss-alignment/checkpoints/Apertus3-70B_iter_798250-tulu3-sft/checkpoint-13446"

dataset = generate_search_query(dataset[0:5], MODEL_PATH, False)
    
with open("qa_dataset_apertus702.json", "w") as f:
    json.dump(dataset, f, indent=2)
