import transformers
import torch
from utils import load_data, fetch_top_docs, scrape_page, generate_search_query
from tqdm import tqdm
import json

#load dataset
with open("qa_dataset_scrape.json", "r") as f:
    dataset = json.load(f)

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

for i, data in enumerate(dataset):
    
    j=0
    text = None
    while text is None and j < len(data["documents"]):
        text = data["documents"][j]["content"]
        j+= 1
    
    messages = [
    {"role": "system", "content": "You are a helpful assistant that generates questions based on provided text. Output a single question, without any additional text and answer. Do not ask questions about cookies, privacy, or terms of service, only about the content of the text."},
    {"role": "user", "content": "Generate a simple question based on the following text: " + text + " ? Add a short context to the question if needed, for example, give a context about the film or book the question is about."},
    ]
    outputs = pipeline(
        messages,
        max_new_tokens=256,
    )
    print(outputs[0]["generated_text"][-1]["content"])

    dataset[i]["query"] = outputs[0]["generated_text"][-1]["content"]
    
    if i% 10 == 0:
        print(f"Processed {i} query")

        # Save dataset
        with open("qa_dataset_query.json", "w") as f:
            json.dump(dataset, f, indent=2)

with open("qa_dataset_query.json", "w") as f:
    json.dump(dataset, f, indent=2)