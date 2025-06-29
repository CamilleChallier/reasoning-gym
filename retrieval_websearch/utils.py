import requests
import json
import transformers

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from googlenewsdecoder import gnewsdecoder
from bs4 import BeautifulSoup
from newspaper import Article
from tqdm import tqdm


# load data jsonl file
def load_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line)["question"] + " ?")
    return data


BASE = "http://localhost:8000"
AUTH = ("admin", "password")  # Remove if not needed

def fetch_top_docs(query, providers=None, top_k=5):
    params = {
        "qs": query,
        "format": "json",
        "result_mixer": "RelevancyMixer"
    }
    if providers:
        params["providers"] = ",".join(providers)
    
    resp = requests.get(f"{BASE}/swirl/search", params=params, auth=AUTH)
    resp.raise_for_status()
    data = resp.json()
    
    docs = data.get("results", [])[:top_k]
    
    return [{"title": d["title"], "url": d["url"], "snippet": d.get("body", "")} for d in docs]

def decode_google_news_url(google_url):
    try:
        result = gnewsdecoder(google_url, interval=1)
        if result.get("status"):
            return result["decoded_url"]
        else:
            print("Error:", result["message"])
            return None
    except Exception as e:
        print(f"Error occurred: {e}")
        return None

def scrape_page(doc):
    
    url = doc["url"]
    
    if "news.google.com" in url :
        url = decode_google_news_url(url)
        if not url:
            return None
        else:
            try:
                article = Article(url)
                article.download()
                article.parse()
                text = article.text

                if text.startswith("Error:") or text.startswith(" \n\n\n\n\n\nAccess to this page has been denied") or len(text) < 100:
                    return None
                return text
            
            except Exception as e:
                print(f"Error scraping {url}: {e}")
                return None
            
    elif "europepmc.org" in url or "archive.org" in url:
        return doc["snippet"]
    
    else:
        try:
            response = requests.get(url)

            soup = BeautifulSoup(response.content, 'html.parser')
            text= soup.get_text()
            if text.startswith("Error:") or text.startswith(" \n\n\n\n\n\nAccess to this page has been denied") or len(text) < 100:
                return None
            return text
        except Exception as e:
            print(f"Error scraping {url}: {e}")
            return None
        
def generate_search_query(dataset, model_id, hf=True):
    
    if hf:
    
        pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )

        for i, data in tqdm(enumerate(dataset)):
            
            query = data["query"]
                
            messages = [
                {"role": "system", "content": "You are a helpful assistant that can suggest search prompts."},
                {"role": "user", "content": f"""Given the following question, your task is to suggest a web search query that you would use to find the answer. 
            Do not answer the question itself — only provide the search prompt you would use. 

            Question: {query}"""}
            ]
            outputs = pipeline(
                messages,
                max_new_tokens=256,
            )
            
            # extract text between \"
            dataset[i]["web_search_query"]  = outputs[0]["generated_text"][-1]["content"].strip('"')
            
        return dataset
    else:
        
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, device_map={"": "cpu"}, torch_dtype=torch.float32)
        model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, device_map={"": "cpu"}, torch_dtype=torch.float32)

        # max_mem = {0: "2GiB"}
        pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            # max_memory=max_mem,
            # model_kwargs={"torch_dtype": torch.bfloat16},
            # device_map="auto",
            device_map="cpu",

        )

        for i, data in tqdm(enumerate(dataset)):
            
            query = data["query"]
                
            messages = [
                {"role": "system", "content": "You are a helpful assistant that can suggest search prompts."},
                {"role": "user", "content": f"""Given the following question, your task is to suggest a web search query that you would use to find the answer. 
            Do not answer the question itself — only provide the search prompt you would use. 

            Question: {query}"""}
            ]    
            
            input_ids = tokenizer(format_chat(messages), return_tensors="pt").input_ids #.cuda()
            outputs = model.generate(input_ids, max_new_tokens=256)#.cuda()
            decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)[len(format_chat(messages)):]
            print(decoded)
            dataset[i]["web_search_query"]  = decoded.strip('"') if type(decoded) is str else decoded

        return dataset

def format_chat(messages):
    prompt = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            prompt += f"<|system|>\n{content}\n"
        elif role == "user":
            prompt += f"<|user|>\n{content}\n"
        elif role == "assistant":
            prompt += f"<|assistant|>\n{content}\n"
    prompt += "<|assistant|>\n"  # For the model to complete
    return prompt

def evaluate_retrieval(dataset):
    """
    Evaluate the retrieval by checking if the titles of the original documents
    and the predicted documents match.
    """
    score = 0
    for i in range(len(dataset)):
        titles_original = {item["title"] for item in dataset[i]["documents"]}
        titles_predicted = {item["title"] for item in dataset[i]["documents_answer"]}
        
        # Find common titles
        common_titles = titles_original & titles_predicted
        dataset[i]["retrieval_score"] = 1 if len(common_titles) > 0 else 0
        score += dataset[i]["retrieval_score"]
        
    return score / len(dataset)