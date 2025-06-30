import torch
import datasets
from datasets import load_dataset, Dataset, DatasetDict, Features, Value, Image
import transformers as t
import kagglehub
import pandas as pd
import os
import requests
import base64
from PIL import Image
from io import BytesIO
print(os.getcwd())


c = t.CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
v = t.ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")


from transformers import AutoModelForCausalLM, AutoTokenizer

q3 = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B-Base")
tokq3 = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")


def params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Qwen3:", params(q3))

qi = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
tokqi = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
proqi = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")



print("CLIP:", params(c))
print("ViT:", params(v))



ds = load_dataset("flickr30k")
print(ds.info)  # Sometimes gives size in bytes

# Download latest version
#path = kagglehub.dataset_download("hsankesara/flickr-image-dataset")



# opendata from local
df = pd.read_csv("AB/data/datasets/hsankesara/flickr-image-dataset/versions/1/flickr30k_images/results.csv", sep='|', names=["image_name", "comment_number", "comment"], engine="python")
# clean data
df.columns = [c.strip() for c in df.columns]  # Fix spacing in headers
df["comment_number"] = df["comment_number"].astype(str).str.strip().astype(int) # fix spaces in comment_number
df = df[df["comment_number"].astype(str).str.strip().str.isdigit()] # Keep only rows where comment_number is a digit
df["image"] = "AB/data/datasets/hsankesara/flickr-image-dataset/versions/1/flickr30k_images/flickr30k_images/" + df["image_name"] # add images
df = df.drop(columns=["comment_number"]) ## there's multiple comments per image, for the moment I dont really care
## convert to HF dataset type

df = df.reset_index(drop=True)  # kills an index
dataset = Dataset.from_pandas(df, features=features, preserve_index=False) ## I dont get how doing this first works but it seems to?


dataset = Dataset.from_pandas(df, features=Features({  ##bring the dataframe to a dataset
    "image_name": Value("string"),
    "comment": Value("string"),
    "image": Image()
}))



#df = pd.read_csv("results.csv", sep='|')


# Keep only one caption per image (e.g., the first)
df = df[df["comment_number"] == 0]
df.columns = [c.strip() for c in df.columns]






def image_to_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def query_lm_studio(image, prompt):
    image_b64 = image_to_base64(image)
    payload = {
        "model": "Qwen2.5-VL-3B-Instruct",
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
            ]}
        ]
    }

    response = requests.post("http://localhost:1234/v1/chat/completions", json=payload)
    return response.json()["choices"][0]["message"]["content"]


for row in dataset