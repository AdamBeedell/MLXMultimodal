from transformers import pipeline, AutoProcessor 
import os 
import torch 
from PIL import Image
import datasets as ds
import dotenv

load_dotenv()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_name = "Qwen/Qwen2.5-VL-3B-Instruct"

# set up the path
image_dir = 'images'
output_json = 'Image_Caption_dataset'

image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg','png','jpeg'))] # note: the method .endwith take a single string or a tuple of strings as input 

def message_describe(image):
    return {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": image,
            },
            {"type": "text", "text": "What is on this picture?"},
        ],
    } 
processor = AutoProcessor.from_pretrained(model_name)
pipe = pipeline(
    task="image-text-to-text", model = model_name, processor = processor, device=0, torch_dtype=torch.bfloat16
)

images = []
filenames = []
captions = []

for f in image_files:
    fp = os.path.join(image_dir, f)
    im = Image.open(fp)
    m = message_describe(im)
    with torch.no_grad():
        out = pipe(text=[m], max_new_tokens=100, return_full_text=False)
    images.append(im)
    filenames.append(f)
    captions.append(out[0]['generated_text'])

data = {
    "image": images,
    "caption": captions,
    "filename": filenames
}

features = ds.Features({
    "image": ds.Image(),
    "caption": ds.Value("string"),
    "filename": ds.Value("string"),
})

dataset = ds.Dataset.from_dict(data, features=features)

hf_key = os.getenv("HF_KEY")

dataset.push_to_hub(
    repo_id="ayzor/history-images",
    private=True,                               # Whether the dataset repo should be private
    token=hf_key,                      # Optional if you're already logged in
    split="train",                              # Optional, only needed if you're pushing a single split
    commit_message="New dataset"             # Optional commit message
)
 