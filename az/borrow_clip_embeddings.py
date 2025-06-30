from PIL import Image
import requests
import torch.nn as nn
import torch

from transformers import CLIPProcessor, CLIPModel

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", output_hidden_states=True)
clip_prepro = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_vi = clip_model.vision_model

 
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
# image.show()

# from transformers import CLIPImageProcessor
# processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
# inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)
# outputs = model(**inputs)
# logits_per_image = outputs.logits_per_image # this is the image-text similarity score
# probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities
 
nima = clip_prepro(image, return_tensors="pt") 
image_features = clip_model.get_image_features(**nima)


### to capture intermediate outputs: use handles (as per chatGPT)
### you would need to use the register_forward_hook() method to attach a hook function 
### to the submodule you want to explore
activations = {}

# Hook function
def hook_fn(module, input, output):
    activations["test_output"] = output

# Register the hook on encoder layer 5
handle = clip_vi.pre_layrnorm.register_forward_hook(hook_fn)
with torch.no_grad():
    res = clip_vi(**nima)

resp = res.to_tuple()