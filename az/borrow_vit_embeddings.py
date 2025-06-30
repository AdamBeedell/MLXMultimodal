from PIL import Image
import requests
import torch.nn as nn
import torch

from transformers import ViTFeatureExtractor, ViTModel

vit_model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
vit_prepro = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")

 
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
# image.show()
 
nima = vit_prepro(image, return_tensors="pt")
print(nima.data['pixel_values'].shape) # these are the dimensions of the preprocessed batch of size 1

with torch.no_grad():
    image_features = vit_model(**nima)
image_features = image_features.to_tuple()

# the output has a tensor of size 1 by 197 (number of patches) by 768 (dimensionality of the output)


