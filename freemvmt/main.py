from transformers import CLIPModel, ViTModel

c = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
v = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

print("CLIP:", c)
print("ViTL", v)
