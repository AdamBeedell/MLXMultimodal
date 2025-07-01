import transformers

# load in pre-trained models
clip_model = transformers.CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
clip_large = transformers.CLIPModel.from_pretrained('openai/clip-vit-large-patch14')
vit_model = transformers.ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
params = lambda m: sum(p.numel() for p in m.parameters())








