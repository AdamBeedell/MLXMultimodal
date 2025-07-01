from transformers import CLIPModel, AutoModelForCauselLM, AutoTokenizer
import torch.nn as nn

# load in pre-trained models
clip_model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')

# ### know your models
# params = lambda m: sum(p.numel() for p in m.parameters())
# # print the number of parameters
# print(f'clip_base:',{params(clip_model)})
# # print the architecture of the model
# print('clip_model:',clip_model)
# # what are the methods in the model
# methods = [func for func in dir(clip_model) if callable(getattr(clip_model, func))]
# print(methods)
# # lsit all parameter names and shapes
# for name, param in clip_model.named_parameters():
#     print('vision_model.' + name, param.shape)

### extract the vision and text encoders from CLIP
vision_encoder = clip_model.vision_model
text_encoder = clip_model.text_model
# freeze the models if you don't want to train it
for param in vision_encoder.parameters():
    param.requires_grad = False
for param in text_encoder.parameters():
    param.requires_grad = False



### extract the decoder from QWen
qwen_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B-Base")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")
decoder = qwen_model.model
print(qwen_model)
# only train the decoder and any new connecting layers that used to connect the encoder and decoder
for param in decoder.parameters():
    param.requires_grad = True 




