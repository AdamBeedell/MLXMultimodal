from datasets import load_dataset
import pandas as pd
import pathlib
from transformers import ViTImageProcessor, BertTokenizer
import torch

pathlib.Path('data').mkdir(parents=True, exist_ok=True)

ds0 = load_dataset("nlphuji/flickr30k") 
ds = pd.DataFrame(ds0['test'][:2000])
ds.reset_index(drop=True, inplace=True)
ds.reset_index(inplace=True)
ds.rename(columns = {"index":"image_id"}, inplace=True)

### make a torch tensor with a grid
dse = ds.loc[:, ['image_id','caption','split']].explode('caption')
dse.reset_index(drop=True, inplace=True)
dse.reset_index(inplace=True)
dse.rename(columns = {"index":"text_id"}, inplace=True)
grid = torch.tensor(dse.loc[:,['image_id','text_id']].to_numpy())
torch.save(grid, f = 'data/grid_flikr30k.pt')

### make a torch tensor with text
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') 
pad_token = 0
s_token = 101
f_token = 102
unk_token = 100
 
tkd = tokenizer(dse['caption'].to_list(), return_tensors="pt", 
                padding = "max_length", max_length=88, truncation=True)
tkd = tkd['input_ids']
torch.save(tkd, f = 'data/text_flikr30k.pt')

### make a torch tensor with images 
vit_prepro = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
images = vit_prepro(ds['image'].to_list(), return_tensors="pt")['pixel_values']
torch.save(images, f = 'data/images_flikr30k.pt')

