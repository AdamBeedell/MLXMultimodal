import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from torchmetrics.classification import MulticlassAccuracy
from transformers import ViTModel, BertModel, BertTokenizer

import pathlib 
pathlib.Path('temp').mkdir(parents=True, exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# set random seed
torch.backends.cudnn.deterministic = True 
torch.manual_seed(1234)
torch.cuda.manual_seed_all(5678) 

### use an existing encoder for the images
vit_model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
vit_model.eval()
bert_model = BertModel.from_pretrained('bert-base-uncased')
bert_model.eval()
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_vals = [v for k, v in tokenizer.vocab.items() if v not in [101,0]]
bert_vals_rev = {v: i for i, v in enumerate(bert_vals)}

## load data and split
grid = torch.load("data/grid_flikr30k.pt", weights_only=True)[:500]
image_data = torch.load("data/images_flikr30k.pt", weights_only=True)
text_data = torch.load("data/text_flikr30k.pt", weights_only=True)  

## use -100 for padding 
unk_token = 100
target = text_data[:,1:].detach().clone().numpy()
for r, row in enumerate(target):
    for c, el in enumerate(row):            
        target[r,c] = -100 if el==0 else bert_vals_rev.get(el, bert_vals_rev[unk_token]) 
target = torch.tensor(target)    

# model
# decoder from Qwen




class Attention(nn.Module):
    def __init__(self,  vk_i_dim, q_i_dim, kq_dim):
        super().__init__()
        self.linear_v = nn.Linear(vk_i_dim, q_i_dim, bias=False)
        self.linear_k = nn.Sequential(
            nn.Linear(vk_i_dim, kq_dim, bias=False),
            nn.LayerNorm(kq_dim)
            )
        self.linear_q = nn.Sequential(
            nn.Linear(q_i_dim, kq_dim, bias=False),
            nn.LayerNorm(kq_dim)
            )
        self.register_buffer("k_scaler", torch.tensor(kq_dim**0.5))
        self.softmax_a = nn.Softmax(dim = -1)
        self.ln = nn.LayerNorm(q_i_dim)
        self.dropout = nn.Dropout(0.1)               
        
    def forward(self, vals, keys, ques, causal_mask = None, key_mask = None):
       v = self.linear_v(vals)
       k = self.linear_k(keys)
       q = self.linear_q(ques) 
       a = q @ k.transpose(-1,-2) / self.k_scaler
       if causal_mask is not None:
           a = a.masked_fill(causal_mask, float('-inf')) 
       if key_mask is not None:
           a = a.masked_fill(key_mask.unsqueeze(1), float('-inf')) 
       return self.ln(self.dropout(self.softmax_a(a) @ v) + ques)

class Decodr(nn.Module):
    def __init__(self,  ext_dim, hidden_dim, kq_dim): 
       super().__init__()
       self.selfatt = Attention(ext_dim, ext_dim, kq_dim)
       self.ff = nn.Sequential(
           nn.Linear(ext_dim, hidden_dim),
           nn.ReLU(),
           nn.Linear(hidden_dim, ext_dim)
           ) 
       self.ln = nn.LayerNorm(ext_dim)
       
    def forward(self, emb, causal_mask, key_mask): 
        d = self.selfatt(emb, emb, emb, causal_mask = causal_mask, key_mask = key_mask)
        d = self.ln(self.ff(d) + d)
        return d

class AddViTEncoding(nn.Module):
    def __init__(self): 
       super().__init__()
       
    def forward(self, images): 
        with torch.no_grad():
            im = vit_model(images).last_hidden_state
        return im

class LongDecode(nn.Module):
    def __init__(self,  im_dim, tx_dim, hidden_dim, seq_len, rseq_len, vocab_len, kq_dim): 
       super().__init__()
       self.im_start = nn.Sequential(
           AddViTEncoding(),
           nn.Linear(im_dim, hidden_dim)
           )
       pos = torch.arange(0, seq_len).unsqueeze(0)
       self.register_buffer('pos', pos, persistent=False)
       self.pos_embedding = nn.Embedding(seq_len, hidden_dim)
       self.tx_start = nn.Sequential(
           bert_model.get_input_embeddings(),
           nn.Linear(tx_dim, hidden_dim)
           )
       rpos = torch.arange(0, rseq_len).unsqueeze(0)
       self.register_buffer('rpos', rpos, persistent=False)  
       self.rpos_embedding = nn.Embedding(rseq_len, hidden_dim)
       cmask_tl = torch.full((seq_len, seq_len), False)
       cmask_tr = torch.full((seq_len, rseq_len), False)
       cmask_lr = torch.ones(rseq_len, rseq_len).tril()==0
       cmask = torch.cat([torch.cat([cmask_tl, cmask_tr], dim=1), 
                          torch.cat([cmask_tr.t(), cmask_lr], dim=1)], dim=0).unsqueeze(0)
       self.register_buffer('causal_mask', cmask, persistent=False)
       self.decoders = nn.ModuleList([Decodr(hidden_dim, hidden_dim, kq_dim) for i in range(2)])
       self.fc_final = nn.Linear(hidden_dim, vocab_len-2)
       
    def forward(self, image, text, pad_token=0):
        im = self.im_start(image) + self.pos_embedding(self.pos)
        tx = self.tx_start(text) + self.rpos_embedding(self.rpos)
        x = torch.cat([im, tx], dim=1)
        key_mask = torch.cat([torch.full((im.shape[0], im.shape[1]), False), 
                              text==pad_token], dim=1) 
        for i, de in enumerate(self.decoders):
            x = de(x, causal_mask = self.causal_mask, key_mask = key_mask)
        return self.fc_final(x[:,im.shape[1]:-1,:])


# ### start test
# model = LongDecode(im_dim = 768, tx_dim = 768 , hidden_dim = 1024, 
#                    seq_len = 197, rseq_len = 88, vocab_len = 30522, kq_dim = 512)
# x = model(images[:3], text[:3])
# y = target[:3]
# criterion = nn.CrossEntropyLoss(ignore_index=-100)
# criterion(x.transpose(2,1), y)
# ### end test

def make(config):
    # Make the data
    sel  = torch.rand(len(grid))
    train_dataset = grid[sel<=0.95]
    test_dataset = grid[sel>0.95]
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'],
                                               shuffle=True,
                                               pin_memory = True )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config['batch_size'],
                                              pin_memory = True)
    
    # Make the model 
    model = LongDecode(im_dim = 768,
                       tx_dim = 768,
                       hidden_dim = config['hidden_dim'], # 1024
                       seq_len = 197,
                       rseq_len = 60,
                       vocab_len = 30522, 
                       kq_dim = config['kq_dim']) #512
    
    # Make the loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
    
    return model, train_loader, test_loader, criterion, optimizer

def train(model, train_loader, criterion, optimizer, config):  
    model = model.to(config['device'], non_blocking=True)
    model.train()
    print(f"Model's device: {next(model.parameters()).device}")
    for epoch in range(config['num_epochs']):
        losses = 0
        for gr in train_loader:          
            im = image_data[gr[:,0]].to(config['device'], non_blocking=True)
            tx = text_data[gr[:,1]].to(config['device'], non_blocking=True)
            lab = target[gr[:,1]].to(config['device'], non_blocking=True) 
            with torch.autocast(config['device']):
                optimizer.zero_grad()
                p = model(im, tx)
                loss = criterion(p.transpose(2,1), lab)
                loss.backward()
                optimizer.step()
                losses += loss
        mean_loss = losses/len(train_loader)
        print(f'Epoch {epoch+1}, Loss: {mean_loss.item():.4f}') 
 
def test(model, test_loader, criterion):
    model.eval()
    mca = MulticlassAccuracy(num_classes=30522)
    losses = 0
    for gr in test_loader:  
        with torch.no_grad():
            im = image_data[gr[:,0]].to(config['device'], non_blocking=True)
            tx = text_data[gr[:,1]].to(config['device'], non_blocking=True)
            lab = target[gr[:,1]].to(config['device'], non_blocking=True)
            p = model(im, tx)
            loss = criterion(p.transpose(2,1), lab)
            losses += loss
            mca.update(p, lab) 
    mean_loss = losses/len(test_loader)
    print(f'Average test loss, Loss: {mean_loss.item():.4f}') 
    print("Accuracy:", mca.compute().item()) 

config = dict(
    num_epochs=10,
    batch_size=8,
    learning_rate=0.01,
    hidden_dim = 256, 
    kq_dim = 20, 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    )

model, train_loader, test_loader, criterion, optimizer = make(config)

print(model)
train(model, train_loader, criterion, optimizer, config)

torch.save(model.state_dict(), "temp/model_state.pt")
#### load model
test(model, test_loader, criterion) 
