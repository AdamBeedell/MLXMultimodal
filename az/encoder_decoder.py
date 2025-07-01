import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from torchmetrics.classification import MulticlassAccuracy
from PIL import Image 

image = Image.open("data/000000039769.jpg")
text = "Two cats are playing on a sofa"
## image.show()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# set random seed
torch.backends.cudnn.deterministic = True 
torch.manual_seed(1234)
torch.cuda.manual_seed_all(5678) 

### use an existing encoder for the images
from transformers import ViTFeatureExtractor, ViTModel, BertTokenizer, BertModel
vit_model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
vit_prepro = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
vit_model.eval()

nima = vit_prepro(image, return_tensors="pt")
print(nima.data['pixel_values'].shape) # these are the dimensions of the preprocessed batch of size 1

### use BERT for embeddings
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_pretrained = BertModel.from_pretrained('bert-base-uncased')
bert_pretrained.eval()
pad_token = 0
s_token = 101
f_token = 102
unk_token = 100

## tokenize text 
# text1 = ["dog "*10, text]
tkt = tokenizer(text, return_tensors="pt", padding = "max_length",  max_length=10, truncation=True)

## target
## we should not be predicting start tokens and padding
bert_vals = [v for k, v in tokenizer.vocab.items() if v not in [101,0]]
bert_vals_rev = {v: i for i, v in enumerate(bert_vals)}

## use -100 for padding 
target = tkt['input_ids'][:,1:].detach().clone().numpy()
for r, row in enumerate(target):
    for c, el in enumerate(row):            
        target[r,c] = -100 if el==0 else bert_vals_rev.get(el, bert_vals_rev[unk_token]) 
target = torch.tensor(target)    

### decoder

# ## load data and split
# images = np.load("data/comb_images.npy")
# labs = np.load("data/comb_labs.npy")
# vocab = np.load("data/vocab.npy")
# lookup = {w: i for i, w in enumerate(vocab)}

# model
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

# in_dim is the dimension of a patch
# seq_len is the number of patches

class LongDecode(nn.Module):
    def __init__(self,  im_dim, tx_dim, hidden_dim, seq_len, rseq_len, vocab_len, kq_dim): 
       super().__init__()
       self.fc_im_start = nn.Linear(im_dim, hidden_dim)
       pos = torch.arange(0, seq_len).unsqueeze(0)
       self.register_buffer('pos', pos, persistent=False)
       self.pos_embedding = nn.Embedding(seq_len, hidden_dim)
       self.fc_tx_start = nn.Linear(tx_dim, hidden_dim)
       rpos = torch.arange(0, rseq_len).unsqueeze(0)
       self.register_buffer('rpos', rpos, persistent=False)  
       self.rpos_embedding = nn.Embedding(rseq_len, hidden_dim)
       cmask_tl = torch.full((seq_len, seq_len), False)
       cmask_tr = torch.full((seq_len, rseq_len), False)
       cmask_lr = torch.ones(rseq_len, rseq_len).tril()==0
       cmask = torch.cat([torch.cat([cmask_tl, cmask_tr], dim=1), 
                          torch.cat([cmask_tr.t(), cmask_lr], dim=1)], dim=0).unsqueeze(0)
       self.register_buffer('causal_mask', cmask, persistent=False)
       self.decoders = nn.ModuleList([Decodr(hidden_dim, hidden_dim, kq_dim) for i in range(1)])
       self.fc_final = nn.Linear(hidden_dim, vocab_len-2)
       
    def forward(self, image, text, pad_token=0):
        im0 = vit_prepro(image, return_tensors="pt")
        with torch.no_grad():
            image_features = vit_model(**im0)
        im0 = image_features.to_tuple()[0]
        im = self.fc_im_start(im0) + self.pos_embedding(self.pos) 
        with torch.no_grad():
            outputs = bert_pretrained(**text)
        tx0 = outputs.last_hidden_state
        tx = self.fc_tx_start(tx0) + self.rpos_embedding(self.rpos) 
        x = torch.cat([im, tx], dim=1)

# generate         
        key_mask = torch.cat([torch.full((im.shape[0], im.shape[1]), False), 
                              text['attention_mask']==0], 
                             dim=1) 
        for i, de in enumerate(self.decoders):
            x = de(x, causal_mask = self.causal_mask, key_mask = key_mask)
        logits = self.fc_final(x[:,im.shape[1]:-1,:])
        return logits


model = LongDecode(im_dim = 768, tx_dim = 768 , hidden_dim = 1024, 
                   seq_len = 197, rseq_len = 10, vocab_len = 30522, kq_dim = 512)

x = model(image, tkt)
y = target

criterion = nn.CrossEntropyLoss(ignore_index=-100)
criterion(x.transpose(2,1), y)



### basement

## target
## we should not be predicting start tokens and padding
bert_vals = [v for k, v in tokenizer.vocab.items() if v not in [101,0]]
bert_vals_rev = {v: i for i, v in enumerate(bert_vals)}

## use -100 for padding 
target = tkd.detach().clone().numpy()
for r, row in enumerate(target):
    for c, el in enumerate(row):            
        target[r,c] = -100 if el==0 else bert_vals_rev.get(el, bert_vals_rev[unk_token]) 
target = torch.tensor(target)    
torch.save(target, f = 'data/target_flikr30k.pt')
torch.save(bert_vals, f = 'data/BERT_values.pt')
 



def make(config):
    # Make the data
    sel_preq = np.tril(np.ones(labs.shape[1], dtype=np.int64), k=-1)==0 
    counts = np.sum(np.logical_and(labs !=lookup['<p>'], labs !=lookup['<s>']), 1)
    nimages = np.repeat(break_and_flatten_3D(images, config['patch_dim']), counts, axis=0)
    infld = np.stack((labs,) * labs.shape[1], axis=-1) 
    infld[:,sel_preq] = lookup['<p>']
    infld = np.concatenate([np.expand_dims(labs, -1), infld], -1)
    infld = infld.reshape(labs.shape[0]*labs.shape[1], labs.shape[1]+1)
    nlabs = infld[np.logical_and(infld[:,0] !=lookup['<p>'], infld[:,0] !=lookup['<s>']),:]
    
    full_dataset = TensorDataset(torch.tensor(nimages), 
                                 torch.tensor(nlabs[:,1:], dtype = torch.long), 
                                 torch.tensor(nlabs[:,0], dtype = torch.long))
    
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [0.9, 0.1])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'],
                                               shuffle=True,
                                               pin_memory = True )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config['batch_size'],
                                              pin_memory = True)
    
    # Make the model 
    model = Annotator(in_dim = config['patch_dim']**2,
                      seq_len = config['seq_len'],
                      rseq_len = config['rseq_len'],
                      vocab_len = len(config['vocab']),
                      hidden_dim = config['hidden_dim'],
                      v_dim = config['v_dim'],
                      kq_dim = config['kq_dim'])
    
    # Make the loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
    
    return model, train_loader, test_loader, criterion, optimizer

def train(model, train_loader, criterion, optimizer, config):  
    model = model.to(config['device'], non_blocking=True)
    model.train()
    print(f"Model's device: {next(model.parameters()).device}")
    for epoch in range(config['num_epochs']):
        losses = 0
        for im, preq, lab in train_loader:
            im = im.to(config['device'], non_blocking=True)
            preq = preq.to(config['device'], non_blocking=True)
            lab = lab.to(config['device'], non_blocking=True)
            mask = (preq == lookup["<p>"]).to(config['device'], non_blocking=True)
            with torch.autocast(config['device']):
                optimizer.zero_grad()
                p = model(im, preq, mask)
                loss = criterion(p, lab)
                loss.backward()
                optimizer.step()
                losses += loss
        mean_loss = losses/len(train_loader)
        print(f'Epoch {epoch+1}, Loss: {mean_loss.item():.4f}') 
 
def test(model, test_loader, criterion):
    model.eval()
    mca = MulticlassAccuracy(num_classes=11)
    losses = 0
    for im, preq, lab in test_loader:  
        with torch.no_grad():
            im = im.to(config['device'], non_blocking=True)
            preq = preq.to(config['device'], non_blocking=True)
            lab = lab.to(config['device'], non_blocking=True)
            mask = (preq == lookup["<p>"]).to(config['device'], non_blocking=True)
            p = model(im, preq, mask)     
            loss = criterion(p, lab)
            losses += loss
            mca.update(p, lab) 
    mean_loss = losses/len(test_loader)
    print(f'Average test loss, Loss: {mean_loss.item():.4f}') 
    print("Accuracy:", mca.compute().item()) 

# def test(model, test_loader, criterion):
#     model.eval()
#     mca = MulticlassAccuracy(num_classes=11)
#     losses = 0
#     # for im, preq, lab in test_loader: 
#     im, preq, lab = next(iter(test_loader))
#     with torch.no_grad():
#         im = im.to(config['device'], non_blocking=True)
#         print(f"im's shape is {im.shape}")
#         preq = preq.to(config['device'], non_blocking=True)
#         print(f"preq's shape is {preq.shape}")
#         lab = lab.to(config['device'], non_blocking=True)
#         print(f"lab's shape is {lab.shape}")
#         mask = (preq == lookup["<p>"]).to(config['device'], non_blocking=True)
#         p = model(im, preq, mask)
#         print(f"predictions' shape is {p.shape}")            
#         loss = criterion(p, lab)
#         losses += loss
#         mca.update(p, lab) 
#         print(p[0])
#         print(lab[0])
#     mean_loss = losses/len(test_loader)
#     print(f'Average test loss, Loss: {mean_loss.item():.4f}') 
#     print("Accuracy:", mca.compute().item()) 
    
config = dict(
    num_epochs=10,
    batch_size=64,
    learning_rate=0.01,
    patch_dim = 4, 
    rseq_len = labs.shape[1],
    vocab = vocab,
    hidden_dim = 32,
    v_dim = 32,
    kq_dim = 20, 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    )
config['seq_len'] = int(np.ceil(images[0].size/config['patch_dim']**2))

model, _, test_loader, criterion, _ = make(config)

print(model)
# train(model, train_loader, criterion, optimizer, config)
#### load model

state = torch.load("temp/current_model_e.pth", map_location=device, weights_only=True) 
model.load_state_dict(state)

test(model, test_loader, criterion) 

## pick an item for input
def pick(n):
    flabs = labs[:n]
    fimages = images[:n]
    sel_preq = np.tril(np.ones(flabs.shape[1], dtype=np.int64), k=-1)==0 
    counts = np.sum(np.logical_and(flabs!=lookup['<p>'], flabs !=lookup['<s>']), 1)
    nimages = np.repeat(break_and_flatten_3D(fimages, config['patch_dim']), counts, axis=0)
    infld = np.stack((flabs,) * flabs.shape[1], axis=-1) 
    infld[:,sel_preq] = lookup['<p>']
    infld = np.concatenate([np.expand_dims(flabs, -1), infld], -1)
    infld = infld.reshape(flabs.shape[0]*flabs.shape[1], flabs.shape[1]+1)
    nlabs = infld[np.logical_and(infld[:,0] !=lookup['<p>'], infld[:,0] !=lookup['<s>']),:]
    return torch.tensor(nimages), torch.tensor(nlabs[:,1:]), torch.tensor(nlabs[:,0])

test_v, test_s, test_t = pick(1) 
mask = (test_s == lookup["<p>"])

ex_input = (test_v[0].unsqueeze(0), test_s[0].unsqueeze(0), mask[0].unsqueeze(0))
torch.onnx.export(model,
                  ex_input,  
                  "temp/working.onnx",
                  input_names = ['image','text','mask'],
                  export_params=True)

import onnxruntime
ort_session = onnxruntime.InferenceSession("temp/working.onnx")

test_image = test_v[0].unsqueeze(0).detach().numpy()

def read(image):
    result = np.array([lookup['<s>']] + [lookup['<p>']]*5)
    illegal = np.array([lookup['<s>'],lookup['<p>']])
    for i in range(5):
        mask = np.expand_dims(np.equal(result, lookup['<p>']), 0) 
        ort_inputs = {"image": image, 
                      "text": np.expand_dims(result, 0),
                      "mask": mask}
        ort_outs = ort_session.run(None, ort_inputs) 
        prval = np.argmax(ort_outs) 
        result[i+1] = prval 
        if np.isin(prval, illegal):
            return "illegal result: " + ''.join([vocab[j] for j in result])
        elif prval == lookup['<f>']:
            return ''.join([vocab[j] for j in result[1:i+1]]) 
    return "illegal result: " + ''.join([vocab[j] for j in result])            
            
print(read(test_image))

## ground truth
print(''.join([vocab[k] for k in labs[0]]))

