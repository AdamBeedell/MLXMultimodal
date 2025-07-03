
### Imports
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn as NN
import torch.nn.functional as F
import torch.optim as optim
import wandb
import transformers as t
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import pandas as pd
import os
from torch.utils.data import random_split
from PIL import UnidentifiedImageError




### choosing CLIP as dealing with a single image vector seems simplier and also closer to the target diagram

c = t.CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
### v = t.ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

### print(c) ## shows the structure of the model, 

processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")



class HotdogDataset(Dataset):
    def __init__(self, csv_path, image_folder, transform=None):
        self.df = pd.read_csv(csv_path, encoding="latin1")
        self.image_folder = image_folder
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image = None
        if pd.notna(row['imagename']):
            image_path = os.path.join(self.image_folder, f"{int(row['imagename'])}.jpg")
            try:
                image = Image.open(image_path).convert("RGB")
                if self.transform:
                    image = self.transform(image)
            except (FileNotFoundError, UnidentifiedImageError):
                image = None

        text = row['text'] if pd.notna(row['text']) else None
        label = int(row['label'])

        if image is None and text is None:
            return None

        return image, text, label

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

dataset = HotdogDataset(csv_path="data/HotdogNotdog/dataset.csv", image_folder="data/HotdogNotdog/images", transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Custom collate function to skip None entries
def collate_skip_none(batch):
    batch = [x for x in batch if x is not None]
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)




### Config (sweep-friendly, but not sweeping yet
config = {
    "embed_dim": 512, ### Matching CLIP's embedding dim - Note this is not doubled for text+image, as we're feeding in just one or an avg
    "epochs": 50,
    "learning_rate": 0.001,
    "batch_size": 128,
}

## vars but not hyperparams
VOCAB_SIZE = 2
PAD_TOKEN_ID = -100

## Import dataset

train_size = int(0.99 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=collate_skip_none)
val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], collate_fn=collate_skip_none)




### Attention Head
### Boy lots to document here, see obsidian notes file for MLX day 11,12, and Attention is all you need.
### Current mental models:
### 1. What if we had a vector DB of meaningful updates we could apply, what if we arranged a NN to be forced into being that
### 2. Take a batch of patches (best thought about as N patches where N recreates original image).
###     - Make 3 copies
###     - Copy 1 is trained to say How much do i want to be changed WHEN IN POSITION X per other stuff
###     - Copy 2 is trained to say How much do i want to change THING IN POSITION X per other stuff
###     - Copy 3 is trained to say How much change do I want to impart
###     - Arrange copy 1 and 2 into a table, where each cell does dotproduct (matmul function here) to see if they agree
###     - Ergo, if X and Y want to mingle, mingle them Z much.

class AttentionHead(NN.Module):  ## We define this as another neural net basically, kinda feeds into mental model 1. above
    def __init__(self, dim, embed_dim):
        super().__init__() ### Go up a class to NN.Module, do it's initialization function
        self.q_proj = NN.Linear(dim, embed_dim) ### we make 1 layer, and dont connect it. # Query
        self.k_proj = NN.Linear(dim, embed_dim)  # Key
        self.v_proj = NN.Linear(dim, embed_dim)  # Value

    def forward(self, x_q, x_kv=None, mask=None):  ## Forward pass - Note backward pass done for all connected objects in the training loop - Murky on how this works at best
        if x_kv is None:  ### does attention or self attention as appropriate
            x_kv = x_q 
        Q = self.q_proj(x_q) # Query
        K = self.k_proj(x_kv) # Key
        V = self.v_proj(x_kv) # Value

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (Q.shape[-1] ** 0.5)  ## do dot product. Do you want to mingle, do they want to be mingled.
        attn_weights = F.softmax(attn_scores, dim=-1) # normalize
        return torch.matmul(attn_weights, V) # Apply values given QV agreement

### Encoder was here, sub in CLIP model



### If we use images, send image embeddings, if we use text, send text embeddings, if both (unintended) then average them

#image = Image.open("cat.jpg").convert("RGB")
#text = ["a photo of a cat", "a photo of a dog"]

processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")



class Decoder(NN.Module):
    def __init__(self, cfg):  ### cfg hopefully to pull from wandb sweep later
        super().__init__()
        self.token_embed = NN.Embedding(VOCAB_SIZE, cfg["embed_dim"])  ######################################################################## FIX ME
        self.attn = AttentionHead(cfg["embed_dim"], cfg["embed_dim"]) ### make an instance of the attentionhead class in the MNISTModel class
        self.mlp = NN.Sequential(   ### I didnt want the model to be the least deep possible, because that sounds bad, and adding a few layers should just be necessary for any sort of complex classification, which we are doing! Consider this block a hidden layer block
            NN.Linear(cfg["embed_dim"], cfg["embed_dim"] * 2),
            NN.ReLU(),
            NN.Linear(cfg["embed_dim"] * 2, cfg["embed_dim"])
        )
        self.norm1 = NN.LayerNorm(cfg["embed_dim"]) ### unspike any weird spikes we might have
        self.norm2 = NN.LayerNorm(cfg["embed_dim"]) ### and again for query
        self.classifier = NN.Linear(cfg["embed_dim"], 2) ### output layer needs nodes equal to each logit to be predicted


    def forward(self, encoder_output):  # [B, 512]
        x = self.attn(self.norm1(encoder_output))          # [B, 512]
        x = self.mlp(self.norm2(x))                        # [B, 512]
        return self.classifier(x)                          # [B, 2]
    


class EncoderDecoderModel(NN.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.c = encoder
        self.decoder = decoder

    def forward(self, image=None, text=None):
        #inputs = processor(text=text, images=image, return_tensors="pt", padding=True).to(next(self.c.parameters()).device)
        device = next(self.c.parameters()).device

        if image is not None and text is not None:
            inputs = processor(text=text, images=image, return_tensors="pt", padding=True).to(device)
            image_emb = self.c.get_image_features(**inputs)
            text_emb = self.c.get_text_features(**inputs)
            encoder_output = (image_emb + text_emb) / 2


        elif image is not None:
            inputs = processor(images=image, return_tensors="pt").to(device)
            encoder_output = self.c.get_image_features(**inputs)

        elif text is not None:
            inputs = processor(text=text, return_tensors="pt").to(device)
            encoder_output = self.c.get_text_features(**inputs)


        else:
            raise ValueError("No input provided to encoder.")

        return self.decoder(encoder_output)
    


### Save + W&B upload
def save_model(model, path): #Locally
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def upload_model_to_wandb(model, run, path): # Remote
    artifact = wandb.Artifact("ABSeeFood", type="model")  
    artifact.add_file(path)
    run.log_artifact(artifact)
    print("Model uploaded to W&B")

### Training loop
def training():
    run = wandb.init(project="MLXMultimodal", config=config)
    cfg = wandb.config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = c.to(device)  # already loaded CLIPModel
    decoder = Decoder(cfg).to(device)
    model = EncoderDecoderModel(encoder, decoder).to(device)    

    optimizer = optim.Adam(model.parameters(), lr=cfg["learning_rate"])  ### Adam is good and I am Adam I shall use Adam. Adpative something momentum.
    loss_fn = NN.CrossEntropyLoss(ignore_index=PAD_TOKEN_ID)  # So padding doesn't count 

    model.train()
    for epoch in range(cfg["epochs"]):
        for batch in train_loader:
            if batch is None:
                continue
            image, text, label = batch
            image = image.to(device)
            label = label.to(device)

            logits = model(image=image, text=text)
            loss = loss_fn(logits, label)
            optimizer.zero_grad()    ### resets gradients between batches/epochs
            loss.backward()   ### Backprop - Do learning
            optimizer.step()

        # Eval on test set each epoch
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                if batch is None:
                    continue
                image, text, label = batch
                image = image.to(device)
                label = label.to(device)
                logits = model(image=image, text=text)
                predictions = torch.argmax(logits, dim=-1)
                correct += (predictions == label).sum().item()
                total += label.size(0)

        accuracy = correct / total * 100
        print(f"Epoch {epoch+1}/{cfg['epochs']}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%")
        wandb.log({
            "loss": loss.item(),
            "epoch": epoch,
            "accuracy": accuracy
        })

    ### Save and upload model

    save_model(model, "ABSeeFood.pth")
    upload_model_to_wandb(model, run, "ABSeeFood.pth")
    wandb.finish()### Launch training


if __name__ == "__main__":
    training()


















### Imports
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn as NN
import torch.nn.functional as F
import torch.optim as optim
import wandb
import transformers as t
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import pandas as pd
import os
from torch.utils.data import random_split
from PIL import UnidentifiedImageError

c = t.CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

class HotdogDataset(Dataset):
    def __init__(self, csv_path, image_folder, transform=None):
        self.df = pd.read_csv(csv_path, encoding="latin1")
        self.image_folder = image_folder
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = None
        if pd.notna(row['imagename']):
            image_path = os.path.join(self.image_folder, f"{int(row['imagename'])}.jpg")
            try:
                image = Image.open(image_path).convert("RGB")
                if self.transform:
                    image = self.transform(image)
            except (FileNotFoundError, UnidentifiedImageError):
                image = None
        text = row['text'] if pd.notna(row['text']) else None
        label = int(row['label'])
        return image, text, label

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

dataset = HotdogDataset(csv_path="data/HotdogNotdog/dataset.csv", image_folder="data/HotdogNotdog/images", transform=transform)

# Safe collate to avoid NoneType

def collate_skip_none(batch):
    batch = [b for b in batch if b is not None and (b[0] is not None or b[1] is not None)]
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

config = {
    "embed_dim": 512,
    "epochs": 50,
    "learning_rate": 0.001,
    "batch_size": 128,
}

VOCAB_SIZE = 2
PAD_TOKEN_ID = -100

train_size = int(0.99 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=collate_skip_none)
val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], collate_fn=collate_skip_none)

class AttentionHead(NN.Module):
    def __init__(self, dim, embed_dim):
        super().__init__()
        self.q_proj = NN.Linear(dim, embed_dim)
        self.k_proj = NN.Linear(dim, embed_dim)
        self.v_proj = NN.Linear(dim, embed_dim)

    def forward(self, x_q, x_kv=None, mask=None):
        if x_kv is None:
            x_kv = x_q
        Q = self.q_proj(x_q)
        K = self.k_proj(x_kv)
        V = self.v_proj(x_kv)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (Q.shape[-1] ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        return torch.matmul(attn_weights, V)

class Decoder(NN.Module):
    def __init__(self, cfg):
        super().__init__()
        self.token_embed = NN.Embedding(VOCAB_SIZE, cfg["embed_dim"])
        self.attn = AttentionHead(cfg["embed_dim"], cfg["embed_dim"])
        self.mlp = NN.Sequential(
            NN.Linear(cfg["embed_dim"], cfg["embed_dim"] * 2),
            NN.ReLU(),
            NN.Linear(cfg["embed_dim"] * 2, cfg["embed_dim"])
        )
        self.norm1 = NN.LayerNorm(cfg["embed_dim"])
        self.norm2 = NN.LayerNorm(cfg["embed_dim"])
        self.classifier = NN.Linear(cfg["embed_dim"], 2)

    def forward(self, encoder_output):
        x = self.attn(self.norm1(encoder_output))
        x = self.mlp(self.norm2(x))
        return self.classifier(x)

class EncoderDecoderModel(NN.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.c = encoder
        self.decoder = decoder

    def forward(self, image=None, text=None):
        device = next(self.c.parameters()).device
        image_emb, text_emb = None, None
        if image is not None:
            inputs = processor(images=image, return_tensors="pt").to(device)
            image_emb = self.c.get_image_features(**inputs)
        if text is not None:
            inputs = processor(text=text, return_tensors="pt", padding=True).to(device)
            text_emb = self.c.get_text_features(**inputs)
        if image_emb is not None and text_emb is not None:
            encoder_output = (image_emb + text_emb) / 2
        elif image_emb is not None:
            encoder_output = image_emb
        elif text_emb is not None:
            encoder_output = text_emb
        else:
            raise ValueError("No input provided to encoder.")
        return self.decoder(encoder_output)

def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def upload_model_to_wandb(model, run, path):
    artifact = wandb.Artifact("ABSeeFood", type="model")
    artifact.add_file(path)
    run.log_artifact(artifact)
    print("Model uploaded to W&B")

def training():
    run = wandb.init(project="MLXMultimodal", config=config)
    cfg = wandb.config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = c.to(device)
    decoder = Decoder(cfg).to(device)
    model = EncoderDecoderModel(encoder, decoder).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    loss_fn = NN.CrossEntropyLoss(ignore_index=PAD_TOKEN_ID)
    model.train()
    for epoch in range(cfg["epochs"]):
        for batch in train_loader:
            if batch is None:
                continue
            image, text, label = batch
            image = image.to(device) if image is not None else None
            label = label.to(device)
            logits = model(image=image, text=text)
            loss = loss_fn(logits, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                if batch is None:
                    continue
                image, text, label = batch
                image = image.to(device) if image is not None else None
                label = label.to(device)
                logits = model(image=image, text=text)
                predictions = torch.argmax(logits, dim=-1)
                correct += (predictions == label).sum().item()
                total += label.size(0)
        accuracy = correct / total * 100
        print(f"Epoch {epoch+1}/{cfg['epochs']}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%")
        wandb.log({"loss": loss.item(), "epoch": epoch, "accuracy": accuracy})
    save_model(model, "ABSeeFood.pth")
    upload_model_to_wandb(model, run, "ABSeeFood.pth")
    wandb.finish()

if __name__ == "__main__":
    training()




    ### Imports
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn as NN
import torch.nn.functional as F
import torch.optim as optim
import wandb
import transformers as t
from transformers import CLIPProcessor, CLIPModel
from PIL import Image, UnidentifiedImageError
import pandas as pd
import os

### Load encoder + processor
c = t.CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

### Dataset
class HotdogDataset(Dataset):
    def __init__(self, csv_path, image_folder, transform=None):
        self.df = pd.read_csv(csv_path, encoding="latin1")
        self.image_folder = image_folder
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.join(self.image_folder, f"{int(row['imagename'])}.jpg")
        try:
            image = Image.open(image_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
        except (FileNotFoundError, UnidentifiedImageError):
            image = torch.zeros(3, 224, 224)

        text = row['text'] if pd.notna(row['text']) else ""
        label = int(row['label'])

        return image, text, label

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

### Dataset & Dataloaders
dataset = HotdogDataset(csv_path="data/HotdogNotdog/dataset.csv", image_folder="data/HotdogNotdog/images", transform=transform)
config = {"embed_dim": 512, "epochs": 50, "learning_rate": 0.001, "batch_size": 128}
VOCAB_SIZE = 2
PAD_TOKEN_ID = -100

train_size = int(0.99 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config["batch_size"]) 

### Attention
class AttentionHead(NN.Module):
    def __init__(self, dim, embed_dim):
        super().__init__()
        self.q_proj = NN.Linear(dim, embed_dim)
        self.k_proj = NN.Linear(dim, embed_dim)
        self.v_proj = NN.Linear(dim, embed_dim)

    def forward(self, x_q, x_kv=None, mask=None):
        if x_kv is None:
            x_kv = x_q 
        Q = self.q_proj(x_q)
        K = self.k_proj(x_kv)
        V = self.v_proj(x_kv)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (Q.shape[-1] ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        return torch.matmul(attn_weights, V)

### Decoder
class Decoder(NN.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn = AttentionHead(cfg["embed_dim"], cfg["embed_dim"])
        self.mlp = NN.Sequential(
            NN.Linear(cfg["embed_dim"], cfg["embed_dim"] * 2),
            NN.ReLU(),
            NN.Linear(cfg["embed_dim"] * 2, cfg["embed_dim"])
        )
        self.norm1 = NN.LayerNorm(cfg["embed_dim"])
        self.norm2 = NN.LayerNorm(cfg["embed_dim"])
        self.classifier = NN.Linear(cfg["embed_dim"], 2)

    def forward(self, encoder_output):
        x = self.attn(self.norm1(encoder_output))
        x = self.mlp(self.norm2(x))
        return self.classifier(x)

### Encoder-Decoder model
class EncoderDecoderModel(NN.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.c = encoder
        self.decoder = decoder
        self.processor = processor
        self.register_buffer("neutral_image_emb", self._embed_image("data/HotdogNotdog/images/1.jpg"))
        self.register_buffer("neutral_text_emb", self._embed_text(" "))

    def _embed_image(self, path):
        img = transform(Image.open(path).convert("RGB"))
        inputs = self.processor(images=img.unsqueeze(0), return_tensors="pt")
        return self.c.get_image_features(**inputs)

    def _embed_text(self, txt):
        inputs = self.processor(text=[txt], return_tensors="pt")
        return self.c.get_text_features(**inputs)

    def forward(self, image, text):
        device = next(self.c.parameters()).device

        inputs = self.processor(text=text, images=image, return_tensors="pt", padding=True).to(device)
        image_emb = self.c.get_image_features(**inputs)
        text_emb = self.c.get_text_features(**inputs)

        image_mask = ~torch.all(image_emb == self.neutral_image_emb, dim=1)
        text_mask = ~torch.all(text_emb == self.neutral_text_emb, dim=1)

        encoder_output = torch.zeros_like(image_emb)
        both = image_mask & text_mask
        encoder_output[both] = (image_emb[both] + text_emb[both]) / 2
        encoder_output[image_mask & ~text_mask] = image_emb[image_mask & ~text_mask]
        encoder_output[~image_mask & text_mask] = text_emb[~image_mask & text_mask]

        return self.decoder(encoder_output)

### Training

def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def upload_model_to_wandb(model, run, path):
    artifact = wandb.Artifact("ABSeeFood", type="model")  
    artifact.add_file(path)
    run.log_artifact(artifact)
    print("Model uploaded to W&B")

def training():
    run = wandb.init(project="MLXMultimodal", config=config)
    cfg = wandb.config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = c.to(device)
    decoder = Decoder(cfg).to(device)
    model = EncoderDecoderModel(encoder, decoder).to(device)

    optimizer = optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    loss_fn = NN.CrossEntropyLoss(ignore_index=PAD_TOKEN_ID)

    model.train()
    for epoch in range(cfg["epochs"]):
        for image, text, label in train_loader:
            image = image.to(device)
            label = label.to(device)
            logits = model(image=image, text=text)
            loss = loss_fn(logits, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for image, text, label in val_loader:
                image = image.to(device)
                label = label.to(device)
                logits = model(image=image, text=text)
                predictions = torch.argmax(logits, dim=-1)
                correct += (predictions == label).sum().item()
                total += label.size(0)

        acc = correct / total * 100
        print(f"Epoch {epoch+1}/{cfg['epochs']}, Loss: {loss.item():.4f}, Accuracy: {acc:.2f}%")
        wandb.log({"loss": loss.item(), "epoch": epoch, "accuracy": acc})

    save_model(model, "ABSeeFood.pth")
    upload_model_to_wandb(model, run, "ABSeeFood.pth")
    wandb.finish()

if __name__ == "__main__":
    training()







    ### Imports
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn as NN
import torch.nn.functional as F
import torch.optim as optim
import wandb
import transformers as t
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import pandas as pd
import os
from torch.utils.data import random_split
from PIL import UnidentifiedImageError

### choosing CLIP as dealing with a single image vector seems simplier and also closer to the target diagram
c = t.CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

class HotdogDataset(Dataset):
    def __init__(self, csv_path, image_folder, transform=None):
        self.df = pd.read_csv(csv_path, encoding="latin1")
        self.image_folder = image_folder
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_path = os.path.join(self.image_folder, f"{int(row['imagename'])}.jpg")
        try:
            image = Image.open(image_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
        except (FileNotFoundError, UnidentifiedImageError):
            image = torch.zeros(3, 224, 224)

        text = row['text'] if pd.notna(row['text']) else ""
        label = int(row['label'])

        return image, text, label

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

dataset = HotdogDataset(csv_path="data/HotdogNotdog/dataset.csv", image_folder="data/HotdogNotdog/images", transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Custom collate function
collate_skip_none = torch.utils.data.dataloader.default_collate

### Config (sweep-friendly, but not sweeping yet
config = {
    "embed_dim": 512,
    "epochs": 50,
    "learning_rate": 0.001,
    "batch_size": 128,
}

VOCAB_SIZE = 2
PAD_TOKEN_ID = -100

train_size = int(0.99 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=collate_skip_none)
val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], collate_fn=collate_skip_none)

class AttentionHead(NN.Module):
    def __init__(self, dim, embed_dim):
        super().__init__()
        self.q_proj = NN.Linear(dim, embed_dim)
        self.k_proj = NN.Linear(dim, embed_dim)
        self.v_proj = NN.Linear(dim, embed_dim)

    def forward(self, x_q, x_kv=None, mask=None):
        if x_kv is None:
            x_kv = x_q 
        Q = self.q_proj(x_q)
        K = self.k_proj(x_kv)
        V = self.v_proj(x_kv)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (Q.shape[-1] ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        return torch.matmul(attn_weights, V)

class Decoder(NN.Module):
    def __init__(self, cfg):
        super().__init__()
        self.token_embed = NN.Embedding(VOCAB_SIZE, cfg["embed_dim"])
        self.attn = AttentionHead(cfg["embed_dim"], cfg["embed_dim"])
        self.mlp = NN.Sequential(
            NN.Linear(cfg["embed_dim"], cfg["embed_dim"] * 2),
            NN.ReLU(),
            NN.Linear(cfg["embed_dim"] * 2, cfg["embed_dim"])
        )
        self.norm1 = NN.LayerNorm(cfg["embed_dim"])
        self.norm2 = NN.LayerNorm(cfg["embed_dim"])
        self.classifier = NN.Linear(cfg["embed_dim"], 2)

    def forward(self, encoder_output):
        x = self.attn(self.norm1(encoder_output))
        x = self.mlp(self.norm2(x))
        return self.classifier(x)

class EncoderDecoderModel(NN.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.c = encoder
        self.decoder = decoder

    def forward(self, image, text):
        device = next(self.c.parameters()).device

        has_image = image.sum(dim=(1, 2, 3)) > 0
        has_text = torch.tensor([bool(t.strip()) for t in text], device=device)

        batch_size = image.shape[0]
        encoder_output = torch.zeros((batch_size, 512), device=device)

        for i in range(batch_size):
            if has_image[i] and has_text[i]:
                img_inputs = processor(images=image[i], return_tensors="pt").to(device)
                txt_inputs = processor(text=[text[i]], return_tensors="pt", padding=True).to(device)
                img_feat = self.c.get_image_features(**img_inputs)
                txt_feat = self.c.get_text_features(**txt_inputs)
                encoder_output[i] = (img_feat + txt_feat) / 2
            elif has_image[i]:
                img_inputs = processor(images=image[i], return_tensors="pt").to(device)
                encoder_output[i] = self.c.get_image_features(**img_inputs)
            elif has_text[i]:
                txt_inputs = processor(text=[text[i]], return_tensors="pt").to(device)
                encoder_output[i] = self.c.get_text_features(**txt_inputs)

        return self.decoder(encoder_output)

def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def upload_model_to_wandb(model, run, path):
    artifact = wandb.Artifact("ABSeeFood", type="model")  
    artifact.add_file(path)
    run.log_artifact(artifact)
    print("Model uploaded to W&B")

def training():
    run = wandb.init(project="MLXMultimodal", config=config)
    cfg = wandb.config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = c.to(device)
    decoder = Decoder(cfg).to(device)
    model = EncoderDecoderModel(encoder, decoder).to(device)    

    optimizer = optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    loss_fn = NN.CrossEntropyLoss(ignore_index=PAD_TOKEN_ID)

    model.train()
    for epoch in range(cfg["epochs"]):
        for batch in train_loader:
            image, text, label = batch
            image = image.to(device)
            label = label.to(device)

            logits = model(image=image, text=text)
            loss = loss_fn(logits, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                image, text, label = batch
                image = image.to(device)
                label = label.to(device)
                logits = model(image=image, text=text)
                predictions = torch.argmax(logits, dim=-1)
                correct += (predictions == label).sum().item()
                total += label.size(0)

        accuracy = correct / total * 100
        print(f"Epoch {epoch+1}/{cfg['epochs']}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%")
        wandb.log({
            "loss": loss.item(),
            "epoch": epoch,
            "accuracy": accuracy
        })

    save_model(model, "ABSeeFood.pth")
    upload_model_to_wandb(model, run, "ABSeeFood.pth")
    wandb.finish()

if __name__ == "__main__":
    training()