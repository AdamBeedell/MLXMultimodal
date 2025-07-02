import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import CLIPProcessor, CLIPModel, AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import os

# 1. Load the Flickr30k dataset from Hugging Face
ds = load_dataset("nlphuji/flickr30k")

# 2. Split the dataset into train:val:test = 8:1:1
# We'll use the 'train' split and split it further
split_ds = ds['train'].train_test_split(test_size=0.2, seed=42)
train_val = split_ds['train'].train_test_split(test_size=0.1111, seed=42)  # 0.1111*0.8 ≈ 0.0888
train_ds = train_val['train']
val_ds = train_val['test']
test_ds = split_ds['test']

# 3. Load CLIP's vision encoder and processor
clip_model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
clip_processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
vision_encoder = clip_model.vision_model
for param in vision_encoder.parameters():
    param.requires_grad = False  # freeze vision encoder

# 4. Load QWen's decoder and tokenizer
qwen_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B-Base")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")
decoder = qwen_model.model
for param in decoder.parameters():
    param.requires_grad = True  # train decoder

# 5. Define a projection layer to map vision features to decoder input
def get_vision_feature_dim():
    dummy = torch.zeros(1, 3, 224, 224)
    with torch.no_grad():
        features = vision_encoder(dummy)
    return features[0].shape[-1]

vision_feature_dim = get_vision_feature_dim()
decoder_embed_dim = decoder.embed_tokens.embedding_dim

class MultimodalCaptionModel(nn.Module):
    def __init__(self, vision_encoder, decoder, vision_feature_dim, decoder_embed_dim):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.decoder = decoder
        # Project vision features to decoder's embedding space
        self.proj = nn.Linear(vision_feature_dim, decoder_embed_dim)
    def forward(self, pixel_values, input_ids, attention_mask=None, labels=None):
        # Extract vision features
        vision_outputs = self.vision_encoder(pixel_values)[0][:, 0, :]  # CLS token
        vision_embeds = self.proj(vision_outputs)
        # Use vision_embeds as prefix for decoder
        # Concatenate vision_embeds to input embeddings
        inputs_embeds = self.decoder.embed_tokens(input_ids)
        # Prepend vision_embeds to the sequence
        vision_embeds = vision_embeds.unsqueeze(1)  # (batch, 1, embed_dim)
        inputs_embeds = torch.cat([vision_embeds, inputs_embeds], dim=1)
        # Adjust attention mask
        if attention_mask is not None:
            vision_mask = torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype, device=attention_mask.device)
            attention_mask = torch.cat([vision_mask, attention_mask], dim=1)
        # Forward through decoder
        outputs = self.decoder(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)
        return outputs

# 6. Preprocessing function for dataset
def preprocess(example):
    # Preprocess image
    image = example['image']
    processed = clip_processor(images=image, return_tensors="pt")
    pixel_values = processed['pixel_values'][0]
    # Preprocess text
    caption = example['sentence']
    tokens = tokenizer(caption, padding='max_length', truncation=True, max_length=32, return_tensors='pt')
    input_ids = tokens['input_ids'][0]
    attention_mask = tokens['attention_mask'][0]
    labels = input_ids.clone()
    return {
        'pixel_values': pixel_values,
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }

# 7. Apply preprocessing to datasets
train_ds = train_ds.map(preprocess)
val_ds = val_ds.map(preprocess)
test_ds = test_ds.map(preprocess)

# 8. DataLoader
batch_size = 8
def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'input_ids': torch.stack([x['input_ids'] for x in batch]),
        'attention_mask': torch.stack([x['attention_mask'] for x in batch]),
        'labels': torch.stack([x['labels'] for x in batch]),
    }
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MultimodalCaptionModel(vision_encoder, decoder, vision_feature_dim, decoder_embed_dim).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# 9. Training and validation loops with model saving/loading
num_epochs = 5  # You can increase this for better results
save_path = 'multimodal_caption_model.pt'

best_val_loss = float('inf')

for epoch in range(num_epochs):
    # Training
    model.train()
    train_loss = 0.0
    for batch in tqdm(train_loader, desc=f'Training Epoch {epoch+1}'):
        for k in batch:
            batch[k] = batch[k].to(device)
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_loss += loss.item() * batch['pixel_values'].size(0)
    train_loss /= len(train_loader.dataset)

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f'Validation Epoch {epoch+1}'):
            for k in batch:
                batch[k] = batch[k].to(device)
            outputs = model(**batch)
            loss = outputs.loss
            val_loss += loss.item() * batch['pixel_values'].size(0)
    val_loss /= len(val_loader.dataset)
    print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

    # Save the best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), save_path)
        print(f"Best model saved at epoch {epoch+1} with val loss {val_loss:.4f}")

# Load the best model for testing
if os.path.exists(save_path):
    model.load_state_dict(torch.load(save_path, map_location=device))
    print(f"Loaded best model from {save_path}")
else:
    print("No saved model found, using last epoch model.")

# 10. Test: generate captions for test images
model.eval()
all_captions = []
all_refs = []
with torch.no_grad():
    for batch in tqdm(DataLoader(test_ds, batch_size=batch_size, collate_fn=collate_fn), desc='Testing'):
        pixel_values = batch['pixel_values'].to(device)
        vision_outputs = model.vision_encoder(pixel_values)[0][:, 0, :]
        vision_embeds = model.proj(vision_outputs)
        input_ids = torch.full((pixel_values.size(0), 1), tokenizer.bos_token_id, dtype=torch.long, device=device)
        generated = input_ids
        for _ in range(32):
            inputs_embeds = model.decoder.embed_tokens(generated)
            inputs_embeds = torch.cat([vision_embeds.unsqueeze(1), inputs_embeds], dim=1)
            outputs = model.decoder(inputs_embeds=inputs_embeds)
            next_token_logits = outputs.logits[:, -1, :]
            next_token = next_token_logits.argmax(-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
        captions = tokenizer.batch_decode(generated, skip_special_tokens=True)
        all_captions.extend(captions)
        # Reference captions for evaluation
        all_refs.extend([ex['sentence'] for ex in batch['labels'].cpu().numpy()])
    # Print a few generated captions
    print("Sample generated captions:")
    for i in range(3):
        print(f"Generated: {all_captions[i]}")
# You can add BLEU or other metrics for evaluation if desired.
# The script now supports real training, validation, test, and model saving/loading, while remaining simple and well-commented for learning. 