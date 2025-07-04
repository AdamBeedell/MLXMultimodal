import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import CLIPProcessor, CLIPModel, AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # ignore warnings from huggingface
import evaluate # for BLEU from huggingface
import gc


# 1. Load the Flickr30k dataset from Hugging Face
ds = load_dataset("nlphuji/flickr30k")
# # try to check the number of samples in train/val/test
# from collections import Counter
# split_counts = Counter(ds['test']['split'])
# print(split_counts)  # train/val/test: 29000/1014/1000
### reduce size
ds = ds['test'].shuffle(seed=42).select(range(500))



# because now for each row we have one image and 5 captions, we want to flatten it to 5 rows/pairs of image+caption
def flatten_dataset(batch):
    images = []
    captions = []
    splits = []
    sentids = []
    img_ids = []
    filenames = []
    for img,caps,spl,sids,img_id,fname in tqdm(zip(
        batch['image'], batch['caption'], batch['split'], batch['sentids'], batch['img_id'], batch['filename']
        )):                            # iterates over each row/image
        for cap,sid in zip(caps,sids): # iterates over each caption; 5 captions for each image
            images.append(img)
            captions.append(cap)
            splits.append(spl)
            sentids.append(sid)
            img_ids.append(img_id)
            filenames.append(fname)
    return {
        'image':images, 'caption':captions,'sentids': sentids,
        'split': splits, 'img_id': img_ids, 'filename': filenames
        }

flat_ds = ds.map(flatten_dataset, batched=True)
    

# 2. Split the dataset into train:val:test according to the pre-defined column of 'split' in the huggingface dataset
train_ds_1 = flat_ds.filter(lambda d: d['split']=='train', keep_in_memory=True)
val_ds_1 = flat_ds.filter(lambda d: d['split']=='val', keep_in_memory=True)
test_ds_1 = flat_ds.filter(lambda d: d['split']=='test', keep_in_memory=True)


# 3. Load CLIP's vision encoder and processor
clip_model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32') # the neural net itself and the forward model; 
# CLIP is essentially a two-tower mnultimodal encoder, has no decoder, to compare the similarity (contrastive learning) between the image and text representations
# # to see what's inside of the model
# print(vision_encoder)
clip_processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32', use_fast=True) # the pre-processing methods for images and text, here we only use it to pre-process images
vision_encoder = clip_model.vision_model
for param in vision_encoder.parameters():
    param.requires_grad = False  # freeze vision encoder


# 4. Load QWen's decoder and tokenizer
qwen_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B-Base") # QWen is a generative model, same as GPT, so it only has decoder no encoder, the entire model is a decoder
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base") # to tokenize the captions/text
# resize the tokenizer to make sure the dimension matches
qwen_model.resize_token_embeddings(len(tokenizer))
decoder = qwen_model
for param in decoder.parameters():
    param.requires_grad = False  
# only fine-tune the last two layers of QWen
for block in decoder.model.layers[-2:]:
    for param in block.parameters():
        param.requires_grad = True

if tokenizer.bos_token_id is None:
    tokenizer.bos_token_id = tokenizer.eos_token_id  # or another valid token id


# 5. Define a projection layer to map vision features to decoder input; make sure encoder output and decoder input are in the same space before glue them together 
vision_feature_dim = vision_encoder.config.hidden_size
decoder_embed_dim = decoder.model.embed_tokens.embedding_dim
### build the architecture
class MultimodalCaptionModel(nn.Module):
    # define the architecture of this neural net, sets up the layers and parameters
    def __init__(self, vision_encoder, decoder, vision_feature_dim, decoder_embed_dim):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.decoder = decoder
        # Project vision features to decoder's embedding space
        self.proj = nn.Linear(vision_feature_dim, decoder_embed_dim)
    # defines the computations performed at every call, how information flows through the model
    def forward(self, pixel_values, input_ids, attention_mask=None, labels=None):
        # Extract vision features (all tokens, not just CLS)
        vision_outputs = self.vision_encoder(pixel_values)[0]  # shape: (batch, num_vision_tokens, vision_feature_dim)
        vision_embeds = self.proj(vision_outputs)  # shape: (batch, num_vision_tokens, decoder_embed_dim)
        # Concatenate vision_embeds to input embeddings
        inputs_embeds = self.decoder.model.embed_tokens(input_ids)  # (batch, seq_len, decoder_embed_dim)
        inputs_embeds = torch.cat([vision_embeds, inputs_embeds], dim=1)  # prepend all vision tokens
        # Adjust attention mask
        if attention_mask is not None:
            vision_mask = torch.ones((attention_mask.shape[0], vision_embeds.shape[1]), dtype=attention_mask.dtype, device=attention_mask.device)
            attention_mask = torch.cat([vision_mask, attention_mask], dim=1)
        # Forward through decoder
        outputs = self.decoder(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        logits = outputs[0]
        shift_logits = logits[:, vision_embeds.shape[1]:-1, :].contiguous()  # ignore vision tokens and last token
        shift_labels = labels[:, :-1].contiguous()  # ignore the last label
        loss_fct = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        class Output:
            pass
        out = Output()
        out.logits = logits
        out.loss = loss
        return out


# 6. Preprocessing function for dataset
def preprocess(example):
    # Preprocess image
    image = example['image']
    processed = clip_processor(images=image, return_tensors="pt")
    # check out what are there in the processed tensor
    # print(processed.keys())
    pixel_values = processed['pixel_values'][0]
    # Preprocess text
    caption = example['caption']
    # tokenizer returns a dict with keys like 'input_ids','attention_mask' etc
    tokens = tokenizer(caption, padding='max_length', truncation=True, max_length=32, return_tensors='pt')
    input_ids = tokens['input_ids'][0] # input_ids is a tensor of shape (1,max_length), so [0] extract the first and only sequence
    attention_mask = tokens['attention_mask'][0] # here attention means actual tokens or not
    labels = input_ids.clone()
    return {
        'pixel_values': pixel_values,
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }


# 7. Apply preprocessing to datasets; will always convert data to a list for serialization and storage
train_ds = train_ds_1.map(preprocess, load_from_cache_file=False)
val_ds = val_ds_1.map(preprocess, load_from_cache_file=False)
test_ds = test_ds_1.map(preprocess, load_from_cache_file=False)


# 8. DataLoader
batch_size = 48
# convert a list of individual tensors into a single batched tensor; add the stack axis as the first dimension
def collate_fn(batch):
    def to_tensor(x):
        if isinstance(x, torch.Tensor):
            return x
        return torch.tensor(x)
    return {
        'pixel_values': torch.stack([to_tensor(x['pixel_values']) for x in batch]),
        'input_ids': torch.stack([to_tensor(x['input_ids']) for x in batch]),
        'attention_mask': torch.stack([to_tensor(x['attention_mask']) for x in batch]),
        'labels': torch.stack([to_tensor(x['labels']) for x in batch]),
    }
# shuffle samples for training to aviod model learn the order of samples, which is useless
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
# not shuffle samples to ensure the same validation dataset for each epoch, to make the comparison between epochs fair
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MultimodalCaptionModel(vision_encoder, decoder, vision_feature_dim, decoder_embed_dim).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# train/fine-tune the projection layer (project from the visual space to the text space)
for param in model.proj.parameters():
    param.requires_grad = True

# 9. Training and validation loops with model saving/loading
num_epochs = 20  # You can increase this for better results
save_path = './multimodal_caption_model_small.pt'

best_val_loss = float('inf')
# using more metrics
bleu_metric = evaluate.load('bleu')

for epoch in range(num_epochs): # iterates over epochs
    # Training
    model.train()
    train_loss = 0.0
    for batch in tqdm(train_loader, desc=f'Training Epoch {epoch+1}'): # iterates over batches
        for k in batch: # iterates over keys in each batch dict, to convert them to(device)
            batch[k] = batch[k].to(device)
        outputs = model(**batch) # ** is unpacking operator for dict, passing each key-value pair as a named argument to the function (that's why you didn't see the explicit input) 
        loss = outputs.loss
        loss.backward()       # compute gradients and accumulate them in each parameter's .grad attribute
        optimizer.step()      # update model parameters using gradients
        optimizer.zero_grad() # clear gradients for the next iteration, so they don't accumulate across batches [by default, Pytorch accumulate gradients]
        train_loss += loss.item() * batch['pixel_values'].size(0)  # loss.item() returns the averaged loss per sample. so we multiple number of sample to convert it back to the total loss in this batch
    train_loss /= len(train_loader.dataset) 
    # it's necessary to first multiple the loss by number of images in the batch, then at the very end, divide by number of all images. Because each batch may have different number of images, especially the last batch

    # Validation
    model.eval()
    val_loss = 0.0
    val_predictions = []
    val_references = []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f'Validation Epoch {epoch+1}'):
            for k in batch:
                batch[k] = batch[k].to(device)
            outputs = model(**batch)
            loss = outputs.loss
            val_loss += loss.item() * batch['pixel_values'].size(0)
            # add other metrics
            generated_captions = []
            for i in range(batch['pixel_values'].size(0)):
                pixel_values = batch['pixel_values'][i].unsqueeze(0)
                # Use all vision tokens, not just CLS
                vision_outputs = model.vision_encoder(pixel_values)[0]  # (1, num_vision_tokens, vision_feature_dim)
                vision_embeds = model.proj(vision_outputs)  # (1, num_vision_tokens, decoder_embed_dim)
                input_ids = torch.full((1, 1), tokenizer.bos_token_id, dtype=torch.long, device=device)
                generated = input_ids
                # Prepend all vision_embeds ONCE
                for _ in range(32):
                    inputs_embeds = model.decoder.model.embed_tokens(generated)
                    inputs_embeds = torch.cat([vision_embeds, inputs_embeds], dim=1)
                    outputs = model.decoder(inputs_embeds=inputs_embeds)
                    next_token_logits = outputs.logits[:, -1, :]
                    # select one of the top 5 predictions to aviod stuck/repeat the top one (greedy decoding)
                    next_token = torch.topk(next_token_logits, k=5, dim=-1).indices[:, torch.randint(0, 5, (1,)).item()].unsqueeze(1)
                    generated = torch.cat([generated, next_token], dim=1)
                    # Stop if EOS generated for all
                    if (next_token == tokenizer.eos_token_id).all():
                        break
                caption = tokenizer.batch_decode(generated, skip_special_tokens=True)
                generated_captions.extend(caption)
            val_predictions.extend(generated_captions)
            references = tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)
            val_references.extend(references)
    val_loss /= len(val_loader.dataset)
    print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

    # Print 5 sample predictions and references for debugging
    print("==================Sample predictions and references:====================")
    for pred, ref in zip(val_predictions[0:26:5], val_references[0:26:5]):
        print(f"Predicted caption: {pred}")
        print(f"Reference caption: {ref}")
        print("----------------------------------")

    # BLEU expects references as list of lists of tokens
    bleu_score = bleu_metric.compute(predictions=val_predictions, references=[[ref] for ref in val_references])['bleu']
    print(f"Validation BLEU: {bleu_score:.4f}")

    # Save the best model  -- Smart!
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

# After dataset preparation, free up the original datasets

del ds
gc.collect()

# After training and validation, before test evaluation

del train_ds, val_ds, train_ds_1, val_ds_1, train_loader, val_loader

gc.collect()

# 10. Test: generate captions for test images
model.eval()
all_captions = []
all_refs = []
with torch.no_grad():
    for batch in tqdm(DataLoader(test_ds, batch_size=batch_size, collate_fn=collate_fn), desc='Testing'):
        pixel_values = batch['pixel_values'].to(device)
        # Use all vision tokens, not just CLS
        vision_outputs = model.vision_encoder(pixel_values)[0]  # (batch, num_vision_tokens, vision_feature_dim)
        vision_embeds = model.proj(vision_outputs)  # (batch, num_vision_tokens, decoder_embed_dim)
        input_ids = torch.full((pixel_values.size(0), 1), tokenizer.bos_token_id, dtype=torch.long, device=device)
        generated = input_ids
        # Prepend all vision_embeds ONCE
        for _ in range(32):
            inputs_embeds = model.decoder.model.embed_tokens(generated)
            inputs_embeds = torch.cat([vision_embeds, inputs_embeds], dim=1)
            outputs = model.decoder(inputs_embeds=inputs_embeds)
            next_token_logits = outputs.logits[:, -1, :]
            # select one of the top 5 predictions to aviod stuck/repeat the top one (greedy decoding)
            next_token = torch.topk(next_token_logits, k=5, dim=-1).indices[:, torch.randint(0, 5, (1,)).item()].unsqueeze(1)
            generated = torch.cat([generated, next_token], dim=1)
            # Stop if EOS generated for all
            finished = (next_token.squeeze(1) == tokenizer.eos_token_id)
            if finished.all():
                break
        captions = tokenizer.batch_decode(generated, skip_special_tokens=True)
        all_captions.extend(captions)
        # Reference captions for evaluation (decode batch['labels'])
        references = tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)
        all_refs.extend(references)
    # Print a few generated captions
    print("Sample generated captions:")
    for i in range(3):
        print(f"Generated: {all_captions[i]}")

# Prepare references for metrics
# If you have only one reference per image:
references = [[ref] for ref in all_refs]  # BLEU expect list of lists
# Compute BLEU
bleu_score = bleu_metric.compute(predictions=all_captions, references=references)['bleu']
print(f"Test BLEU: {bleu_score:.4f}")

# After test evaluation, free up test data and model

del test_ds, test_ds_1, flat_ds, model, decoder, vision_encoder, clip_model, qwen_model, tokenizer, clip_processor

gc.collect()
