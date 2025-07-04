import random

from datasets import Dataset, load_dataset
from pandas import DataFrame
import torch
from torch import nn
from transformers import (
    AutoModel,
    AutoProcessor,
    AutoTokenizer,
)


CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

SEED = 42


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # pull CLIP model to serve as img encoder
    clip = AutoModel.from_pretrained(CLIP_MODEL_NAME, torch_dtype=torch.float16, device_map="auto").eval()
    clip_processor = AutoProcessor.from_pretrained(CLIP_MODEL_NAME, device_map="auto")

    # load entire Flickr30k dataset (only available in one split)
    flickr_ds: Dataset = load_dataset("nlphuji/flickr30k", split="test")  # type: ignore
    print(f"Loaded full Flickr30k dataset: {flickr_ds}")

    # split into train, val, and test sets using HF dataset filtering (more efficient)
    train_ds = flickr_ds.filter(lambda x: x["split"] == "train")  # 29k samples
    val_ds = flickr_ds.filter(lambda x: x["split"] == "val")  # ~ 1k samples
    test_ds = flickr_ds.filter(lambda x: x["split"] == "test")  # 1k samples

    print(f"Train set size: {len(train_ds)}")
    print(f"Validation set size: {len(val_ds)}")
    print(f"Test set size: {len(test_ds)}")

    # inspect a random row from the test set
    i = random.randint(0, len(test_ds) - 1)
    sample = test_ds[i]
    image = sample["image"]
    print(f"\nInspecting random sample {i + 1}:")
    print(f"Image shape: {image.size}")
    print(f"Image mode: {image.mode}")
    print(f"Captions: {sample['caption']}")
    image.show()

    # process/tokenize and encode the image using CLIP vision model (exemplar)
    inputs = clip_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = clip.vision_model(
            pixel_values=inputs["pixel_values"],
            output_hidden_states=True,
        )

    hidden = outputs["last_hidden_state"]
    print(f"CLIP hidden state shape: {hidden.shape}")  # (1, 196, 768)
    clip_d = hidden.size(2)  # 50 for 32x32 / 196 for 16x16 (i.e. num of patches)
    print(f"CLIP overall model dimension: {clip_d}")
    clip_l = hidden.size(1)  # 768 (i.e. encoder model dimension)
    print(f"CLIP layers (i.e. num of patches): {clip_l}")

    def encode(batch):
        # PIL → tensor → CLIP → float16 numpy
        inputs = clip_processor(images=batch["image"], return_tensors="pt").to(device, torch.float16)
        with torch.no_grad():
            outputs = clip.vision_model(
                pixel_values=inputs["pixel_values"],
                output_hidden_states=True,
            )
            h = outputs["last_hidden_state"].cpu().numpy().astype("float16")
        return {"clip_embed": h}
    
    # encode the entire dataset, removing image column to keep it compact (or do we need it??)
    train_ds = train_ds.map(
        encode,
        batched=True,
        batch_size=64,
        remove_columns=["image"],
        num_proc=1,
        desc=f"Encoding train set with {CLIP_MODEL_NAME}",
      )


def encode(batch, processor):


if __name__ == "__main__":
    print("Running dataset preparation...")
    main()
