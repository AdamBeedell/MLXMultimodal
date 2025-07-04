import random

from datasets import Dataset, load_dataset
from pandas import DataFrame
import torch
from torch import nn
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    # Qwen2_5_VLForConditionalGeneration,
)


# could use CLIP w/ patch sizes of 32x32 to speed up training by ~4x
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
QWEN_LLM_MODEL_NAME = "Qwen/Qwen3-0.6B-base"

SEED = 42


class Decoder(nn.Module):
    def __init__(self, clip_d: int, clip_l: int):
        super().__init__()
        self.clip_d = clip_d
        self.clip_l = clip_l
        self.mha = nn.MultiheadAttention(embed_dim=clip_d, num_heads=8, batch_first=True)

    def forward(self, x):
        pass


class CaptionGenerator(nn.Module):
    def __init__(
        self,
        clip_d: int,
        clip_l: int,
        qwen_d: int,
        vocab_size: int,
    ):
        super().__init__()
        self.clip_d = clip_d
        self.clip_l = clip_l
        self.qwen_d = qwen_d
        self.vocab_size = vocab_size

        self.vision_proj = nn.Linear(clip_d, qwen_d)  # project image features up to match text token dim
        self.decoder = Decoder(clip_d, clip_l)
        self.output_layer = nn.Linear(clip_d, vocab_size)

    def forward(self, img, caption):
        # img: (batch_size, clip_l, clip_d)
        # caption: (batch_size, seq_len)
        img_proj = self.vision_proj(img)
        pass


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # pull CLIP model to serve as img encoder
    clip = AutoModel.from_pretrained(CLIP_MODEL_NAME, torch_dtype=torch.float16, device_map="auto").eval()
    clip_processor = AutoProcessor.from_pretrained(CLIP_MODEL_NAME, device_map="auto")

    # load entire Flickr30k dataset (only available in one split)
    flickr_ds: Dataset = load_dataset("nlphuji/flickr30k", split="test")  # type: ignore
    print(f"Loaded full Flickr30k dataset: {flickr_ds}")

    # convert to pandas DataFrame for easier inspection
    df: DataFrame = flickr_ds.to_pandas()  # type: ignore
    print(f"Flickr30k dataset as DataFrame:\n{df.head(10)}")

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

    # process/tokenize and encode the image using CLIP vision model
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

    # so now we have the final hidden state for a random image from flickr30k, after encoding with CLIP vision transformer
    # we want to train a custom decoder so that if we feed it the encoded image and the start token, feed the result back in,
    # and so on, then it will eventually generate a reasonable caption!
    # therefore we'll need to use 'teacher forcing' with masking to learn this 'causal' predictive effect

    # get Qwen LLM model/tokenizer - see https://huggingface.co/Qwen/Qwen3-0.6B#quickstart
    qwen_tok = AutoTokenizer.from_pretrained(QWEN_LLM_MODEL_NAME)

    # don't run this on CPU!
    # qwen_base = AutoModelForCausalLM.from_pretrained(QWEN_LLM_MODEL_NAME, torch_dtype="auto", device_map="auto")


def transform(batch, processor):
    inputs = processor(images=batch["image"], text=batch["caption"], padding="max_length")
    inputs.update({"labels": batch["caption"]})
    return inputs


if __name__ == "__main__":
    print("Running caption generation training script...")
    main()
