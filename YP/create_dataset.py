# use QWen model to generate flickr30k style image-caption pairs to augment the flickr30K datasets 

from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import os
import json
import re
import torch

# set up the path
image_dir = './Images4Qwen'
output_json = 'Image_Caption_dataset'

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct",
    device_map="auto",
)
# default processer
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

### generate 5 one-sentence long captions for each image
image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg','png','jpeg'))] # note: the method .endwith take a single string or a tuple of strings as input 
prompts = ['Add a one-sentence caption for this image.',
    'Describe this image in one senence with good enough details.',
    'What is happening in this image?',
    'Describe this image in one sentence to me as if I were a 5-yers-one',
    'Caption this image in one sentence to include the elements of who, where, when, and doing what'] # 5 difference prompts to encourage a varation when generating captions
# The default range for the number of visual tokens per image in the model is 4-16384.
# You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

n_captions = 5
dataset = []
for idx,image in enumerate(image_files):
    image_path = os.path.join(image_dir,image)
    captions = []
    
    for i in range(n_captions):    
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {"type": "text", "text": prompts[i]},
                ],
            }
        ]
        # Preparation for inference
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
        # Inference: Generation of the output
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=128)
            
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        # ensure a one-sentence output
        caption_onesentence = re.split(r'(?<=[.!?])\s',output_text[0].strip())[0]
        captions.append(caption_onesentence) 



    # add to dataset
    dataset.append({
        'img_id': str(idx),
        'filename': image,
        'caption': captions
    })

# save out
with open(output_json, 'w') as f:
    json.dump(dataset, f, indent=2)

print(f'generated dataset saved to {output_json}')