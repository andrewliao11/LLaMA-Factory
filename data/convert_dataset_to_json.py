import os
import argparse
import json
from PIL import Image
from tqdm import tqdm

import ipdb

IMAGE_TAG = "<image>"


image_root = "/h/andrewliao/large-scratch/datasets/llama_factory_dataset"
def extract_cifar10(hf_dataset, split):
    image_dir = os.path.join(image_root, "cifar10", split)
    os.makedirs(image_dir, exist_ok=True)
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    class_label = ", ".join(classes)
    
    sft_data = []
    for i, sample in tqdm(enumerate(hf_dataset)):
        
        image = sample['img']
        image_path = os.path.join(image_dir, f'cifar10_{str(i).zfill(10)}.png')
        image.save(image_path)
        
        label = sample['label']
        label_name = classes[label]
        
        # Formulate prompt, messages, and sft_data
        # Template: https://arxiv.org/abs/2309.10313
        prompt = f"{IMAGE_TAG}What is the number/object in the image?  Please only answer a single number/object in [{class_label}]."
        messages = [
            {
                "role": "user", 
                "content": prompt
            }, 
            {
                "role": "assistant", 
                "content": f"{label_name}"
            }
        ]
        sft_data.append({
            "messages": messages, 
            "images": [image_path]
        })
        
    return sft_data


def main():
    # Define relate arguments for the script 
    parser = argparse.ArgumentParser(description='Convert dataset to JSON')
    parser.add_argument('--huggingface_dataset_id', type=str, default='uoft-cs/cifar10', help='Dataset name')
    parser.add_argument('--output_name', type=str, required=True)
    args = parser.parse_args()
    
    from datasets import load_dataset
    hf_dataset = load_dataset(args.huggingface_dataset_id)
    
    if "cifar" in args.huggingface_dataset_id:
        for split in hf_dataset.keys():
            sft_data = extract_cifar10(hf_dataset[split], split=split) #[:100]
            json.dump(sft_data, open(f"{args.output_name}-{split}.json", 'w'), indent=4)


if __name__ == "__main__":
    main()