# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json

import fire
import torch
from torchvision import transforms
import numpy as np
from transformers import Seq2SeqTrainingArguments

from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.extras.misc import check_version, get_device_count
from llamafactory.extras.packages import is_vllm_available
from llamafactory.hparams import get_infer_args
from llamafactory.model import load_tokenizer
import multiprocessing as mp
from pathlib import Path
from tqdm import tqdm
from functools import partial
import ipdb


if is_vllm_available():
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest


def yield_chunks(dataset_module, template_obj, tokenizer, image_resolution, chunk_size):
    inputs, prompts, labels = [], [], []
    for sample in tqdm(dataset_module["train_dataset"], desc="Preparing data"):
        if sample["images"]:
            multi_modal_data = {
                "image": template_obj.mm_plugin._regularize_images(sample["images"], image_resolution=image_resolution)
            }
        else:
            multi_modal_data = None

        inputs.append({"prompt_token_ids": sample["input_ids"], "multi_modal_data": multi_modal_data})
        prompts.append(tokenizer.decode(sample["input_ids"], skip_special_tokens=False))
        labels.append(
            tokenizer.decode(list(filter(lambda x: x != IGNORE_INDEX, sample["labels"])), skip_special_tokens=False)
        )

        if chunk_size > 0 and len(inputs) >= chunk_size:
            yield inputs, prompts, labels
            inputs, prompts, labels = [], [], []

    if inputs:
        yield inputs, prompts, labels
        
      
# Qwen 2.5 default setup
# temperature 0.7, top_p 0.8, repetition_penalty 1.05: https://github.com/QwenLM/Qwen2.5?tab=readme-ov-file#vllm 
def vllm_infer(
    model_name_or_path: str,
    adapter_name_or_path: str = None,
    dataset: str = "alpaca_en_demo",
    dataset_dir: str = "data",
    template: str = "default",
    cutoff_len: int = 2048,
    max_samples: int = None,
    n_samples_per_input: int = 1,
    vllm_config: str = "{}",
    save_name: str = "generated_predictions.jsonl",
    temperature: float = 0.95,
    top_p: float = 0.7,
    top_k: int = 50,
    max_new_tokens: int = 1024,
    repetition_penalty: float = 1.0,
    max_num_seqs: int = 256,        # default: 256
    infer_dtype: str = "auto",
    pipeline_parallel_size: int = 1,
    image_resolution: int = 512 * 512, 
    chunk_size: int = 1000,
):
    r"""
    Performs batch generation using vLLM engine, which supports tensor parallelism.
    Usage: python vllm_infer.py --model_name_or_path meta-llama/Llama-2-7b-hf --template llama --dataset alpaca_en_demo
    """

    check_version("vllm>=0.4.3,<=0.6.5")
    if pipeline_parallel_size > get_device_count():
        raise ValueError("Pipeline parallel size should be smaller than the number of gpus.")

    model_args, data_args, _, generating_args = get_infer_args(
        dict(
            image_resolution=image_resolution, 
            model_name_or_path=model_name_or_path,
            adapter_name_or_path=adapter_name_or_path,
            dataset=dataset,
            dataset_dir=dataset_dir,
            template=template,
            cutoff_len=cutoff_len,
            max_samples=max_samples,
            preprocessing_num_workers=16,
            vllm_config=vllm_config,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            infer_dtype=infer_dtype, 
            trust_remote_code=True
        )
    )

    training_args = Seq2SeqTrainingArguments(output_dir="dummy_dir")
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template_obj = get_template_and_fix_tokenizer(tokenizer, data_args)
    template_obj.mm_plugin.expand_mm_tokens = False  # for vllm generate
    dataset_module = get_dataset(template_obj, model_args, data_args, training_args, "ppo", **tokenizer_module)

    
    sampling_params = SamplingParams(
        n=n_samples_per_input,
        repetition_penalty=generating_args.repetition_penalty or 1.0,  # repetition_penalty must > 0
        temperature=generating_args.temperature,
        top_p=generating_args.top_p or 1.0,  # top_p must > 0
        top_k=generating_args.top_k,
        stop_token_ids=template_obj.get_stop_token_ids(tokenizer),
        max_tokens=generating_args.max_new_tokens,
        skip_special_tokens=False,
        #logprobs=1, 
        seed=123
    )
    
    if model_args.adapter_name_or_path is not None:
        lora_request = LoRARequest("default", 1, model_args.adapter_name_or_path[0])
    else:
        lora_request = None

    engine_args = {
        "model": model_args.model_name_or_path,
        "trust_remote_code": True,
        "dtype": model_args.infer_dtype,
        "tensor_parallel_size": (get_device_count() // pipeline_parallel_size) or 1,
        "pipeline_parallel_size": pipeline_parallel_size,
        "disable_log_stats": True,
        "max_num_seqs": max_num_seqs,
        #"max_num_batched_tokens": max_num_batched_tokens, 
        "enable_lora": model_args.adapter_name_or_path is not None,
    }
    if template_obj.mm_plugin.__class__.__name__ != "BasePlugin":
        engine_args["limit_mm_per_prompt"] = {"image": 4, "video": 2}

    if isinstance(model_args.vllm_config, dict):
        engine_args.update(model_args.vllm_config)

    print(f"Engine args: {engine_args}")
    llm = LLM(**engine_args)
    
    save_name = Path(save_name)
    
    if save_name.exists():
        n_data_dumped = len(save_name.open().readlines())
    else:
        n_data_dumped = 0    
        
    #save_name.unlink(missing_ok=True)
    n_chunk_to_skip = n_data_dumped // chunk_size

    # NOTE: We use a smaller chunk size to avoid opening too many files at the same time.
    for i, (inputs, prompts, labels) in enumerate(yield_chunks(dataset_module, template_obj, tokenizer, image_resolution, chunk_size)):
        if i < n_chunk_to_skip:
            continue
        
        results = llm.generate(inputs, sampling_params, lora_request=lora_request)
        preds = [[o.text for o in result.outputs] for result in results]
        with open(save_name, "a", encoding="utf-8") as f:
            for text, pred, label in zip(prompts, preds, labels):
                f.write(json.dumps({"prompt": text, "predict": pred, "label": label}, ensure_ascii=False) + "\n")

    print("*" * 70)
    print(f"{len(prompts)} generated results have been saved at {save_name}.")
    print("*" * 70)
    

if __name__ == "__main__":
    fire.Fire(vllm_infer)
