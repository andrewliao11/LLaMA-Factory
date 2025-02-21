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
from pathlib import Path

import fire
import numpy as np
from tqdm import tqdm
from transformers import Seq2SeqTrainingArguments
from copy import deepcopy

from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.extras.misc import check_version, get_device_count
from llamafactory.extras.packages import is_vllm_available
from llamafactory.hparams import get_infer_args
from llamafactory.model import load_tokenizer

if is_vllm_available():
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
import ipdb


class EditPrompts():
    
    think_start_token = "<think>"
    think_end_token = "</think>"
    wait_tokens = ["Wait...", "Hmm..."]

    def __init__(self, force_thinking, force_wait, n_reflections, tokenizer, sampling_kwargs):
        self.force_thinking = force_thinking
        self.force_wait = force_wait
        self.n_reflections = n_reflections
        self.tokenizer = tokenizer
        self.sampling_kwargs = sampling_kwargs
        self.thinking_tokens = tokenizer.encode(self.think_start_token)
        self.npr = np.random.RandomState(123)
        
    def before_inference(self, inputs):
        new_inputs = deepcopy(inputs)
        if self.force_thinking:
            for i, example in enumerate(new_inputs):
                new_inputs[i]["prompt_token_ids"] = example["prompt_token_ids"] + self.thinking_tokens
        
        return new_inputs
    
    def after_inference(self, inputs, preds):
        if self.force_wait:
            wait_sampling_kwargs = deepcopy(self.sampling_kwargs)
            wait_sampling_kwargs["min_tokens"] = 1
            wait_sampling_kwargs["n"] = self.n_reflections
            wait_sampling_params = SamplingParams(**wait_sampling_kwargs)
            
            # construct inputs
            new_inputs = []
            for inp, pred in zip(inputs, preds):
                prompt = self.tokenizer.decode(inp["prompt_token_ids"])
                for p in pred:
                    suppress_end_of_thinking = p.split(self.think_end_token)[0].strip()
                    wait_text = self.npr.choice(self.wait_tokens)
                    new_prompt = prompt + suppress_end_of_thinking + f" {wait_text} "
                    new_propmt_token_ids = self.tokenizer.encode(new_prompt)
                    new_inp = deepcopy(inp)
                    new_inp["prompt_token_ids"] = new_propmt_token_ids
                    new_inputs.append(new_inp)
                    
            return {
                "prompts": new_inputs, 
                "sampling_params": wait_sampling_params
            }
        
        else:
            return None

    def dump_results(self, save_name, data_to_dump, labels):
        with open(save_name, "a", encoding="utf-8") as f:
            n_rounds = len(data_to_dump)
            n_data = len(data_to_dump[0]["inputs"])
            
            for i in range(n_data):
                d = {"label": labels[i]}
                for r in range(n_rounds):
                    d[f"inputs_{r}"] = self.tokenizer.decode(data_to_dump[r]["inputs"][i]["prompt_token_ids"])
                    d[f"preds_{r}"] = data_to_dump[r]["preds"][i]
                    
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        
    
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
    max_num_seqs: int = 256,  # default: 256
    infer_dtype: str = "auto",
    pipeline_parallel_size: int = 1,
    image_resolution: int = 512 * 512,
    chunk_size: int = 1000,
    force_thinking: bool = False,
    force_wait: bool = False,
    n_reflections: int = 1,
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
            trust_remote_code=True,
        )
    )

    training_args = Seq2SeqTrainingArguments(output_dir="dummy_dir")
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template_obj = get_template_and_fix_tokenizer(tokenizer, data_args)
    template_obj.mm_plugin.expand_mm_tokens = False  # for vllm generate
    dataset_module = get_dataset(template_obj, model_args, data_args, training_args, "ppo", **tokenizer_module)

    sampling_kwargs = {
        "n": n_samples_per_input,
        "repetition_penalty": generating_args.repetition_penalty or 1.0,  # repetition_penalty must > 0
        "temperature": generating_args.temperature,
        "top_p": generating_args.top_p or 1.0,  # top_p must > 0
        "top_k": generating_args.top_k,
        "stop_token_ids": template_obj.get_stop_token_ids(tokenizer),
        "max_tokens": generating_args.max_new_tokens,
        "skip_special_tokens": False,
        "seed": 123
    }
    sampling_params = SamplingParams(**sampling_kwargs)

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
        # "max_num_batched_tokens": max_num_batched_tokens,
        "enable_lora": model_args.adapter_name_or_path is not None,
    }
    if template_obj.mm_plugin.__class__.__name__ != "BasePlugin":
        engine_args["limit_mm_per_prompt"] = {"image": 1, "video": 0}

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

    # NOTE: We use a smaller chunk size to avoid opening too many files at the sașe time.
    editor = EditPrompts(force_thinking, force_wait, n_reflections, tokenizer, sampling_kwargs)
        
    n_total_samples = 0
    for i, (inputs, prompts, labels) in enumerate(yield_chunks(dataset_module, template_obj, tokenizer, image_resolution, chunk_size)):
        n_total_samples += len(inputs)
        if i < n_chunk_to_skip:
            continue
        
        data_to_dump = []
        
        ### Edit inputs prompts if necessary
        new_iniputs = editor.before_inference(inputs)
        results = llm.generate(new_iniputs, sampling_params, lora_request=lora_request)
        preds = [[o.text for o in result.outputs] for result in results]
        data_to_dump.append({"inputs": new_iniputs, "preds": preds})
        
        # suppress the thinking token
        new_input_dict = editor.after_inference(new_iniputs, preds)
        if new_input_dict is not None:
            results = llm.generate(**new_input_dict, lora_request=lora_request)
            preds = [[o.text for o in result.outputs] for result in results]
            data_to_dump.append({"inputs": new_input_dict["prompts"], "preds": preds})
            
        editor.dump_results(save_name, data_to_dump, labels)
        

    print("*" * 70)
    print(f"{n_total_samples} generated results have been saved at {save_name}.")
    print("*" * 70)

# python LLaMA-Factory/scripts/vllm_infer.py --model_name_or_path /h/andrewliao/large-scratch/pretrained_weights/Qwen2.5-VL-7B-Instruct/ --dataset image/gqa_variants/filtered_gqa_v0_Qwen2.5-32B-Instruct_answer_5000 --dataset_dir LLaMA-Factory/data --template qwen2_vl --infer_dtype half --max_new_tokens --save_name outputs/force_thinking/filtered_gqa_v0_Qwen2.5-32B-Instruct_answer_5000/generated_predictions.jsonl --max_num_seqs 1 --n_samples_per_input 1 --temperature 0. --vllm_config="{'enforce_eager': true, 'gpu_memory_utilization': 0.95}"

if __name__ == "__main__":
    fire.Fire(vllm_infer)
