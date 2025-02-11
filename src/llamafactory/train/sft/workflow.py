# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/examples/pytorch/summarization/run_summarization.py
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

import os
import json
import uuid
import wandb
from collections import Counter
from typing import TYPE_CHECKING, List, Optional

from ...data import SFTDataCollatorWith4DAttentionMask, get_dataset, get_template_and_fix_tokenizer
from ...data.parser import get_dataset_list
from ...extras.constants import IGNORE_INDEX
from ...extras.logging import get_logger
from ...extras.misc import calculate_tps, get_logits_processor
from ...extras.ploting import plot_loss
from ...model import load_model, load_tokenizer
from ..trainer_utils import create_modelcard_and_push
from .metric import ComputeAccuracy, ComputeSimilarity, eval_logit_processor, topk_logit_processor, ComputeSuccess, DummyMetrics, ComputeMetricsVQA
from .trainer import CustomSeq2SeqTrainer


if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments

import ipdb

logger = get_logger(__name__)


# NOTE: Seems that this does not always work as expected
def is_main_process():
    return getattr(os.environ, "LOCAL_RANK", "0") == "0"
    

from transformers import TrainerCallback
class EvaluateCallback(TrainerCallback):
    def __init__(self, output_dir, finetuning_args):
        self.output_dir = output_dir
        self.finetuning_args = finetuning_args
        
    def on_save(self, args, state, control, **kwargs):
        if state.is_world_process_zero and self.finetuning_args.submit_eval_during_training:
            work_dir = self.output_dir
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            checkpoint_dir = os.path.join(work_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")
            command = f"python main.py evaluate_experiment {work_dir} {checkpoint_dir} --sampled_eval=True"
            print(f"Use checkpoint: {checkpoint_dir}\nExecute: {command}")
            #os.system(command)
            import subprocess
            parent_env = json.load(open(os.path.join(self.output_dir, "parent_env.json")))
            subprocess.run(command.split(" "), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=parent_env)
            
            if self.finetuning_args.remove_optimizer_states:
                # remove all the optimizer states (except for the last one) to save disk space
                from pathlib import Path
                paths = list(Path(self.output_dir).glob("checkpoint-*/global_step*"))
                if len(paths) > 1:
                    last_path = sorted(paths, key=lambda x: int(x.name.replace("global_step", "")))[-1]
                    for p in paths:
                        if p != last_path:
                            print(f"Remove optimizer states: {p}")
                            command = f"rm -rf {p}"
                            subprocess.run(command.split(" "), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=parent_env)


def run_sft(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[List["TrainerCallback"]] = None,
):

    if is_main_process() and "wandb" in training_args.report_to:
        from dataclasses import dataclass, asdict
        p = os.path.join(training_args.output_dir, "wandb_id")
        if os.path.exists(p):
            unique_id = open(p).read().strip()
        else:
            unique_id = uuid.uuid4().hex[:8]
            open(p, "w").write(unique_id)
        
        config = {}
        if "SLURM_JOB_ID" in os.environ:
            config["SLURM_JOB_ID"] = os.environ["SLURM_JOB_ID"]
            
        config.update(asdict(model_args))
        config.update(asdict(data_args))
        config.update(training_args.to_dict())
        config.update(asdict(finetuning_args))
        config.update(asdict(generating_args))
        wandb.init(resume="allow", id=unique_id, project=os.getenv("WANDB_PROJECT", "huggingface"), config=config, name=training_args.run_name)
        
    training_args.learning_rate = float(training_args.learning_rate)
    
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    dataset_module = get_dataset(template, model_args, data_args, training_args, stage="sft", **tokenizer_module)
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)

    if getattr(model, "is_quantized", False) and not training_args.do_train:
        setattr(model, "_hf_peft_config_loaded", True)  # hack here: make model compatible with prediction

    data_collator = SFTDataCollatorWith4DAttentionMask(
        template=template,
        model=model if not training_args.predict_with_generate else None,
        pad_to_multiple_of=8 if training_args.do_train else None,  # for shift short attention
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
        block_diag_attn=model_args.block_diag_attn,
        attn_implementation=getattr(model.config, "_attn_implementation", None),
        compute_dtype=model_args.compute_dtype,
        **tokenizer_module,
    )

    # Override the decoding parameters of Seq2SeqTrainer
    training_args.generation_max_length = training_args.generation_max_length or data_args.cutoff_len
    training_args.generation_num_beams = data_args.eval_num_beams or training_args.generation_num_beams
    training_args.remove_unused_columns = False  # important for multimodal dataset

    # Metric utils
    metric_module = {}
    if training_args.predict_with_generate:
        data_sources = []
        for eval_dataset_name in data_args.eval_dataset:
            filename = str(get_dataset_list([eval_dataset_name, ], data_args.dataset_dir)[0])
            size_of_eval_dataset = len(json.load(open(filename)))
            
            if data_args.max_samples is not None:
                data_sources += [eval_dataset_name, ] * min(data_args.max_samples, size_of_eval_dataset)
            else:
                data_sources += [eval_dataset_name, ] * size_of_eval_dataset
            
        metric_module["compute_metrics"] = ComputeMetricsVQA(tokenizer=tokenizer, data_sources=data_sources)
    elif finetuning_args.compute_accuracy:
        metric_module["compute_metrics"] = ComputeAccuracy()
        metric_module["preprocess_logits_for_metrics"] = eval_logit_processor
    elif finetuning_args.log_top_k_preds:
        metric_module["compute_metrics"] = DummyMetrics(tokenizer=tokenizer)
        metric_module["preprocess_logits_for_metrics"] = topk_logit_processor
    

    # NOTE: Add a callback to trigger evaluation job
    callbacks.append(EvaluateCallback(training_args.output_dir, finetuning_args))
    # Initialize our Trainer
    trainer = CustomSeq2SeqTrainer(
        model=model,
        args=training_args,
        finetuning_args=finetuning_args,
        data_collator=data_collator,
        callbacks=callbacks,
        **dataset_module,
        **tokenizer_module,
        **metric_module,
    )

    # Keyword arguments for `model.generate`
    gen_kwargs = generating_args.to_dict(obey_generation_config=True)
    gen_kwargs["eos_token_id"] = [tokenizer.eos_token_id] + tokenizer.additional_special_tokens_ids
    gen_kwargs["pad_token_id"] = tokenizer.pad_token_id
    gen_kwargs["logits_processor"] = get_logits_processor()

    # Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        if finetuning_args.include_effective_tokens_per_second:
            train_result.metrics["effective_tokens_per_sec"] = calculate_tps(
                dataset_module["train_dataset"], train_result.metrics, stage="sft"
            )

        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            plot_loss(training_args.output_dir, keys=["loss", "eval_loss", "eval_accuracy"])

    if training_args.predict_with_generate:
        tokenizer.padding_side = "left"  # use left-padding in generation

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval", **gen_kwargs)
        if finetuning_args.log_top_k_preds:
            eval_topk_tokens = metrics.pop("eval_topk_tokens")
            eval_topk_probs = metrics.pop("eval_topk_probs")
            trainer.save_logprobs(dataset_module["eval_dataset"], {"topk_tokens": eval_topk_tokens, "topk_probs": eval_topk_probs})
            
        if training_args.predict_with_generate:  # eval_loss will be wrong if predict_with_generate is enabled
            metrics.pop("eval_loss", None)
            
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Predict
    if training_args.do_predict:
        logger.warning_rank0_once("Batch generation can be very slow. Consider using `scripts/vllm_infer.py` instead.")
        predict_results = trainer.predict(dataset_module["eval_dataset"], metric_key_prefix="predict", **gen_kwargs)
        trainer.log_metrics("predict", predict_results.metrics)
        trainer.save_metrics("predict", predict_results.metrics)
        trainer.save_predictions(dataset_module["eval_dataset"], predict_results, generating_args.skip_special_tokens)

    # Create model card
    create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args)
