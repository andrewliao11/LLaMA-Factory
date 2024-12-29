# Copyright 2024 HuggingFace Inc., THUDM, and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library and the THUDM's ChatGLM implementation.
# https://github.com/huggingface/transformers/blob/v4.40.0/examples/pytorch/summarization/run_summarization.py
# https://github.com/THUDM/ChatGLM-6B/blob/main/ptuning/main.py
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

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Optional, List, Any
from copy import deepcopy

import numpy as np
import torch
from transformers.utils import is_jieba_available, is_nltk_available

from ...extras.constants import IGNORE_INDEX
from ...extras.misc import numpify
from ...extras.packages import is_rouge_available


if TYPE_CHECKING:
    from transformers import EvalPrediction, PreTrainedTokenizer


if is_jieba_available():
    import jieba  # type: ignore


if is_nltk_available():
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu


if is_rouge_available():
    from rouge_chinese import Rouge


def topk_logit_processor(logits: "torch.Tensor", labels: "torch.Tensor") -> "torch.Tensor":
    if isinstance(logits, (list, tuple)):
        if logits[0].dim() == 3:  # (batch_size, seq_len, vocab_size)
            logits = logits[0]
        else:  # moe models have aux loss
            logits = logits[1]

    if logits.dim() != 3:
        raise ValueError("Cannot process the logits.")

    probs = torch.nn.functional.softmax(logits, dim=-1)
    shifted_probs = probs[:, :-1, :]
    shifted_labels = labels[:, 1:]
    mask = shifted_labels != IGNORE_INDEX
    
    label_tokens = []
    label_probs = []
    for m, prob, label in zip(mask, shifted_probs, shifted_labels):
        topk_probs, topk_tokens = prob[m].topk(5, dim=-1)
        
        selected_tokens = torch.cat([label[m].unsqueeze(1), topk_tokens], dim=1)
        seq_len, vocab_size = prob[m].shape
        ind = (torch.arange(seq_len).to(label.device) * vocab_size) + label[m]
        selected_prob = prob[m].reshape(-1)[ind]
        selected_prob = torch.cat([selected_prob.unsqueeze(1), topk_probs], dim=1)
        label_tokens.append(selected_tokens)
        label_probs.append(selected_prob)
    
        
    return torch.stack(label_probs, dim=0), torch.stack(label_tokens, dim=0)


def eval_logit_processor(logits: "torch.Tensor", labels: "torch.Tensor") -> "torch.Tensor":
    r"""
    Computes the token with the largest likelihood to reduce memory footprint.
    """
    if isinstance(logits, (list, tuple)):
        if logits[0].dim() == 3:  # (batch_size, seq_len, vocab_size)
            logits = logits[0]
        else:  # moe models have aux loss
            logits = logits[1]

    if logits.dim() != 3:
        raise ValueError("Cannot process the logits.")

    return torch.argmax(logits, dim=-1)



@dataclass
class DummyMetrics:
    tokenizer: "PreTrainedTokenizer"

    def _dump(self) -> Optional[Dict[str, float]]:
        result = None
        if hasattr(self, "data_dict"):
            result = {k: v.tolist() for k, v in self.data_dict.items()}

        self.data_dict = {}
        return result

    def __post_init__(self):
        self._dump()

    def __call__(self, eval_preds: "EvalPrediction", compute_result: bool = True) -> Optional[Dict[str, float]]:
        topk_probs, topk_tokens = numpify(eval_preds.predictions)
        padded_topk_tokens = []
        
        for probs, tokens in zip(topk_probs, topk_tokens):
            
            tokens = np.where(tokens != IGNORE_INDEX, tokens, self.tokenizer.pad_token_id)
            decoded_tokens = self.tokenizer.batch_decode(tokens.reshape(-1))
            decoded_tokens = np.array(decoded_tokens).reshape(-1, tokens.shape[1])
            padded_topk_tokens.append(decoded_tokens)
            
        padded_topk_tokens = np.array(padded_topk_tokens)
        self.data_dict.update({"topk_tokens": padded_topk_tokens, "topk_probs": topk_probs})
        if compute_result:
            return self._dump()
        
    
@dataclass
class ComputeAccuracy:
    r"""
    Computes accuracy and supports `batch_eval_metrics`.
    """

    def _dump(self) -> Optional[Dict[str, float]]:
        result = None
        if hasattr(self, "score_dict"):
            result = {k: float(np.mean(v)) for k, v in self.score_dict.items()}

        self.score_dict = {"accuracy": []}
        return result

    def __post_init__(self):
        self._dump()

    def __call__(self, eval_preds: "EvalPrediction", compute_result: bool = True) -> Optional[Dict[str, float]]:
        preds, labels = numpify(eval_preds.predictions), numpify(eval_preds.label_ids)
        for i in range(len(preds)):
            pred, label = preds[i, :-1], labels[i, 1:]
            label_mask = label != IGNORE_INDEX
            self.score_dict["accuracy"].append(np.mean(pred[label_mask] == label[label_mask]))

        if compute_result:
            return self._dump()


import gymnasium as gym
import sys
sys.path.append("/h/andrewliao/research/visual_reasoning_pomdp/data_gen")
import env
import ipdb

def extract_coordinates(input_text):
    import re
    # Find the content within the <answer> tag
    #answer_match = re.search(r'<answer>(.*?)</answer>', input_text, re.DOTALL)
    answer_match = re.findall(r'<answer>(.*?)</answer>', input_text, re.DOTALL)[-1]
    
    #if not answer_match:
    #    return []
    
    # Extract all coordinates from GoTo commands
    coordinates = re.findall(r'GoTo\((\d+), (\d+)\)', answer_match)
    
    # Convert coordinates to tuples of integers
    return [(int(x), int(y)) for x, y in coordinates]


@dataclass
class ComputeSuccess:
    tokenizer: "PreTrainedTokenizer"
    gym_env_name: str = "FrozenLakeMultiGoalGotoEnv-v0"
    gym_env_args: List[Dict[str, Any]] = None
    
    def _dump(self) -> Optional[Dict[str, float]]:
        result = None
        if hasattr(self, "score_dict"):
            result = {k: float(np.mean(v)) for k, v in self.score_dict.items()}
            result.update({"n_samples": len(self.score_dict["success_rate"])})

        self.score_dict = {"success_rate": [], "avg_path_cost": [], "valid_path": [], "optimal_rate": []}
        return result

    def __post_init__(self):
        self._dump()

    def __call__(self, eval_preds: "EvalPrediction", compute_result: bool = True) -> Optional[Dict[str, float]]:
        
        preds, labels = numpify(eval_preds.predictions), numpify(eval_preds.label_ids)

        preds = np.where(preds != IGNORE_INDEX, preds, self.tokenizer.pad_token_id)
        labels = np.where(labels != IGNORE_INDEX, labels, self.tokenizer.pad_token_id)
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=False)
        decoded_preds_concise = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=False)

        for i, (pred_concise, pred, label, env_args) in enumerate(zip(decoded_preds_concise, decoded_preds, decoded_labels, self.gym_env_args)):
            env = gym.make(
                self.gym_env_name, 
                desc=env_args["desc"], 
                n_goals_to_reach=env_args["n_goals_to_reach"], 
                render_mode="ansi"
            )
            env.reset()
            print(env.render())
            
            unwrapped_env = env
            while isinstance(unwrapped_env, gym.Wrapper):
                unwrapped_env = unwrapped_env.env
                
            _, optimal_cost, _, _ = unwrapped_env.find_minimum_cost_solution()
            if optimal_cost == np.inf:
                continue
            
            # Parse action
            try:
                coordinates = extract_coordinates(pred)
            except IndexError:
                coordinates = []
                
            print(pred_concise)
            if len(coordinates) == 0:
                self.score_dict["optimal_rate"].append(False)
                self.score_dict["success_rate"].append(False)
                self.score_dict["valid_path"].append(False)
            else:
                    
                action_plan = []
                for coord in coordinates:
                    action_plan.append({"goto": (int(coord[0]), int(coord[1]))})
                    
                total_reward = 0.
                path_cost = 0.
                valid_path = True
                for step in action_plan:
                    obs, reward, _, _, info = env.step(step)
                    
                    total_reward += reward
                    if info["action_failed"]:
                        valid_path = False
                        break
                    
                    path_cost += len(info["path"])
                                
            
                self.score_dict["optimal_rate"].append(total_reward > 0 and total_reward <= optimal_cost)
                self.score_dict["success_rate"].append(total_reward > 0)
                self.score_dict["valid_path"].append(valid_path)
                if valid_path:
                    self.score_dict["avg_path_cost"].append(path_cost)

        if compute_result:
            return self._dump()
        
        
@dataclass
class ComputeSimilarity:
    r"""
    Computes text similarity scores and supports `batch_eval_metrics`.

    Wraps the tokenizer into metric functions, used in CustomSeq2SeqTrainer.
    """

    tokenizer: "PreTrainedTokenizer"

    def _dump(self) -> Optional[Dict[str, float]]:
        result = None
        if hasattr(self, "score_dict"):
            result = {k: float(np.mean(v)) for k, v in self.score_dict.items()}

        self.score_dict = {"rouge-1": [], "rouge-2": [], "rouge-l": [], "bleu-4": []}
        return result

    def __post_init__(self):
        self._dump()

    def __call__(self, eval_preds: "EvalPrediction", compute_result: bool = True) -> Optional[Dict[str, float]]:
        preds, labels = numpify(eval_preds.predictions), numpify(eval_preds.label_ids)

        preds = np.where(preds != IGNORE_INDEX, preds, self.tokenizer.pad_token_id)
        labels = np.where(labels != IGNORE_INDEX, labels, self.tokenizer.pad_token_id)

        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        for pred, label in zip(decoded_preds, decoded_labels):
            hypothesis = list(jieba.cut(pred))
            reference = list(jieba.cut(label))

            if len(" ".join(hypothesis).split()) == 0 or len(" ".join(reference).split()) == 0:
                result = {"rouge-1": {"f": 0.0}, "rouge-2": {"f": 0.0}, "rouge-l": {"f": 0.0}}
            else:
                rouge = Rouge()
                scores = rouge.get_scores(" ".join(hypothesis), " ".join(reference))
                result = scores[0]

            for k, v in result.items():
                self.score_dict[k].append(round(v["f"] * 100, 4))

            bleu_score = sentence_bleu([list(label)], list(pred), smoothing_function=SmoothingFunction().method3)
            self.score_dict["bleu-4"].append(round(bleu_score * 100, 4))

        if compute_result:
            return self._dump()
