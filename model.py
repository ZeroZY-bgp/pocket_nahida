import copy
import warnings
from threading import Thread
from typing import Optional, Callable, List, Dict

import torch
import openai
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TextIteratorStreamer
from transformers.generation.logits_process import LogitsProcessor
from transformers.utils import logging, is_torch_npu_available
from transformers.generation.utils import LogitsProcessorList, StoppingCriteriaList, GenerationConfig, ModelOutput
from peft import PeftModel, PeftConfig

from huggingface_hub import InferenceClient

logger = logging.get_logger(__name__)


class QwenModel:

    def __init__(self,
                 cache_dir=None,
                 model_name_or_path=None,
                 quantized=True,
                 device='cuda',
                 token=None):
        if quantized:
            nf4_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16
            )
        try:
            # 尝试加载为 LoRA 模型
            config = PeftConfig.from_pretrained(model_name_or_path,
                                                device_map=device,
                                                torch_dtype=torch.bfloat16,
                                                trust_remote_code=True,
                                                token=token,
                                                cache_dir=cache_dir)
            base_model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path,
                                                              device_map=device,
                                                              torch_dtype=torch.bfloat16,
                                                              trust_remote_code=True,
                                                              token=token,
                                                              cache_dir=cache_dir,
                                                              quantization_config=nf4_config if quantized else None)
            self.model = PeftModel.from_pretrained(base_model, model_name_or_path)
            # 获取基础模型路径
            self.tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path,
                                                           use_fast=False,
                                                           trust_remote_code=True,
                                                           token=token,
                                                           cache_dir=cache_dir)
            print(f"Loaded LoRA model from {model_name_or_path} with base model {config.base_model_name_or_path}")
        except Exception as e:
            # 如果加载 LoRA 模型失败，则尝试加载为普通模型
            print(f"Error loading LoRA model: {e}")
            self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                              device_map=device,
                                                              torch_dtype=torch.bfloat16,
                                                              trust_remote_code=True,
                                                              token=token,
                                                              cache_dir=cache_dir,
                                                              quantization_config=nf4_config if quantized else None
                                                              )
            tokenizer_path = self.model.config.name_or_path
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path,
                                                           use_fast=False,
                                                           trust_remote_code=True,
                                                           token=token,
                                                           cache_dir=cache_dir)
            print(f"Loaded base model from {model_name_or_path}")
        self.device = device
        self.eos_token = '<|im_end|>'
        self.streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True)

    def stream_chat(self, messages, temperature=0.6, max_new_tokens=512):
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        model_inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        generation_kwargs = dict(model_inputs, streamer=self.streamer, temperature=temperature, max_new_tokens=max_new_tokens)
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)

        with torch.no_grad():
            thread.start()
            for chunk in self.streamer:
                if self.eos_token in chunk:
                    chunk = chunk.replace(self.eos_token, '')
                yield chunk

    def chat(self, messages, temperature=0.6, max_new_tokens=512):
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        model_inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                model_inputs["input_ids"],
                attention_mask=model_inputs["attention_mask"],
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                use_cache=True
            )

        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]


class GPTModel:
    def __init__(self,
                 api_key,
                 model_name='gpt-3.5-turbo',
                 temperature=0.1,
                 max_token=1000,
                 streaming=False):
        openai.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        self.max_token = max_token
        self.streaming = streaming
        self.cur_prompt_tokens = 0
        self.cur_completion_tokens = 0
        self.cur_total_tokens = 0

    def send(self, messages):
        res = openai.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_token
        )
        self.cur_prompt_tokens = res.usage.prompt_tokens
        self.cur_completion_tokens = res.usage.completion_tokens
        self.cur_total_tokens = res.usage.total_tokens
        return res.choices[0].message.content

    def get_token_info(self):
        return {'prompt_tokens': self.cur_prompt_tokens,
                'completion_tokens': self.cur_completion_tokens,
                'total_tokens': self.cur_total_tokens}

    def send_stream(self, messages):
        for chunk in openai.ChatCompletion.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_token,
                stream=True,
        ):
            res = chunk["choices"][0].get("delta", {}).get("content")
            if res is not None:
                yield res

    def chat(self, messages, temperature=0.6, max_new_tokens=512):
        self.temperature = temperature
        self.max_token = max_new_tokens
        if self.streaming:
            return self.send_stream(messages)
        else:
            return self.send(messages)
