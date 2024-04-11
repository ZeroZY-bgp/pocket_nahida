import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


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
        self.device = device

    def chat(self, messages, temperature=1.0, max_new_tokens=512):
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
