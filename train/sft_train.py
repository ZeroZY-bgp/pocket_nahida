import copy
import os

import torch
from torch.utils.data import Dataset as TorchDataset
from peft import LoraConfig, TaskType, get_peft_model
from tqdm import tqdm
from transformers import TrainingArguments, Trainer, AutoTokenizer, TrainerCallback, AutoModelForCausalLM, \
    DataCollatorForSeq2Seq, BitsAndBytesConfig

from utils import load_json

# ====== 参数设置区域 ======

dataset_path = "datas/sft.json"
ori_model_path = "result/qwen2/1.5B/pretrain"
log_dir = "result/qwen2/1.5B/sft"

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

torch_dtype = torch.bfloat16
use_peft = False
quantized = False

if use_peft:
    # peft设置
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        inference_mode=False,
        r=64,
        lora_alpha=32,
        lora_dropout=0.1
    )

if quantized:
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch_dtype
    )

training_args = TrainingArguments(
    output_dir=log_dir,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    logging_steps=1,
    bf16=True,
    num_train_epochs=6,
    save_strategy="no",
    # warmup_steps=0,
    warmup_ratio=0.1,
    learning_rate=1e-4,
    gradient_checkpointing=False,
    logging_dir=log_dir,
    save_total_limit=1
)

# ==================

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# 模型
model = AutoModelForCausalLM.from_pretrained(ori_model_path,
                                             torch_dtype=torch_dtype,
                                             trust_remote_code=True,
                                             quantization_config=nf4_config if quantized else None
                                             )
if use_peft:
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
# model.enable_input_require_grads()
print(model)

# 分词器
tokenizer_path = model.config.name_or_path
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False, trust_remote_code=True)

start_token_id = tokenizer.convert_tokens_to_ids('<|im_start|>')
end_token_id = tokenizer.convert_tokens_to_ids('<|im_end|>')
assistant_token_id = tokenizer.convert_tokens_to_ids('assistant')
newline_token_id = tokenizer.convert_tokens_to_ids('\n')


def multi_turn_process_func(messages):
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    model_inputs = tokenizer(text, return_tensors="pt")
    input_ids = model_inputs['input_ids'][0]
    attention_mask = model_inputs['attention_mask'][0]
    # 构造标签
    labels = torch.full((len(input_ids),), -100)
    is_assistant_section = False
    i = 0
    while i != len(input_ids):
        if is_assistant_section and input_ids[i].item() != assistant_token_id:
            labels[i] = copy.deepcopy(input_ids[i])
        if input_ids[i] == start_token_id:
            is_assistant_section = (input_ids[i + 1].item() == assistant_token_id)
            i += 3
            continue
        elif input_ids[i] == end_token_id:
            is_assistant_section = False
        i += 1

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


# 数据集
class MessagesDataset(TorchDataset):

    def __init__(self, datas):
        self.datas = []
        for data in datas:
            self.datas.append(multi_turn_process_func(data))

    def copy_from_dataset(self, dataset):
        self.datas = copy.deepcopy(dataset.datas)

    def subset(self, indices):
        ori_datas = copy.deepcopy(self.datas)
        self.datas = [ori_datas[i] for i in indices]

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, item):
        return self.datas[item]


# 加载多轮对话微调数据集
all_data = load_json(dataset_path)
train_dataset = MessagesDataset(all_data)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)
)

trainer.train()

model.save_pretrained(log_dir)
tokenizer.save_pretrained(log_dir)
