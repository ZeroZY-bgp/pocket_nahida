import copy
import os
import random

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset as TorchDataset
from peft import LoraConfig, TaskType, get_peft_model
from tqdm import tqdm
from transformers import TrainingArguments, Trainer, AutoTokenizer, TrainerCallback, AutoModelForCausalLM, \
    BitsAndBytesConfig

from utils import load_json, calc_total_params

# ====== 参数设置区域 ======

dataset_path = "datas/pretrain.json"
ori_model_path = "Qwen/Qwen2-1.5B-Instruct"
model_cache_dir = ""
log_dir = "result/qwen2/1.5B/pretrain"

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

torch_dtype = torch.bfloat16
use_peft = True
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
    logging_steps=1,
    learning_rate=1e-4,
    bf16=True,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=6,
    warmup_ratio=0.1,
    evaluation_strategy="no",
    save_strategy="no",
    logging_dir=log_dir,
    save_total_limit=1,
    gradient_checkpointing=False,
)

# ==================

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# 加载原始模型
model = AutoModelForCausalLM.from_pretrained(ori_model_path,
                                             cache_dir=model_cache_dir,
                                             torch_dtype=torch_dtype,
                                             trust_remote_code=True)
if use_peft:
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

model.enable_input_require_grads()
print(model)
calc_total_params(model)

# 分词器
tokenizer_path = model.config.name_or_path
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False, trust_remote_code=True)

test_device = "cuda"


def inference(prompt):
    inputs = tokenizer.encode_plus(prompt, add_special_tokens=True, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(input_ids=inputs["input_ids"].to(test_device),
                                 attention_mask=inputs["attention_mask"].to(test_device),
                                 max_length=1024)
    return tokenizer.decode(outputs[0], skip_special_tokens=False)


# 数据
def collate_fn(batch):
    input_ids = []
    attention_mask = []
    for b in batch:
        input_ids.append(b['input_ids'])
        attention_mask.append(b['attention_mask'])

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.eos_token_id)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    if "labels" in batch[0].keys():
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids
        }
    else:
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }


class PretrainDataset(TorchDataset):

    def __init__(self, data, do_train=False):
        self.datas = data
        self.do_train = do_train

    def __len__(self):
        return len(self.datas)

    def copy_from_dataset(self, dataset):
        self.datas = copy.deepcopy(dataset.datas)

    def subset(self, indices):
        ori_datas = copy.deepcopy(self.datas)
        self.datas = [ori_datas[i] for i in indices]

    def get_raw_item(self, item):
        return self.datas[item]

    def __getitem__(self, item):
        inputs = tokenizer.encode_plus(self.datas[item], add_special_tokens=True, return_tensors='pt')
        if self.do_train:
            return {
                "input_ids": inputs['input_ids'][0],
                "attention_mask": inputs['attention_mask'][0],
                "labels": inputs['input_ids'][0]
            }
        else:
            return {
                "input_ids": inputs['input_ids'][0],
                "attention_mask": inputs['attention_mask'][0]
            }


all_data = load_json(dataset_path)
random.shuffle(all_data)
train_dataset = PretrainDataset(all_data, do_train=True)
eval_dataset = PretrainDataset(all_data, do_train=True)


class SelectiveTrainingCallback(TrainerCallback):
    def __init__(self, model, train_dataset):
        self.ori_train_dataset = copy.deepcopy(train_dataset)
        self.train_dataset = train_dataset
        self.model = model

    def on_epoch_end(self, args, state, control, **kwargs):
        # 单条数据测试
        random_idx = random.randint(0, len(eval_dataset) - 1)
        prompt = eval_dataset.get_raw_item(random_idx)
        ran_idx = random.randint(1, len(prompt) - 1)
        print(f"Input :{prompt[:ran_idx]}")
        print(f"Model generate: {inference(prompt[:ran_idx])}")
        print(f"Original sentence: {prompt}")


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=collate_fn,
    callbacks=[SelectiveTrainingCallback(model, train_dataset)]
)

trainer.train()

model.save_pretrained(log_dir)
tokenizer.save_pretrained(log_dir)
