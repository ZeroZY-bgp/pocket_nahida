import copy
import random

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset as torchDataset
from tqdm import tqdm
from transformers import TrainingArguments, Trainer, AutoTokenizer, TrainerCallback, AutoModelForCausalLM

from utils import load_json, calc_total_params


dataset_path = "datas/pretrain.json"
ori_model_path = ""
log_dir = "result/qwen1.5/1.8B/pretrain"
device = "cuda"

torch_dtype = torch.bfloat16
use_loss_threshold_filter = True
loss_threshold = 1.0
end_train_dataset_threshold = 0

training_args = TrainingArguments(
    output_dir=log_dir,
    learning_rate=1e-5,
    # auto_find_batch_size=True,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    num_train_epochs=10,
    weight_decay=0.01,
    warmup_steps=10,
    evaluation_strategy="no",
    eval_steps=200,
    save_strategy="epoch",
    logging_dir=log_dir,
    save_total_limit=2,
    gradient_checkpointing=False,
)

# 加载原始模型
model = AutoModelForCausalLM.from_pretrained(ori_model_path,
                                             torch_dtype=torch.bfloat16,
                                             trust_remote_code=True)
# model.enable_input_require_grads()
print(model)
calc_total_params(model)

# 分词器
tokenizer_path = model.config.name_or_path
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False, trust_remote_code=True)


def inference(prompt):
    inputs = tokenizer.encode_plus(prompt, add_special_tokens=True, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(input_ids=inputs["input_ids"].to(device),
                                 attention_mask=inputs["attention_mask"].to(device),
                                 max_length=512)
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


class PretrainDataset(torchDataset):

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
    def __init__(self, model, train_dataset, loss_threshold):
        self.loss_threshold = loss_threshold
        self.ori_train_dataset = copy.deepcopy(train_dataset)
        self.train_dataset = train_dataset
        self.model = model

    def evaluate_train_loss(self):
        self.model.eval()
        total_loss = 0.0
        losses = []
        for d in tqdm(self.ori_train_dataset):
            with torch.no_grad():
                outputs = self.model(input_ids=d['input_ids'].unsqueeze(0).to(device),
                                     attention_mask=d['attention_mask'].unsqueeze(0).to(device),
                                     labels=d["labels"].unsqueeze(0).to(device))
                total_loss += outputs.loss
                losses.append(outputs.loss.item())
        test_avg_loss = total_loss / len(self.ori_train_dataset)
        print(f"Test average loss: {test_avg_loss}")
        return losses

    def on_epoch_end(self, args, state, control, **kwargs):
        if use_loss_threshold_filter:
            losses = self.evaluate_train_loss()
            indices = [i for i, loss in enumerate(losses) if loss > self.loss_threshold]
            self.train_dataset.copy_from_dataset(self.ori_train_dataset)
            self.train_dataset.subset(indices)
            print(f"Dataset length: {len(self.train_dataset)}")
            if len(self.train_dataset) < end_train_dataset_threshold:
                model.save_pretrained(log_dir)
                self.train_dataset.datas.clear()
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
    callbacks=[SelectiveTrainingCallback(model, train_dataset, loss_threshold)]  # 添加自定义回调
)

trainer.train()

model.save_pretrained(log_dir)
tokenizer.save_pretrained(log_dir)
