import copy

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import TrainingArguments, Trainer, AutoTokenizer, TrainerCallback, AutoModelForCausalLM
from trl import DPOTrainer

from utils import load_json


dataset_path = "datas/dpo.json"
ori_model_path = r""

log_dir = "result/qwen1.5/1.8B/dpo"
# device = "cuda"

use_peft = False
torch_dtype = torch.bfloat16

if use_peft:
    # peft设置
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        # target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1
    )

training_args = TrainingArguments(
    output_dir=log_dir,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    logging_steps=2,
    num_train_epochs=2,
    save_strategy="no",
    # save_steps=10,
    warmup_ratio=0.1,
    learning_rate=1e-4,
    gradient_checkpointing=False,
    logging_dir=log_dir,
    save_total_limit=1,
    remove_unused_columns=False
)

# 模型
model = AutoModelForCausalLM.from_pretrained(ori_model_path,
                                             torch_dtype=torch_dtype,
                                             trust_remote_code=True
                                             )
tokenizer_path = model.config.name_or_path
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path,
                                          use_fast=False,
                                          trust_remote_code=True)
if use_peft:
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

# model.enable_input_require_grads()
print(model)


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


json_datas = load_json(dataset_path)
train_dataset = Dataset.from_dict(json_datas)


# class SelectiveTrainingCallback(TrainerCallback):
#     def __init__(self, model, train_dataset, loss_threshold):
#         self.loss_threshold = loss_threshold
#         self.ori_train_dataset = copy.deepcopy(train_dataset)
#         self.train_dataset = train_dataset
#         self.model = model
#
#     def evaluate_train_loss(self):
#         self.model.eval()
#         total_loss = 0.0
#         losses = []
#         for d in tqdm(self.ori_train_dataset):
#             with torch.no_grad():
#                 outputs = self.model(input_ids=d['input_ids'].unsqueeze(0).to(device),
#                                      attention_mask=d['attention_mask'].unsqueeze(0).to(device),
#                                      labels=d["labels"].unsqueeze(0).to(device))
#                 total_loss += outputs.loss
#                 losses.append(outputs.loss.item())
#         test_avg_loss = total_loss / len(self.ori_train_dataset)
#         print(f"Test average loss: {test_avg_loss}")
#         return losses
#
#     def on_epoch_end(self, args, state, control, **kwargs):
#         losses = self.evaluate_train_loss()
#         indices = [i for i, loss in enumerate(losses) if loss > self.loss_threshold]
#         self.train_dataset.copy_from_dataset(self.ori_train_dataset)
#         self.train_dataset.subset(indices)


tokenizer.bos_token_id = start_token_id

trainer = DPOTrainer(
    model=model,
    tokenizer=tokenizer,
    max_length=512,
    max_prompt_length=256,
    args=training_args,
    train_dataset=train_dataset,
    # data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    # callbacks=[SelectiveTrainingCallback(model, train_dataset, loss_threshold)]  # 添加自定义回调
)

trainer.train()

model.save_pretrained(log_dir)
tokenizer.save_pretrained(log_dir)

