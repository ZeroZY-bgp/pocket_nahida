import os

from utils import load_json, save_json


def history_collect(root_path, save_path):
    history_path = root_path
    res = []
    for file_name in os.listdir(history_path):
        if file_name.endswith('.json'):
            messages = load_json(history_path + '/' + file_name)
            res.append(messages)
    print(f"Total messages: {len(res)}")
    save_json(save_path, res)
    print(f"Messages data saved to {save_path}")


if __name__ == '__main__':
    root_path = "history"
    save_path = "train/datas/sft.json"
    history_collect(root_path, save_path)
