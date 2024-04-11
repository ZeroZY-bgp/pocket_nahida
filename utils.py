import json


def load_json(file_name):
    with open(file_name, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


def save_json(file_name, data):
    with open(file_name, 'w', encoding='utf-8') as file:
        # 将字典写入到文件作为JSON
        json.dump(data, file, indent=3, ensure_ascii=False)


def load_txt_to_lst(file_name):
    with open(file_name, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    return lines


def load_txt_to_str(file_name):
    with open(file_name, 'r', encoding='utf-8') as file:
        return file.read()


def save_data(file_name, data):
    with open(file_name, 'w', encoding='utf-8') as file:
        for line in data:
            file.write(line + "\n")


def save_txt(file_name, data):
    with open(file_name, 'w', encoding='utf-8') as file:
        file.write(data)
