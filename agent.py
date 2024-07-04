import jieba
import jieba.posseg as pseg
import copy
import json
from datetime import datetime

import utils
from retrieval import Retrieval
from model import QwenModel, GPTModel


class RoleAgent:

    def __init__(self,
                 role_name,
                 system_prompt,
                 config):
        self.config = config
        self.role_name = role_name
        self.user_name = "旅行者"
        self.system_prompt = system_prompt

        self.first_user_prompt = ""  # 记录第一条用户提问，用于保存对话历史
        self.rag_top_k = config.rag_top_k
        self.embedding_top_k = config.embedding_top_k
        self.top_k_per_word = config.top_k_per_word
        self.kb_max_len = config.kb_max_len
        self.show_rag_detail = config.show_rag_detail

        # 加载embedding和知识库部分
        if self.rag_top_k > 0:
            self.retrieval = Retrieval(config.embedding_model_name_or_path, config.embedding_device,
                                       config.reranker_model_name_or_path, config.reranker_device)

            if config.first_load_memory and config.first_load_kb_path and config.idx_kb_path:
                self.first_load_memory(kb_path=config.first_load_kb_path,
                                       idx_kb_path=config.idx_kb_path)
                print(f"New kb index saved to {config.idx_kb_path}")
            elif config.idx_kb_path:
                self.retrieval.load_index(config.idx_kb_path)

            if config.keywords_path:
                self.genshin_words = utils.load_json(config.keywords_path)
                self._load_n_words(self.genshin_words)
                self.retrieval.load_n_lst(self.genshin_words)

        self.default_messages = [
            {"role": "system", "content": system_prompt}
        ]
        self.messages = copy.deepcopy(self.default_messages)

        self.dialog_window = config.dialog_window if config.dialog_window > 0 else 3
        self.temperature = config.temperature
        self.max_new_tokens = config.max_new_tokens

        if config.model_name_or_path in ["gpt-3.5-turbo", "gpt-4-turbo", "gpt-4o"]:
            self.model = GPTModel(model_name='gpt-4o',
                                  api_key=config.gpt_api_key,
                                  max_token=config.max_new_tokens,
                                  temperature=config.temperature)
        else:
            self.model = QwenModel(cache_dir=config.model_cache_dir,
                                   model_name_or_path=config.model_name_or_path,
                                   quantized=config.model_quantized,
                                   device=config.llm_device,
                                   token=config.hf_token)

        self.pre_mem_prompt = ""

    def first_load_memory(self, kb_path, idx_kb_path):
        # 选择加载的知识库类型
        if kb_path.endswith('.csv'):
            self.retrieval.load_csv(kb_path)
        elif kb_path.endswith('.txt'):
            self.retrieval.load_txt(kb_path)
        self.retrieval.save_index(idx_kb_path)

    def set_user_name(self, user_name):
        self.user_name = user_name

    @staticmethod
    def _load_n_words(words):
        for word in words:
            jieba.add_word(word, tag='n_genshin')

    def load_messages(self, path):
        with open(path, 'r', encoding='utf-8') as file:
            self.messages = json.load(file)

    def save_messages(self):
        now = datetime.now()
        path = "history/" + self.first_user_prompt + " " + now.strftime("%Y-%m-%d_%H-%M-%S.json")
        # 保存成json
        with open(path, 'w', encoding='utf-8') as file:
            json.dump(self.messages, file, indent=4, ensure_ascii=False)
        return path

    def clear_messages(self):
        self.messages = copy.deepcopy(self.default_messages)

    def re_chat(self):
        if len(self.messages) == 1:
            return None
        self.messages = self.messages[: -1]
        res = self.model.chat(self.messages)
        self.messages.append({"role": "assistant", "content": res})
        return res

    def next_chat(self):
        if len(self.messages) <= 1:
            return None
        self.messages.append({"role": "user", "content": self.role_name + "：" + self.messages[-1]["content"]})
        res = self.model.chat(self.messages)
        self.messages.append({"role": "assistant", "content": res})
        return res

    def chat(self, user_prompt):
        # 记录潜在的文件名
        if len(self.messages) == 1:
            self.first_user_prompt = user_prompt

        # RAG策略
        if self.rag_top_k > 0:
            # 提取名词
            ori_n_lst = self._get_n(user_prompt)
            # 扩充名词
            n_lst = self.retrieval.search_n(ori_n_lst, top_k_per_word=self.top_k_per_word)

            for n in ori_n_lst:
                if n not in n_lst:
                    n_lst.append(n)

            if len(n_lst) == 0:  # 如果没有名词，则加入名词为角色名字
                n_lst.append(self.role_name)

            n_prompt = " ".join(n_lst)
            # search_prompt = n_prompt + '\n' + user_prompt
            search_prompt = n_prompt

            mem_prompt = self.role_name + "的记忆片段：\n" + "=" * 6 + "\n"
            mem_prompt += ("脑海中的关键词：" + n_prompt + '\n\n')
            mem_prompt += self._search_relevant_memory(search_prompt, reranker_query=user_prompt)

            self.pre_mem_prompt = mem_prompt
            if self.show_rag_detail:
                print(mem_prompt)

            user_prompt = mem_prompt + '\n\n' + self.user_name + "：" + user_prompt

        # LLM通信
        self.messages.append({"role": "user", "content": user_prompt})

        res = ''
        for chunk in self.model.stream_chat(self.messages, self.temperature, self.max_new_tokens):
            res += chunk
            yield chunk

        self.messages.append({"role": "assistant", "content": res})

        if len(self.messages) - 1 >= self.dialog_window * 2:
            self.messages.pop(1)
            self.messages.pop(1)
        return res

    def _search_relevant_memory(self, search_prompt, reranker_query):
        docs_and_scores = self.retrieval.search(search_prompt,
                                                reranker_query=reranker_query,
                                                embedding_top_k=self.embedding_top_k,
                                                reranker_top_k=self.rag_top_k)
        mem_prompt = ""
        kb_str = ''
        for i, doc in enumerate(docs_and_scores):
            kb_str += f"片段{str(i + 1)}：\n"
            kb_str += doc[1]
            if i == len(docs_and_scores) - 1:
                kb_str += '\n'
            else:
                kb_str += '\n\n'
            if len(kb_str) > self.kb_max_len:
                break
        mem_prompt += kb_str
        mem_prompt += "=" * 10 + "\n\n"
        return mem_prompt

    @staticmethod
    def _get_n(prompt):
        words = pseg.cut(prompt)
        n_lst = []
        for word, flag in words:
            # print(f'{word}/{flag}', end=' ')
            if flag[0] == 'n':
                # print(word, end=' ')
                if word not in n_lst:
                    n_lst.append(word)
        # print()
        return n_lst
