import configparser


class ConfigParserNoLower(configparser.ConfigParser):
    def __init__(self):
        configparser.ConfigParser.__init__(self)

    def optionxform(self, optionstr):
        return optionstr


config_file = ConfigParserNoLower()
config_file.read('config.ini', encoding='utf-8-sig')


class BaseConfig:

    def __init__(self):

        self.user_name = config_file.get('BASE', 'user_name')
        # LLM
        self.model_name_or_path = config_file.get('LLM', 'model_name_or_path')
        self.model_cache_dir = config_file.get('LLM', 'model_cache_dir')
        self.model_quantized = config_file.getboolean('LLM', 'model_quantized')
        self.temperature = config_file.getfloat('LLM', 'temperature')
        self.max_new_tokens = config_file.getint('LLM', 'max_new_tokens')
        self.llm_device = config_file.get('LLM', 'llm_device')
        self.hf_token = config_file.get('LLM', 'hf_token')

        # RAG
        self.embedding_model_name_or_path = config_file.get('RAG', 'embedding_model_name_or_path')
        self.embedding_device = config_file.get('RAG', 'embedding_device')
        self.first_load_memory = config_file.getboolean('RAG', 'first_load_memory')
        self.first_load_kb_path = config_file.get('RAG', 'first_load_kb_path')
        self.idx_kb_path = config_file.get('RAG', 'idx_kb_path')
        self.keywords_path = config_file.get('RAG', 'keywords_path')
        self.rag_top_k = config_file.getint('RAG', 'rag_top_k')
        self.top_k_per_word = config_file.getint('RAG', 'top_k_per_word')
        self.kb_max_len = config_file.getint('RAG', 'kb_max_len')
        self.show_rag_detail = config_file.getboolean('RAG', 'show_rag_detail')


base_config = BaseConfig()
