from langchain_community.vectorstores.faiss import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


class Reranker:

    def __init__(self, reranker_model_name, device, cache_dir=None):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(reranker_model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(reranker_model_name,
                                                                        cache_dir=cache_dir,
                                                                        device_map=device)
        self.model.eval()

    @staticmethod
    def make_pairs(query, docs):
        pairs = []
        for doc in docs:
            new_pair = [query, doc]
            pairs.append(new_pair)
        return pairs

    def calc_score(self, query, docs):
        pairs = self.make_pairs(query, docs)
        with torch.no_grad():
            inputs = self.tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=1024).\
                to(self.device)
            scores = self.model(**inputs, return_dict=True).logits.view(-1, ).float()
        res = []
        for i in range(len(docs)):
            res.append([scores[i].item(), docs[i]])
        return res


class Retrieval:

    def __init__(self, embedding_model_name, embedding_device,
                 reranker_model_name, reranker_device, cache_dir=None):
        self.embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name,
                                                     cache_folder=cache_dir,
                                                     model_kwargs={'device': embedding_device})
        self.reranker = Reranker(reranker_model_name, reranker_device, cache_dir)
        self.documents = None
        self.split_docs = None
        self.faiss = FAISS
        self.vector_store = None
        self.n_lst = None
        self.n_vs = None

    def load_csv(self, file_path):
        from langchain_community.document_loaders import CSVLoader
        loader = CSVLoader(file_path, encoding='utf-8')
        tsp = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        self.documents = loader.load()
        self.split_docs = tsp.split_documents(self.documents)
        self.vector_store = self.faiss.from_documents(self.split_docs, self.embedding_model)

    def load_txt(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            txt = file.read()
        self.split_docs = txt.split(sep='\n\n')
        self.vector_store = self.faiss.from_texts(self.split_docs, self.embedding_model)

    def load_n_lst(self, n_lst):
        self.n_lst = n_lst
        self.n_vs = self.faiss.from_texts(self.n_lst, self.embedding_model)

    def search_n(self, words, top_k_per_word=3):
        result_words_and_scores = []
        for word in words:
            n_and_scores = self.n_vs.similarity_search_with_score(word, top_k_per_word)
            for n_s in n_and_scores:
                new_word = n_s[0].page_content
                if not any(new_word == w for w, _ in result_words_and_scores):
                    result_words_and_scores.append((new_word, n_s[1]))
        result_words_and_scores = sorted(result_words_and_scores, key=lambda x: x[1])
        result_words = [word for word, _ in result_words_and_scores]
        return result_words

    def load_index(self, path):
        self.vector_store = FAISS.load_local(path, self.embedding_model, allow_dangerous_deserialization=True)

    def save_index(self, path):
        self.vector_store.save_local(path)

    def search_by_embedding(self, query, top_k=3):
        docs_and_scores = self.vector_store.similarity_search(query, top_k)
        return docs_and_scores

    def search(self, query, reranker_query, embedding_top_k=6, reranker_top_k=3):
        # 向量数据库搜索
        docs_and_scores = self.search_by_embedding(query, embedding_top_k)
        # reranker打分排序，取前top_k个
        docs = []
        for doc in docs_and_scores:
            docs.append(doc.page_content)
        rerank_res = self.reranker.calc_score(reranker_query, docs)
        sorted_rerank_res = sorted(rerank_res, key=lambda x: x[0], reverse=True)
        res = sorted_rerank_res[:reranker_top_k]
        return res
