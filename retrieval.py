from langchain_community.vectorstores.faiss import FAISS
from langchain.text_splitter import CharacterTextSplitter


class Retrieval:

    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
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
        self.vector_store = FAISS.load_local(path, self.embedding_model)

    def save_index(self, path):
        self.vector_store.save_local(path)

    def search(self, query, top_k=3):
        docs_and_scores = self.vector_store.similarity_search(query, top_k)
        return docs_and_scores

    def search_and_return_string(self, query, top_k=3):
        docs_and_scores = self.search(query, top_k)
        res = ""
        for doc in docs_and_scores:
            res += (doc.page_content + '\n' + "=" * 10 + '\n')
        return res
