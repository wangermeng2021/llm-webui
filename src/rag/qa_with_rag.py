
from langchain.text_splitter import RecursiveCharacterTextSplitter ,CharacterTextSplitter
from langchain.vectorstores import Chroma,FAISS
from langchain.chains import ConversationalRetrievalChain ,LLMChain ,ConversationChain ,RetrievalQA
from langchain.embeddings import LlamaCppEmbeddings ,HuggingFaceEmbeddings ,OpenAIEmbeddings
from langchain.document_loaders import TextLoader ,PyPDFLoader ,Docx2txtLoader
from langchain.prompts import PromptTemplate
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from chromadb.errors import InvalidDimensionException
from src.finetune.huggingface_inference import HuggingfaceInference
from src.finetune.llama_cpp_inference import LlamaCppInference


class QAWithRAG():
    def __init__(self ,config: dict ={}):
        self.text_splitter = None
        self.embedding_function = None
        self.vectorstore = None
        self.retriever = None
        self.chat_llm = None

        self.chat_history =[]
        # self.persist_directory = "./chroma_db"
        self.persist_directory = None
        self.qa = None
        self.langchain_llm = None
    def free_memory(self):
        if self.chat_llm:
            self.chat_llm.free_memory()
            del self.chat_llm
            self.chat_llm = None
        if self.langchain_llm:
            del self.langchain_llm
            self.langchain_llm = None
        if self.qa:
            del self.qa
            self.qa = None


    def get_text_splitter(self ,chunk_size ,chunk_overlap ,separators):
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len,
                                                            separators=separators)
    def load_embedding_model(self ,model_path=""):
        self.embedding_function = HuggingFaceEmbeddings(model_name=model_path ,model_kwargs = {'device': 'cpu'})
    def load_chat_model(self ,model_path,using_4bit_quantization,low_cpu_mem_usage,
                        max_new_tokens, temperature, top_k, top_p, repeat_penalty
                        ):
        self.set_prompt_template(model_path)
        load_model_status = 0
        if model_path.split('.')[-1] == "gguf":
            self.chat_llm = LlamaCppInference(model_path=model_path, max_new_tokens=max_new_tokens, temperature=temperature,
                                              top_k=top_k, top_p=top_p, repetition_penalty=repeat_penalty)
            load_model_status, msg = self.chat_llm.load_model()
            self.langchain_llm = self.chat_llm.model
        else:
            self.chat_llm = HuggingfaceInference(model_path, max_new_tokens, temperature, top_p, top_k, repeat_penalty, using_4bit_quantization,low_cpu_mem_usage)
            load_model_status, msg = self.chat_llm.load_model()
            self.langchain_llm = HuggingFacePipeline(pipeline=self.chat_llm.model)

        return load_model_status, msg

    #
    def get_document_data(self ,doc_path):
        self.chat_history = []
        self.chat_history.clear()
        self.doc_ext = doc_path.split('.')[-1]
        if self.doc_ext == "txt":
            loader = TextLoader(doc_path, encoding='utf8')
        elif self.doc_ext == "pdf":
            loader = PyPDFLoader(doc_path)
        elif self.doc_ext == "docx":
            loader = Docx2txtLoader(doc_path)
        else:
            raise ValueError(f"Unsupported format: {self.doc_ext}")
        data = loader.load()
        return data
    def add_document_to_vector_store(self, doc_path ,search_top_k ,search_score_threshold):
        data = self.get_document_data(doc_path)
        data = self.text_splitter.split_documents(data)
        try:
            self.vectorstore = Chroma.from_documents(data, self.embedding_function
                                                     ,collection_metadata={"hnsw:space": "cosine"}
                                                     ,persist_directory=self.persist_directory)
            # self.vectorstore = FAISS.from_documents(data, self.embedding_function)                                                    
        except InvalidDimensionException:
            Chroma().delete_collection()
            self.vectorstore = Chroma.from_documents(data, self.embedding_function
                                                     ,collection_metadata={"hnsw:space": "cosine"}
                                                     ,persist_directory=self.persist_directory)
            # self.vectorstore = FAISS.from_documents(data, self.embedding_function)                                   
        self.set_retriever(search_top_k ,search_score_threshold)

    def set_retriever(self ,search_top_k ,score_threshold):
        self.retriever = self.vectorstore.as_retriever(search_type='similarity_score_threshold',
                                                       search_kwargs={'k': search_top_k, "score_threshold": score_threshold})
    def set_prompt_template(self ,chat_model_path):

        if chat_model_path.lower().find("mistral") >= 0 and chat_model_path.lower().find("instruct") >= 0:
            prompt_template = """<s>[INST] Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.\n\n
            Context: {context}\n
            Question: {question}\n
            Answer: [/INST]"""
        elif chat_model_path.lower().find("llama") >= 0 and chat_model_path.lower().find("chat") >= 0:
            prompt_template = """<s>[INST] Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.\n\n
            Context: {context}\n
            Question: {question}\n
            Answer: [/INST]"""
        elif chat_model_path.lower().find("zephyr") >= 0:
            prompt_template = """<|user|>\n Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.\n\n
            Context: {context}\n
            Question: {question}\n
            Answer: </s><|assistant|>\n"""
        else:
            prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.\n\n
            Context: {context}\n
            Question: {question}\n
            Answer:"""

        self.prompt_template = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )
    def generate(self, question):
        self.chat_history = []
        if self.retriever:

            chain_type_kwargs = {"prompt": self.prompt_template ,"verbose": False}
            self.qa = RetrievalQA.from_chain_type(llm=self.langchain_llm, chain_type="stuff", retriever=self.retriever,
                                                  return_source_documents=True,
                                                  chain_type_kwargs=chain_type_kwargs)
            result = self.qa({"query": question}, return_only_outputs=True)
            retrieved_txt_list = []
            if len(result['source_documents'] ) >0:
                if self.doc_ext == "txt":
                    for doc_text in result['source_documents']:
                        retrieved_txt_list.append(list(doc_text)[0][1])
                elif self.doc_ext == "pdf":
                    for doc_text in result['source_documents']:
                        retrieved_txt_list.append(list(doc_text)[0][1])
                elif self.doc_ext == "docx":
                    for doc_text in result['source_documents']:
                        retrieved_txt_list.append(list(doc_text)[0][1])
                answer = result['result']
            else:
                answer = "Sorry, I can't find any relevant information in document. " + result['result']
            return answer, retrieved_txt_list
        else:
            return "", retrieved_txt_list
