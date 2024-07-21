from llama_index.core.storage.chat_store import SimpleChatStore
from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core import Settings, VectorStoreIndex, StorageContext, SimpleDirectoryReader, Document
from llama_index.llms.together import TogetherLLM
from llama_index.embeddings.together import TogetherEmbedding
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.readers.web import SimpleWebPageReader
from llama_index.core import PromptTemplate

llm = TogetherLLM(
    model="meta-llama/Meta-Llama-3-70B-Instruct-Turbo", api_key="dae9b33beef8cc57bc9475d53519e9e70037e6820b8c2d54b0b2f349e11101b0"
)

embed_model = TogetherEmbedding(
    model_name = "togethercomputer/m2-bert-80M-8k-retrieval", api_key="dae9b33beef8cc57bc9475d53519e9e70037e6820b8c2d54b0b2f349e11101b0"
)

documents = SimpleDirectoryReader(input_dir="./data", recursive=True).load_data()
vector_store = MilvusVectorStore(uri="./milvus_demo.db", dim=768, overwrite=True)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(documents=documents)

Settings.embed_model = embed_model
Settings.llm = llm

promptTemplate = (
    "We have provided context information below. \n"
    "You are expected to provide appropriate over the counter medications to patients. \n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "Given this information, please answer the question: {query_str}\n"
    "If the question is out of the scope of the given information, answer the questions to the best of your ability \n"
    "Do not mention details about the context provided to you and directly answer the question.\n"
)

qa_template = PromptTemplate(promptTemplate)


try:
    chat_store = SimpleChatStore.from_persist_path(
        persist_path="chat_memory.json"
    )
except FileNotFoundError:
    chat_store = SimpleChatStore()


memory = ChatMemoryBuffer.from_defaults(
    token_limit=8000,
    chat_store=chat_store,
    chat_store_key="user_X"
    )  

chat_engine = SimpleChatEngine.from_defaults(memory=memory)
while True:
    user_message = input("You: ")
    if user_message.lower() == 'exit':
        print("Exiting chat...")
        break
    prompt = qa_template.format(context_str = documents, query_str = user_message)
    response = chat_engine.chat(prompt)
    print(f"Chatbot: {response}")

chat_store.persist(persist_path="chat_memory.json")