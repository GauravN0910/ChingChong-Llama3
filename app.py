import streamlit as st
from llama_index.core import VectorStoreIndex, ServiceContext, StorageContext
from llama_index.llms.together import TogetherLLM
from llama_index.embeddings.together import TogetherEmbedding
from llama_index.core import SimpleDirectoryReader
from llama_index.readers.json import JSONReader

# from llama_index.vector_stores.milvus import MilvusVectorStore

# import openai
# from llama_index.llms.openai import OpenAI

st.title("OTC Bot 3000")

open_llm = TogetherLLM(
    model="meta-llama/Meta-Llama-3-70B-Instruct-Turbo", 
    api_key="dae9b33beef8cc57bc9475d53519e9e70037e6820b8c2d54b0b2f349e11101b0"
)

embedded_model = TogetherEmbedding(
    model_name="togethercomputer/m2-bert-80M-8k-retrieval",
    api_key="dae9b33beef8cc57bc9475d53519e9e70037e6820b8c2d54b0b2f349e11101b0"
)

# vector_store = MilvusVectorStore(
#     uri="./milvus_demo.db", dim=768, overwrite=True
# )

# Initialize the chat message history
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "Give me a symptom and Ill recommend some OTC Medicine!"}
    ]

@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Loading OTC Docs"):

        # Initialize JSONReader
        reader = JSONReader(
            # The number of levels to go back in the JSON tree. Set to 0 to traverse all levels. Default is None.
            levels_back=0,
            # The maximum number of characters a JSON fragment would be collapsed in the output. Default is None.
            # collapse_length="<Collapse Length>",
            # If True, ensures that the output is ASCII-encoded. Default is False.
            # ensure_ascii="<Ensure ASCII>",
            # If True, indicates that the file is in JSONL (JSON Lines) format. Default is False.
            # is_jsonl="<Is JSONL>",
            # If True, removes lines containing only formatting from the output. Default is True.
            # clean_json="<Clean JSON>",
        )
        docs = reader.load_data(input_file='output_min.json', extra_info={})
        print(docs)

        # reader = SimpleDirectoryReader(input_files=["./output_min.json"]).load_data()
        # reader = SimpleDirectoryReader("./data/paul_graham").load_data()
        # docs = reader
        
        service_context = ServiceContext.from_defaults(embed_model=embedded_model,llm=open_llm)
        # storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # index = VectorStoreIndex.from_documents(docs, service_context=service_context,storage_context=storage_context)
        index = VectorStoreIndex.from_documents(docs, service_context=service_context)
        return index

index = load_data()

chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

# Prompt for user input and save to chat history
if prompt := st.chat_input("Your question"): 
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chat_engine.chat(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message) # Add response to message history