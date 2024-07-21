
from llama_index.core import Settings, VectorStoreIndex, StorageContext, SimpleDirectoryReader, Document, ServiceContext
from llama_index.llms.together import TogetherLLM
from llama_index.embeddings.together import TogetherEmbedding
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core import load_index_from_storage
from llama_index.core.callbacks.base import CallbackManager
from prompts import OTC_ASSISTANT_TEMPLATE
import chainlit as cl

documents = SimpleDirectoryReader(input_dir="./data", recursive=True).load_data()

@cl.on_chat_start
async def factory():
    embed_model = TogetherEmbedding(
        model_name="togethercomputer/m2-bert-80M-8k-retrieval", api_key="dae9b33beef8cc57bc9475d53519e9e70037e6820b8c2d54b0b2f349e11101b0"
    )

    llm = TogetherLLM(
        model="meta-llama/Meta-Llama-3-70B-Instruct-Turbo", api_key="dae9b33beef8cc57bc9475d53519e9e70037e6820b8c2d54b0b2f349e11101b0"
    )

    service_context = ServiceContext.from_defaults(embed_model=embed_model, llm=llm,
                        callback_manager=CallbackManager([cl.LlamaIndexCallbackHandler()]),
    )

    try:
        # rebuild storage context
        storage_context = StorageContext.from_defaults(persist_dir="./medicine_index",service_context=service_context)
        # load index
        index = load_index_from_storage(storage_context, service_context=service_context)
    except:
        index = VectorStoreIndex.from_documents(documents,service_context=service_context)
        index.storage_context.persist("medicine_index")

    query_engine = index.as_query_engine(
        service_context=service_context,
        similarity_top_k=2,
        # streaming=True,
    )

    await cl.Message(
        author="Assistant", content="Hey there! I'm Latom and I am built to assist you with Over the counter medication information. How may I help you?"
    ).send()

    cl.user_session.set("query_engine", query_engine)

@cl.on_message
async def main(message: cl.Message):
    query_engine = cl.user_session.get("query_engine")  
    prompt = OTC_ASSISTANT_TEMPLATE.format(context=documents, question = message.content, chat_history=cl.user_session.get("chat_history", ""))
    response = await cl.make_async(query_engine.query)(prompt)

    response_message = cl.Message(content="")

    for token in response.response:
        await response_message.stream_token(token=token)

    await response_message.send()