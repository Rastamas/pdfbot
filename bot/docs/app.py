from llama_index import SimpleDirectoryReader, GPTListIndex, GPTVectorStoreIndex, LLMPredictor, PromptHelper, StorageContext, load_index_from_storage
from langchain.chat_models import ChatOpenAI
import gradio as gr
import openai
import sys
import os

os.environ["OPENAI_API_KEY"] = 'XXX'


openai.api_key = os.environ["OPENAI_API_KEY"]

def construct_index(directory_path):
    max_input_size = 4096
    num_outputs = 512
    max_chunk_overlap = 20
    chunk_size_limit = 600

    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

    llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo", max_tokens=num_outputs))

    documents = SimpleDirectoryReader(directory_path).load_data()

    index = GPTVectorStoreIndex.from_documents(documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper)
    index.storage_context.persist()

    return index

def chatbot(input_text):
    # rebuild storage context
    storage_context = StorageContext.from_defaults(persist_dir='./storage')
    # load index
    index = load_index_from_storage(storage_context)
    response = index.as_query_engine().query(input_text)
    return response.response

iface = gr.Interface(fn=chatbot,
                     inputs=gr.components.Textbox(lines=7, label="Enter your text"),
                     outputs="text",
                     title="Custom-trained AI Chatbot")

index = construct_index("docs")
iface.launch(share=True)