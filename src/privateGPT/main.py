# Python
import os
# DOT-ENV
from dotenv import load_dotenv
# LangChain
import langchain
from langchain.chains import RetrievalQA
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.embeddings import HuggingFaceEmbeddings, LlamaCppEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
# Cache -> Langchain + GPTCache
from langchain.cache import GPTCache
from gptcache.adapter.api import init_similar_cache
from constants import CHROMA_SETTINGS

# Loading environment variables
load_dotenv()

# Embeddings
embeddings_model = os.environ.get("EMBEDDINGS_MODEL_NAME")
llama_embeddings_model = os.environ.get("LLAMA_EMBEDDINGS_MODEL")
# VectorStore
persist_directory = os.environ.get('PERSIST_DIRECTORY') 
# LLM Information
model_type = os.environ.get('MODEL_TYPE')
model_path = os.environ.get('MODEL_PATH')
model_n_ctx = int(os.environ.get('MODEL_N_CTX'))
chunk_size = int(os.environ.get('CHUNK_SIZE'))

def main():
    # Hugging Face Embeddings
    huggingFaceEmbeddings = HuggingFaceEmbeddings(model_name=embeddings_model)

    # LLama Embeddings
    llama = LlamaCppEmbeddings(model_path=llama_embeddings_model, n_ctx=model_n_ctx)
    
    # Database
    db = Chroma(persist_directory=persist_directory, embedding_function=huggingFaceEmbeddings, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever()
    
    # Prepare the LLM
    callbacks = [StreamingStdOutCallbackHandler()]
    
    llm = GPT4All(model=model_path, n_ctx=model_n_ctx, backend='gptj', callbacks=callbacks, verbose=False)

    # Cache
    langchain.llm_cache = GPTCache(init_func=lambda cache: init_similar_cache(cache_obj=cache))

    # template = """
    # Answer the question based on the context below. If the
    # question cannot be answered using the information provided answer with "I don't know".
    
    # Request: You are an expert twitter thread writer. You goal is to write a engaging Twitter Thread with the given context.

    # Answer: {thread}
    
    # """
    
    # prompt = PromptTemplate(template=template, input_variables=["thread"])
    
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
    
    # Interactive questions and answers
    while True:
        query = input("\nEnter a query: ")
        if query == "exit":
            break

        if query.strip() == "":
            print("error: query empty!")
            continue
        
        # Get the answer from the chain
        res = qa(query)    
        answer, docs = res['result'], res['source_documents']

        # Print the result
        print("\n\n> Question:")
        print(query)
        print("\n> Answer:")
        print(answer)
        
        # Print the relevant sources used for the answer
        for document in docs:
            print("\n> " + document.metadata["source"] + ":")
            print(document.page_content)

if __name__ == "__main__":
    main()
