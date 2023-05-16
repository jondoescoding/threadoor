from langchain import PromptTemplate, LLMChain
from langchain.llms import GPT4All
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import Document
from dotenv import load_dotenv
import pdfplumber
import os

# Loading environment variables
load_dotenv()
pdf_file_path = r"C:\_CODING\PYTHON\PROJECTS\contentCreator\docs\toBeIngested\promptLength.pdf"
persist_directory = os.environ.get('PERSIST_DIRECTORY')
llama_embeddings_model = os.environ.get('LLAMA_EMBEDDINGS_MODEL')
openAiToken = os.environ.get('openAi')

with pdfplumber.open(pdf_file_path) as pdf:
    pages = pdf.pages
    file_content = "\n".join(page.extract_text() for page in pages)

doc = Document(page_content=file_content)

template = """Question: {question}

Answer: Let's think step by step."""
prompt = PromptTemplate(template=template, input_variables=["question"])

embedding = OpenAIEmbeddings(openai_api_key=openAiToken)
vectordb = Chroma.from_documents(documents=[doc], embedding=embedding, persist_directory=persist_directory)
vectordb.persist()

model = GPT4All(model=r"C:\_CODING\PYTHON\PROJECTS\contentCreator\models\LLMs\ggml-gpt4all-j-v1.3-groovy.bin", n_ctx=512, n_threads=8)

question = "What is the main topic of the file?"
formatted_prompt = prompt.format(question=question, file_content=file_content)

response = model(formatted_prompt)

print(response)