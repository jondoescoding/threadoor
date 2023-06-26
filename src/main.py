# Python 
import glob
import os
import datetime
import requests
from dotenv import load_dotenv
from PIL import Image
from io import BytesIO
# Langchain
from langchain.llms import *
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import SequentialChain
from langchain.document_loaders import UnstructuredMarkdownLoader
from langchain.memory import SimpleMemory
# Custom Utilities
import helper as hp

# ENVIRONMENT VARIABLES
load_dotenv()  # take environment variables from .env
OPENAI_TOKEN = os.environ.get('openAi')
REPLICATE_API_TOKEN = os.environ.get('replicate')
os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN

# Set the directory paths
contentDirectory = os.environ.get("CONTENT_FOLDER")
threadDirectory = os.environ.get("THREAD_FOLDER")
imagesDirectory = os.environ.get("IMAGES_FOLDER")

# Get a list of all markdown files in the directory using glob
md_files = glob.glob(contentDirectory + '/*.md')

# Check if there is more than one markdown file in the directory
if len(md_files) > 1:
    raise Exception('There is more than one markdown file in the directory')

# If there is exactly one markdown file, print its path
elif len(md_files) == 1:
    md_file_path = md_files[0]
    print(f'The markdown file path is: {md_file_path}')
    textSplitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=128)
    loader = UnstructuredMarkdownLoader(md_file_path)
    markDownFile = loader.load_and_split(textSplitter)

# If there are no markdown files, handle the empty directory case
else:
    raise Exception('The directory does not contain any markdown files')

# Setting up LLMs
llmOpenAi = hp.initializeLLM(
    nameOfLLM="OpenAI", 
    temperature=0.75,
    max_tokens=2000,
    openai_api_key=OPENAI_TOKEN)

# ROLES BEGIN HERE
"""
Threadoor 
- Objective -> Take the notes I create from a file (markdown, txt, etc.) and convert them into a twitter thread.
"""

threadoor = hp.chain(
    llm=llmOpenAi,
    template="""
    Goal: Paraphrase, shorten and convert the markdown notes into a Twitter Thread without the use of exclaimation points, hashtags and emojis. Ensure a logical flow and coherence of the Twitter thread while maintaining the {noteStructure} of the notes.
    Markdown Notes: {mdNotes}
    Twitter thread:
    """,
    inputVariables=["noteStructure", "mdNotes"],
    output_key="thread"
)

hookoor = hp.chain(
    llm=llmOpenAi,
    template="""
    Role: You are a headline generator for Twitter Thread.
    Goal: Using the given Twitter thread construct 3 introductory tweet without using exclaimation points and hashtags in Gary Halbert's writing style to capture the attention of a tech user based the given Format.
    Format: 
        Problem: (Name the problem)
        Choose one option for the end of the tweet: 👇🧵, ↓↓↓
    Twitter Thread: {thread}
    Introductory Tweets:

    """,
    inputVariables=['thread'],
    output_key="hook"
)

# CHAIN START HERE
chain = SequentialChain(
    memory=SimpleMemory(memories={"mdNotes":markDownFile}),
    chains=[
        threadoor,
        hookoor
    ],
    input_variables=["noteStructure"],
    output_variables=["thread", "hook"],
    verbose=True
)

# FILE PRINTING
# Get the current date and time
today = datetime.datetime.now().strftime('%Y-%m-%d')

# Get the file name
mdFilename = os.path.basename(md_file_path)

repsonse = list(
    chain({"noteStructure":"tone, voice and vocabulary of the content's writing style"}).items())[-4:]


# Open a new file for writing
print("Writing To File")
with open(f'{threadDirectory}\\{today}_{mdFilename}', 'w', encoding="utf-8") as f:
    # Write the last three key-value pairs to the file, one per line for only the first two output variables
    for key, value in repsonse[:2]:
        # Write the key with '# Key:' prefix and newline character
        f.write(f"\n# Key: {key}\n")
        # Write the value with '# Value' prefix and newline character
        f.write(f"# Value: \n{value} \n")

# Send a request to the server and get the response -> the number represents the position in the list of where the image is (the third position in the list is the img as of 29/05/23)
response = requests.get(repsonse[3][1])

# Open the image using Pillow
img = Image.open(BytesIO(response.content))

# Save the image to disk
print("Saving image")
img.save(f'{imagesDirectory}\\{today}_hookImage.png')