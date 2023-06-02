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
    loader = UnstructuredMarkdownLoader(md_file_path)
    markDownFile = loader.load()
# If there are no markdown files, handle the empty directory case
else:
    raise Exception('The directory does not contain any markdown files')

# Setting up LLMs
llmOpenAi = OpenAI(openai_api_key=OPENAI_TOKEN, temperature=0.7, max_tokens=3000)

llmVicuna = Replicate(model="replicate/vicuna-13b:6282abe6a492de4145d7bb601023762212f9ddbbe78278bd6771c8b3b2f2a13b",input={"max_length":2000}, verbose=True)

llmText2Img = Replicate(model="mcai/dreamshaper-v6:8b0deb0306a54dec7c6e5c955b83320f5d8b7fc659769667e0e71e56d0f488ed")

# ROLES BEGIN HERE
"""
Threadoor 
- Objective -> Take the notes I create from a file (markdown, txt, etc.) and convert them into a twitter thread.
"""

threadoor = hp.chain(
    llm=llmOpenAi,
    template="""
    Role: 
    You are a Twitter Thread Creator. You have the ability to explain difficult topics down to their fundamentals using first principle thinking applying the Feynman Technique.
    Goal:
    - Convert the markdown notes into a Twitter thread without the use of exclaimation points, hashtags and emojis.
    - Ensure a logical flow and coherence of the Twitter thread while maintaining the {noteStructure} of the notes.
    Markdown Notes: {mdNotes}
    Twitter thread:

    """,
    inputVariables=["noteStructure", "mdNotes"],
    output_key="thread"
)

hookoor = hp.chain(
    llm=llmOpenAi,
    template="""
    Role: You are a Twitter Thread headline generator.
    Goal: Using the given Twitter thread construct 2 introductory tweet without using exclaimation points, hashtags in a persuasive writing style based the given Format.
    Format: 
        Problem: (Name the problem)
        Steps: (Pinpoint actionable steps)
        Output: (Celebrate outcome)
    Twitter Thread: {thread}
    Introductory Tweet:
    #1: 

    """,
    inputVariables=['thread'],
    output_key="hook"
)

promptoor = hp.chain(
    llm=llmOpenAi,
    template="""
    Role: You are an agent who writes descriptive short text phrases which will used to generate images.
    Format: 
        Scene = {scene}
        Art Style = (art style eg: normcore, cubism, abstract, surrealism, minimalism, realism, pop art)
        Artist = (japanese manga author's name eg: Eiichiro Oda, Hiromu Arakawa, Akira Toriyama, Katsuhiro Otomo)
        Medium = (art medium eg: illustration, painting, photograph, pastel, sculpture, drawing, ink, digital art) 
    Goal: Using the Format, create a 60 word short text phrase depicting a furturistic scene. The more detailed and imaginative your description, the more interesting the resulting image will be.
    Generated text phrase: 

    """,
    inputVariables=["scene"],
    output_key="prompt"
)

imgGenoor = hp.chain(
    llm=llmText2Img,
    template="{prompt}",
    inputVariables=["prompt"],
    output_key="img"
)

# CHAIN START HERE
chain = SequentialChain(
    memory=SimpleMemory(memories={"mdNotes":markDownFile}),
    chains=[
        threadoor,
        hookoor,
        promptoor,
        imgGenoor
    ],
    input_variables=["noteStructure", "scene"],
    output_variables=["thread", "hook", "prompt", "img"],
    verbose=True
)

# FILE PRINTING
# Get the current date and time
today = datetime.datetime.now().strftime('%Y-%m-%d')

# Get the file name
mdFilename = os.path.basename(md_file_path)

repsonse = list(
    chain({"noteStructure":"tone and vocabulary but change the voice to reflect Mark Manson", "scene":"futuristic details, cyberpunk atmosphere"}).items())[-4:]

# Open a new file for writing
with open(f'{threadDirectory}\\{today}_{mdFilename}', 'w', encoding="utf-8") as f:
    # Write the last three key-value pairs to the file, one per line
    for key, value in repsonse:
        # Write the key with '# Key:' prefix and newline character
        f.write(f"\n# Key: {key}\n")
        # Write the value with '# Value' prefix and newline character
        f.write(f"# Value: \n{value} \n")

# Send a request to the server and get the response -> the number represents the position in the list of where the image is (the third position in the list is the img as of 29/05/23)
response = requests.get(repsonse[3][1])

# Open the image using Pillow
img = Image.open(BytesIO(response.content))

# Save the image to disk
img.save(f'{imagesDirectory}\\{today}_hookImage.png')