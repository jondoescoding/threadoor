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

# Set the directory path
directory = os.environ.get("CONTENT_FOLDER")

# Get a list of all markdown files in the directory using glob
md_files = glob.glob(directory + '/*.md')

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
llmOpenAi = OpenAI(openai_api_key=OPENAI_TOKEN, temperature=0.65, max_tokens=500)

llmVicuna = Replicate(model="replicate/vicuna-13b:6282abe6a492de4145d7bb601023762212f9ddbbe78278bd6771c8b3b2f2a13b",input={"max_length":2000}, verbose=True)

llmText2Img = Replicate(model="ai-forever/kandinsky-2:601eea49d49003e6ea75a11527209c4f510a93e2112c969d548fbb45b9c4f19f")

# ROLES BEGIN HERE
"""
Threadoor 
- Objective -> Take the notes I create from a file (markdown, txt, etc.) and convert them into a twitter thread.
"""

threadoor = hp.chain(
    llm=llmOpenAi,
    template="""
    ### Instructions ###
    - You are writing agent that doesn't use exclaimation points, hashtags or emojis.
    - Paraphase the main points and ideas of the notes as a Twitter thread.
    - Highlight each of the key concepts from the notes in each tweet.
    - Break the content into smaller and digestible tweets.
    - Write clear and concise tweets for each point without altering the original meaning.
    - Ensure a logical flow and coherence in the Twitter thread while maintaining the {noteStructure} of the notes.
    Markdown Notes: {mdNotes}
    Twitter thread:

    """,
    inputVariables=["noteStructure", "mdNotes"],
    output_key="thread"
)

hookoor = hp.chain(
    llm=llmOpenAi,
    template="""
    Role: You are a content writing agent that uses a given Twitter thread to construct an engaging introductory tweet that does not contain any exclaimation marks or hashtags.
    Goal: Create 3 short introductory tweets based on the Twitter thread using persuasive writing style.
    Formula: Problem: (Name the problem), Steps: (Pinpoint actionable steps), Output: (Celebrate outcome)
    Twitter Thread: {thread}
    Introductory Tweet:

    """,
    inputVariables=['thread'],
    output_key="hook"
)

promptoor = hp.chain(
    llm=llmOpenAi,
    template="""
    Role: You are an agent who writes descriptive short text phrases which will used to generate images.
    Format: Scene: (cyberpunk scene we are depicting), Keywords: (5 descriptive keywords related to robotics, ai, normcore and cyberpunk), Art Style: (famous art style), Artist: (famous japanese manga artist name), Medium: (art medium) 
    Goal: Using the Format, create a 60 word short text phrase depicting a furturistic scene based on the theme of the Twitter Thread.
    Twitter Thread: {thread}
    Generated text phrase:

    """,
    inputVariables=["thread"],
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
    input_variables=["noteStructure"],
    output_variables=["thread", "hook", "prompt", "img"],
    verbose=True
)

# FILE PRINTING

# Get the current date and time
now = datetime.datetime.now()

# Create a string with today's date in the format YYYY-MM-DD
date_string = now.strftime('%Y-%m-%d')

# Create a file with today's date in the name
filename = f'{date_string}'

# Get the file name
md_file_name = os.path.basename(md_file_path)

repsonse = list(
    chain({"noteStructure":"tone, voice, vocabulary and sentence structure"}).items())[-4:]

# Open a new file for writing
with open(f'src\\threadoor\\threads\\{filename}_{md_file_name}', 'w', encoding="utf-8") as f:
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
img.save('src\\threadoor\\images\\hookImage.png')