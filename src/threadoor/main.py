# Python 
import glob
import os
import datetime
# Langchain
from langchain.llms import *
from langchain.chains import SequentialChain
from langchain.document_loaders import UnstructuredMarkdownLoader
from langchain.memory import SimpleMemory
# Custom Utilities
import helper as hp 

# ENVIRONMENT VARIABLES
OPENAI_TOKEN = os.environ.get('openAi')

# Set the directory path
directory = r"C:\_CODING\PYTHON\PROJECTS\contentCreator\docs\toBeIngested"

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

# Setting up OpenAi
llm = OpenAI(openai_api_key=OPENAI_TOKEN, temperature=0.7, max_tokens=500)

# ROLES BEGIN HERE

# Writer
writer = hp.Role(
    llm=llm,
    template="""
    ### Instructions ###
    Role: The following is an agent which generates Twitter threads from raw notes. As if explaining the concept to a 10 year old, the agent should use only the information provided in the notes.
    Format: Provide an explanation of the concept as a way of introducing the reader to the topic that will be discussed, illustrating the concept to a parallel in real life and and tying it to other points in the discussion. 
    Voice and style guide: Use a {technique} writing style in {writerVoice}'s voice. Be conversational and use natural language.
    Task: Based on the instructions given, rewrite the raw markdown notes into a Twitter Thread applying the specific voice and styling guide
    Raw Notes: {notes}
    Draft:

    """
)

# Editor
editor = hp.Role(
    llm=llm,
    template="""
    Role: You are a professional Editor who was worked in the online written industry for over 10 years. 
    Objective: Given a draft for a Twitter thread you are to provide a bulleted list of feedback addressing grammar, sentence structure, and if there are any areas within the draft where there is usage of technical terms of complex languages, re-write these sections in simpler words.
    Draft: {draft}
    Bullet List of Feedback:

    """
)

production = hp.Role(
    llm=llm,
    template="""
    Role: You are a professional copywriter.
    Objective: The goal is to construct a multi-tweet Twitter thread with no more than 15 tweets using feedback from an editor. Create a sense of curiosity and intrigue in my audience by introducing incomplete questions in the first tweet
    Draft: {draft}
    Edits: {edit}
    The final twitter thread:

    """
)

# CHAINS BEGIN HERE

writerChain = writer.createChain(
    llm=llm,
    promptTemplate=writer.setPromptTemplate(
        ["notes", "technique", "writerVoice"],
        template=writer.template
    ),
    output_key="draft"
)

editorChain = editor.createChain(
    llm=llm,
    promptTemplate=editor.setPromptTemplate(
        ["draft"],
        template=editor.template
    ),
    output_key="edit"
)

productionChain = production.createChain(
    llm=llm,
    promptTemplate=production.setPromptTemplate(
        ["draft", "edit"],
        template=production.template
    ),
    output_key="final"
)

# Running the chains
response = SequentialChain(
    memory=SimpleMemory(memories={"notes":markDownFile}),
    chains=[
        writerChain, editorChain, productionChain
    ],
    input_variables=["technique", "writerVoice"],
    output_variables=["final"],
    verbose=True
)

# Get the current date and time
now = datetime.datetime.now()

# Create a string with today's date in the format YYYY-MM-DD
date_string = now.strftime('%Y-%m-%d')

# Create a file with today's date in the name
filename = f'{date_string}.txt'

# Get the file name
md_file_name = os.path.basename(md_file_path)

# Writing to a file
with open(f'{filename}_{md_file_name}.txt', 'w') as f:
    f.write(
        response(
            {
                "technique":"persuasive",
                "writerVoice":"Mark Manson"
            })['final']
    )