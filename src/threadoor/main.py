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
llm = OpenAI(openai_api_key=OPENAI_TOKEN, temperature=0.7, max_tokens=1000)

# ROLES BEGIN HERE

"""
Threadoor 
- Objective -> Take the notes I create from a file (markdown, txt, etc.) and convert them into a twitter thread.
"""

threadoor = hp.Role(
    llm=OpenAI,
    template="""
    Role Description: You are a content writing agent that summarizes the main ideas for the given context in a Twitter thread format.
    Goal: Summarize the main points and ideas of the given markdown notes in a Twitter thread format capturing the {noteStructure} of the notes using no hashtags. As if explaining the concept to a 10 year old, break down the notes into smaller and digestible tweets using only the information provided in the markdown notes. The thread's content should be formatted as such:
        - Provide a simple explanation of the concept as an introduction to what will be discussed by writing clear and concise tweets for each point without altering the original meaning.
        - Highlight the key concepts or the headings from the notes within each tweet.
        - Illustrate the concept to an example in real life.
        - Ensure a logical flow and coherence in the Twitter thread while maintaining the essence of the notes.
    Markdown Notes: {mdNotes}
    Twitter thread:

    """
)

"""
Hookoor
- Objective: Using the written Twitter thread to generate 3 possible hook tweets to capture the attention of a potential reader
"""

hookoor = hp.Role(
    llm=llm,
    template="""
    Role Description: You are a content writing agent that uses the main ideas from a given Twitter thread to generate a headliner.
    The 5 elements of writing an effective headline are: Be CLEAR, not Clever, Be CLEAR, not Clever, Specify the WHAT, Specify the WHY, throw a curve ball. 
    The 6 proven ways to write an engaging headline: Open with 1 strong, declarative sentence, Open with a thought-provoking question, Open with a controversial opinion, Open with a moment in time, Open with a vulnerable statement
    Goal: Take all five elements of effective headline writing and one of the six proven ways to write an engaging headline and create three headlines that reflect the Twitter thread provided
    Twitter Thread: {thread}
    Headlines: 
    """
)

promptoor = hp.Role(
    llm=llm,
    template="""
    Role: You are an agent who writes descriptive short text phrases which will used to generate images.
    Format: (image we're prompting), (5 descriptive keywords or phrases), (art style), (artist name), (art medium)
    Goal: Create a 60 word short text phrase depicting a furturistic scene based on the theme of a given Twitter Thread using the given Format. Here are some examples short text phrases:
    Example 1: a commercial photograph of a burger with dripping hot sauce, natural light, hyper-detailed,
    105 mm macro, shutter speed at 1/800 sec, ISO 6400, f/8 aperture, 8k, octane rendering
    Example 2: a hyper-realistic image of kids happily playing on a wide green field, manual camera mode, ISO 400, shutter speed at 1/200, aperture f/4, sunny
    Example 3: a full-body shot of a fashion model walking on a runway wearing modern streetwear, fierce face, shot on a mirrorless camera, continuous focus mode
    Twitter Thread: {thread}
    Generated text phrase: 
    """
)

# CHAINS
threadoorChain = threadoor.createChain(
    llm=llm,
    promptTemplate=threadoor.setPromptTemplate(
        inputVariables=["noteStructure", "mdNotes"],
        template=threadoor.template
    ),
    output_key="thread"
)

hookoorChain = hookoor.createChain(
    llm=llm,
    promptTemplate=hookoor.setPromptTemplate(
        inputVariables=["thread"],
        template=hookoor.template
    ),
    output_key="hook"
)

promptoorChain = promptoor.createChain(
    llm=llm,
    promptTemplate=promptoor.setPromptTemplate(
        inputVariables=["thread"],
        template=promptoor.template
    ),
    output_key="prompt"
)

chain = SequentialChain(
    memory=SimpleMemory(memories={"mdNotes":markDownFile}),
    chains=[
        threadoorChain,
        hookoorChain,
        promptoorChain
    ],
    input_variables=["noteStructure"],
    output_variables=["thread", "hook", "prompt"],
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
    chain({"noteStructure":"tone, voice, vocabulary, and sentence structure"}).items())[-3:]

# Open a new file for writing
with open('output.txt', 'w') as f:
    # Write the last three key-value pairs to the file, one per line
    for key, value in repsonse:
        # Write the key with '# Key:' prefix and newline character
        f.write(f"# Key: {key}\n")
        # Write the value with '# Value' prefix and newline character
        f.write(f"# Value:\n {value} \n")



