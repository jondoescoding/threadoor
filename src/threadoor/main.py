# Python 
import os
# Langchain
from langchain.llms import *
from langchain.chains import SequentialChain
from langchain.document_loaders import UnstructuredMarkdownLoader
from langchain.memory import SimpleMemory
# Custom Utilities
import helper as hp 

# ENVIRONMENT VARIABLES
OPENAI_TOKEN = os.environ.get('openAi')
HUGGINGFACE_TOKEN = os.environ.get('huggingFace')

# Gathering context
filePath = r"C:\_CODING\PYTHON\PROJECTS\contentCreator\docs\toBeIngested\Midjourney_Prompt_Length.md"
loader = UnstructuredMarkdownLoader(filePath)
markDownFile = loader.load()


# Setting up OpenAi
llm = OpenAI(openai_api_key=OPENAI_TOKEN, temperature=0.7, max_tokens=500)

# ROLES BEGIN HERE

# Writer -> 1st
writer = hp.Role(
    llm=llm,
    template="""
    Role: You are a professional copywriter. You have the ability to explain difficult topics down to their fundamentals using first principle thinking applying the Feynman Technique.
    Objective: For the given context rewrite a draft based on your role applying the specific style of writing.
    Author's Writing Style: {writingStyle}
    Context: {context}
    Draft:
    """
)

# Editor -> 2nd
editor = hp.Role(
    llm=llm,
    template="""
    Role: You are a professional Editor who was worked in  the online written industry for over 10 years. 
    Objective: Given a draft for a blog post you are to provide a bulleted list of feedback and advice based on the area of expertise
    Draft: {draft}
    Feedback:
    """
)

# SEO -> 3rd
seo = hp.Role(
    llm=llm,
    template="""
    Role: You are a professional SEO marketer who has been working for over 10 years.
    Objective: Given a draft for a blog post, you are to provide a bulleted list feedback and advice on how to improve the SEO
    Draft: {draft}
    SEO-Feedback: 
    """
)

production = hp.Role(
    llm=llm,
    template="""
    Role: You are a professional copywriter.
    Objective: Given a blog draft you are exptected to use the feedback from an editor and changes to seo keywords to construct a blog post.
    Draft: {draft}
    Edits: {edit}
    Seo: {seo}
    Final Blog:
    """
)

# CHAINS BEGIN HERE

writerChain = writer.createChain(
    llm=llm,
    promptTemplate=writer.setPromptTemplate(
        ["context", "writingStyle"],
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

seoChain = seo.createChain(
    llm=llm,
    promptTemplate=seo.setPromptTemplate(
        ["draft"],
        template=seo.template
    ),
    output_key="seo"
)

productionChain = production.createChain(
    llm=llm,
    promptTemplate=production.setPromptTemplate(
        ["draft", "edit", "seo"],
        template=production.template
    ),
    output_key="final"
)

# Running the chains
response = SequentialChain(
    memory=SimpleMemory(memories={"context":markDownFile}),
    chains=[
        writerChain, editorChain, seoChain, productionChain
    ],
    input_variables=["writingStyle"],
    output_variables=["final"],
    verbose=True
)

print(
    response({
        "writingStyle":"Mark Manson"
    })
)