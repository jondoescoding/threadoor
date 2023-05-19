# Python 
import os
# Langchain
from langchain.llms import *
from langchain.chains import SequentialChain
# Custom Utilities
import helper as hp 

# ENVIRONMENT VARIABLES
OPENAI_TOKEN = os.environ.get('openAi')
HUGGINGFACE_TOKEN = os.environ.get('huggingFace')

# Setting up LLMS

# OpenAi
llm = OpenAI(openai_api_key=OPENAI_TOKEN, temperature=0.7, max_tokens=500)

# ROLES BEGIN HERE -> Writer, Editor, SEO, Writer, Prompt Engineer (Optional) 

# Writer -> 1st
writer = hp.Role(
    llm=llm,
    template="""
    Role: You are a professional freelance copy writer. You have the ability to explain difficult topics down to their fundamentals using first principle thinking applying the Feynman Technique. Your tone is similar to the author Mark Manson.
    Objective: Given an topic for a blog post you are to write a first draft of the target topic.
    Topic: {topic}
    Expertise: {expertise}
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
        ["topic", "expertise"],
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
chain = SequentialChain(
    chains=[
        writerChain, editorChain, seoChain, productionChain
    ],
    input_variables=["topic", "expertise"],
    output_variables=["final"]
)

print(
    chain({
        "topic":"What are algorithms",
        "expertise":"Computer Science"
    })
)