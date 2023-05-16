import os
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains import SequentialChain


# Environment Variables
openAiToken = os.environ.get('openAi')

# Using OpenAi as a Sequential Chain
llm = OpenAI(openai_api_key=openAiToken, temperature=0.7, max_tokens=250)

template = """
You are a expert writer with over 10 years of experience writing content online. Your tone is sarcastic. Given the subject being discussed and the ideal customer persona it is your objectiveto write a single Twitter Thread starter tweet to engage with potential customer for a topic within a specific niche to get them to read the Twitter thread

Subject: {subject}
Ideal Customer Persona: {customer}
Engaging Tweet: This is the Twitter Thread Starter Tweet for the above criteria: 

"""

promptTemplate = PromptTemplate(input_variables=["subject","customer"], template=template)

twitterThreadStarter_chain = LLMChain(llm=llm, prompt=promptTemplate, output_key="threadStarter")


templateMidJ = """
I want you to act as a prompt generator for Midjourney’s artificial intelligence program. Your job is to provide detailed and creative descriptions that will inspire unique and interesting images from the AI based on the overarcing theme of the tweet. Keep in mind that the AI is capable of understanding a wide range of language and can interpret abstract concepts, so feel free to be as imaginative and descriptive as possible. For example, you could describe a scene from a futuristic city, or a surreal landscape filled with strange creatures. The more detailed and imaginative your description, the more interesting the resulting image will be. Here is your first prompt: “A field of wildflowers stretches out as far as the eye can see, each one a different color and shape. In the distance, a massive tree towers over the landscape, its branches reaching up to the sky like tentacles.”

Twitter Thread Starter:
{threadStarter}

Midjourney prompt based on the thread above:
"""

promptTemplate_midj = PromptTemplate(input_variables=["threadStarter"], template=templateMidJ)
midj_chain = LLMChain(llm=llm, prompt=promptTemplate_midj, output_key="midj")

overallChain = SequentialChain(
    chains=[twitterThreadStarter_chain, midj_chain],
    input_variables=["subject", "customer"],
    output_variables=["threadStarter", "midj"]
)

response = overallChain(
    {"subject":"What is a LLM?", "customer":"Adult male early 20. Interested in tech and has some background in it from his education. Casually browses r/Python on reddit"
    }
)

print(response)