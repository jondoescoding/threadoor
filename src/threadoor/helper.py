# Python 
import os
from dataclasses import dataclass
from typing import List
# LangChain
from langchain import LLMChain, llms
from langchain import PromptTemplate

@dataclass
class Role:
    """Wrapper for generating chains
    """
    llm: llms
    template: str

    def setPromptTemplate(cls, inputVariables:List[str], template:str):
        """Setting the promptTemplate

        Args:
            inputVariables (list): list of variable which will be used by the prompt template
            template (str): the long ass template

        Returns:
            _type_: a prompt template object to be used by a chain
        """
        return PromptTemplate(input_variables=inputVariables, template=template)
    
    def createChain(cls, llm:llms, promptTemplate:str, output_key:str):
        """Creates a LLMChain

        Args:
            llm (llms): whatever LLM being used to generate results
            promptTemplate (str): the template being used
            output_key (str): the name of the output model

        Returns:
            _type_: An LLMChain
        """
        return LLMChain(llm=llm, prompt=promptTemplate, output_key=output_key)