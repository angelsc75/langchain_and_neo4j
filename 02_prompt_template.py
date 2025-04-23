import os
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from template_prueba import template_sport

'''You can create a prompt from a string by calling
the PromptTemplate.from_template() static method 
or load a prompt from a file using the PromptTemplate.from_file() static method.
Esto se hace con template2 y el archivo template_prueba'''

llm = OpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"))


template = PromptTemplate(template="""
You are a cockney fruit and vegetable seller.
Your role is to assist your customer with their fruit and vegetable needs.
Respond using cockney rhyming slang.

Tell me about the following fruit: {fruit}
""", input_variables=["fruit"])

response = llm.invoke(template.format(fruit="apple"))
template2 = PromptTemplate(template=template_sport, input_variables=["deporte"])
response2 = llm.invoke(template2.format(deporte="hockey"))
print(response, response2)

