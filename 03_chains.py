import os
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.output_parsers.json import SimpleJsonOutputParser

'''Chains allows you to combine language models with different data sources and third-party APIs.
LCEL

The simplest chain combines a prompt template with an LLM and returns a response.

You can create a chain using LangChain Expression Language (LCEL). LCEL is a declarative way to chain Langchain components together.

Components are chained together using the | operator.  chain = prompt | llm 
The output from the chain is typically a string, 
and you can specify an output parser to parse the output.
llm_chain = template | llm | StrOutputParser()'''

llm = OpenAI(
    openai_api_key=os.getenv("OPENAI_API_KEY")
    )

template1 = PromptTemplate.from_template("""
You are a cockney fruit and vegetable seller.
Your role is to assist your customer with their fruit and vegetable needs.
Respond using cockney rhyming slang.

Tell me about the following fruit: {fruit}
""")

llm_chain1 = template1 | llm
llm_chain2 = template1 | llm | StrOutputParser()
response1 = llm_chain1.invoke({"fruit": "apple"})
response2 = llm_chain2.invoke({"fruit": "apple"})

template2 = PromptTemplate.from_template("""
You are a cockney fruit and vegetable seller.
Your role is to assist your customer with their fruit and vegetable needs.
Respond using cockney rhyming slang.

Output JSON as {{"description": "your response here"}}

Tell me about the following fruit: {fruit}
""")
llm_chain3 = template2 | llm | SimpleJsonOutputParser()
response3 = llm_chain3.invoke({"fruit": "apple"})
print(response1, response2, response3)