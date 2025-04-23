import os
from langchain_openai import OpenAI

llm = OpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"))

""" llm = OpenAI(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-3.5-turbo-instruct",
    temperature=0
) """
response = llm.invoke("What is Neo4j?")

print(response)