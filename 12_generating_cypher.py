'''Langchain includes the GraphCypherQAChainchain that can interact with a Neo4j graph database. 
It uses a language model to generate Cypher queries and then uses the graph to answer the question.

GraphCypherQAChain chain requires the following:

    An LLM (llm) for generating Cypher queries

    A graph database connection (graph) for answering the queries

    A prompt template (cypher_prompt) to give the LLM the schema and question

    An appropriate question which relates to the schema and data in the graph

The program below will generate a Cypher query based on the schema in the graph database and the question.


Allow Dangerous Requests:
You are trusting the generation of Cypher to the LLM. It may generate invalid Cypher 
queries that could corrupt data in the graph or provide access to sensitive information.
You have to opt-in to this risk by setting the allow_dangerous_requests flag to True.
In a production environment, you should ensure that access to data is limited,
and sufficient security is in place to prevent malicious queries. This could 
include the use of a read only user or role based access control.
'''
import os
from langchain_openai import ChatOpenAI
from langchain_neo4j import GraphCypherQAChain, Neo4jGraph
from langchain.prompts import PromptTemplate

llm = ChatOpenAI(
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD")
)

CYPHER_GENERATION_TEMPLATE = """
You are an expert Neo4j Developer translating user questions into Cypher to answer questions about movies and provide recommendations.
Convert the user's question based on the schema.

Schema: {schema}
Question: {question}
"""

cypher_generation_prompt = PromptTemplate(
    template=CYPHER_GENERATION_TEMPLATE,
    input_variables=["schema", "question"],
)

cypher_chain = GraphCypherQAChain.from_llm(
    llm,
    graph=graph,
    cypher_prompt=cypher_generation_prompt,
    verbose=True,
    allow_dangerous_requests=True
)

result = cypher_chain.invoke({"query": "What is the plot of the movie Toy Story?"})

print(result)