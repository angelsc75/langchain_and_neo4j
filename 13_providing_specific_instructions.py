'''The LLM’s training data included many Cypher statements, but these statements were not specific
to the structure of your graph database. As a result, the LLM may generate Cypher 
statements that are not valid and do not conform to the schema.

You can provide specific instructions to the LLM to state that the generated Cypher 
statements should follow the schema.

For example, you could give instructions only to use the provided relationship types 
and properties in the schema.

As well as instructing the LLM on how to deal with the question, you can also instruct the 
LLM on how to respond.

You could instruct the LLM only to respond when the Cypher statement returns data.

If no data is returned, do not attempt to answer the question.

You may want the LLM to only respond to questions in the scope of the task. 
For example:

Only respond to questions that require you to construct a Cypher statement.

Do not respond to any questions that might ask anything else than for you to 
construct a Cypher statement.

Concise responses from the LLM may be needed:

Do not include any explanations or apologies in your responses.

Or you may want to restrict the format of the response:

Do not include any text except the generated Cypher statement.

Ultimately, you must fine-tune your instructions for the specific task to 
ensure the best results.
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

Instructions:
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.
For movie titles that begin with "The", move "the" to the end, For example "The 39 Steps" becomes "39 Steps, The" or "The Matrix" becomes "Matrix, The".

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

result = cypher_chain.invoke({"query": "Who acted in The Matrix and what roles did they play?"})

print(result)
