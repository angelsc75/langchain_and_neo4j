'''The Neo4jVector class has a as_retriever() method that returns a retriever.

The RetrievalQA class is a chain that uses a retriever as part of its pipeline. It will use the retriever to retrieve documents and pass them to a language model.

By incorporating Neo4jVector into a RetrievalQA chain, you can use data and vectors in Neo4j in a Langchain application.

When the program runs, the RetrievalQA chain will use the movie_plot_vector retriever to retrieve documents from the moviePlots index and pass them to the chat_llm language model.

By setting the optional verbose and return_source_documents arguments to True when creating the RetrievalQA chain, you can see the source documents and the retriever’s score for each document.
'''

import os
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_neo4j import Neo4jGraph, Neo4jVector

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY
)

embedding_provider = OpenAIEmbeddings(
    openai_api_key=OPENAI_API_KEY
)

graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD")
)

movie_plot_vector = Neo4jVector.from_existing_index(
    embedding_provider,
    graph=graph,
    index_name="moviePlots",
    embedding_node_property="plotEmbedding",
    text_node_property="plot",
)

plot_retriever = RetrievalQA.from_llm(
    llm=llm,
    retriever=movie_plot_vector.as_retriever(),
    # verbose=True,
    # return_source_documents=True
)

response = plot_retriever.invoke(
    {"query": "A movie where a mission to the moon goes wrong"}
)

print(response)