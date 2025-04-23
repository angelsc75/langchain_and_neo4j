'''Agents:
Agents wrap a model and give it access to a set of tools. These tools may access additional
data sources, APIs, or functionality. The model is used to determine which of the tools to
use to complete a task.
The agent you will create will be able to chat about movies and search YouTube for movie trailers.
Tools:
A tool is a specific abstraction around a function that makes it easy for a language model
to interact with it. Langchain provides several tools out of the box, and you can create
tools to extend the functionality of your agents.

There are different types of agents that you can create. This example creates a 
ReAct - Reasoning and Acting) agent type.

An agent requires a prompt. You could create a prompt, but in this example, 
the program pulls a pre-existing prompt from the Langsmith Hub.

The hwcase17/react-chat prompt instructs the model to provide an answer using 
the tools available in a specific format.

The create_react_agent function creates the agent and expects the following parameters:

    The llm that will manage the interactions and decide which tool to use

    The tools that the agent can use

    The prompt that the agent will use

The AgentExecutor class runs the agent. It expects the following parameters:

    The agent to run

    The tools that the agent can use

    The memory which will store the conversation history

AgentExecutor parameters

You may find the following additional parameters useful when initializing an agent:

    max_iterations - the maximum number of iterations to run the LLM for. This is useful in preventing the LLM from running for too long or entering an infinite loop.

    verbose - if True the agent will print out the LLM output and the tool output.

    handle_parsing_errors - if True the agent will handle parsing errors and return a message to the user.


'''

import os
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain import hub
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.schema import StrOutputParser
from langchain_community.tools import YouTubeSearchTool
from langchain_neo4j import Neo4jChatMessageHistory, Neo4jGraph
from uuid import uuid4

SESSION_ID = str(uuid4())
print(f"Session ID: {SESSION_ID}")

llm = ChatOpenAI(
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD")
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a movie expert. You find movies from a genre or plot.",
        ),
        ("human", "{input}"),
    ]
)

movie_chat = prompt | llm | StrOutputParser()

youtube = YouTubeSearchTool()

def get_memory(session_id):
    return Neo4jChatMessageHistory(session_id=session_id, graph=graph)

def call_trailer_search(input):
    input = input.replace(",", " ")
    return youtube.run(input)

tools = [
    Tool.from_function(
        name="Movie Chat",
        description="For when you need to chat about movies. The question will be a string. Return a string.",
        func=movie_chat.invoke,
    ),
    Tool.from_function(
        name="Movie Trailer Search",
        description="Use when needing to find a movie trailer. The question will include the word trailer. Return a link to a YouTube video.",
        func=call_trailer_search,
    ),
]

agent_prompt = hub.pull("hwchase17/react-chat")
agent = create_react_agent(llm, tools, agent_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)

chat_agent = RunnableWithMessageHistory(
    agent_executor,
    get_memory,
    input_messages_key="input",
    history_messages_key="chat_history",
)

while (q := input("> ")) != "exit":
    
    response = chat_agent.invoke(
        {
            "input": q
        },
        {"configurable": {"session_id": SESSION_ID}},
    )
    
    print(response["output"])