import streamlit as st
import pandas as pd
# import os

from langchain.agents import ZeroShotAgent, Tool, AgentExecutor, initialize_agent, create_pandas_dataframe_agent
from langchain.memory import ConversationBufferMemory, ReadOnlySharedMemory, StreamlitChatMessageHistory
from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
from langchain.chains import RetrievalQA, RetrievalQAWithSourcesChain
from langchain.chat_models import ChatOpenAI

from langchain.vectorstores import Chroma
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings


st.set_page_config(page_title="Streamlit Langchain with Tools", page_icon="🛠️")
st.title("📖 StreamlitChatMessageHistory")



# Get an OpenAI API Key before continuing
if "OPENAI_API_KEY" in st.secrets:
    openai_api_key = st.secrets["OPENAI_API_KEY"]
else:
    openai_api_key = st.sidebar.text_input("OpenAI API Key")
if not openai_api_key:
    st.info("Enter an OpenAI API Key to continue")
    st.stop()

#tool definition and set up
templates = """This is a conversation between a human and a bot:

{chat_history}

Write a summary of the conversation for {input}:
"""

#set up memory
msgs = StreamlitChatMessageHistory(key="langchain_messages")
memory = ConversationBufferMemory(memory_key="chat_history",chat_memory=msgs, return_messages=False)

# Load vectorstore
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2") #MUST align with what the data is stored
vector_db = Chroma(persist_directory='./chroma_db', embedding_function=embeddings)

##########################################################
#This is where we BUILD tools
##########################################################

#Search
duckduck_search = DuckDuckGoSearchAPIWrapper()

#Summary Chain
prompt = PromptTemplate(input_variables=["input", "chat_history"], template=templates)
readonlymemory = ReadOnlySharedMemory(memory=memory)
summary_chain = LLMChain(
    llm=OpenAI(openai_api_key=openai_api_key),
    prompt=prompt,
    verbose=True,
    memory=readonlymemory,  # use the read-only memory to prevent the tool from modifying the memory
)

#Campaign Data
cdata = create_pandas_dataframe_agent(OpenAI(openai_api_key=openai_api_key,temperature=0), pd.read_csv('data/campaign_date.csv'), verbose=True)

#Email data

qa_template = """
        You are a helpful AI assistant who is an expert email communicator. The user gives you a file its content is represented by the following pieces of context, use them to answer the question at the end.
        If you don't know the answer, just say you don't know. Do NOT try to make up an answer.
        Use as much detail as possible when responding.
    
        {context}
        
        """

qa_prompt = PromptTemplate(input_variables=["context"], template=qa_template)
chain_type_kwargs = {"prompt": qa_prompt}
memail = RetrievalQA.from_chain_type(llm=OpenAI(openai_api_key=openai_api_key), chain_type_kwargs=chain_type_kwargs, chain_type="stuff", retriever=vector_db.as_retriever()) 




##########################################################
#This is where we DEFINE tools
##########################################################
tools = [
    Tool(
        name="Search",
        func=duckduck_search.run,
        description="this tool should be used as a last resort, to find basic answers to questions about facts outside your training history such as current events. interpret the response and ensure appropriateness before replying to the user, else regenerate the response",
    ),
    Tool(
        name="Summary",
        func=summary_chain.run,
        description="useful for when you summarize a conversation. The input to this tool should be a string, representing who will read this summary.",
    ),
    Tool(
        name = "Campaign data",
        func=cdata.run,
        description="Campaign data performance into a data frame for analysis and comparison, useful when you need to answer questions about campaign performance and dates, date conatins the dates the campaign ran but may have multiple campaigns per day, default best metric is Cost per Action (CPA) and a lower number is better the date column is the day the campaign was active)"
    ),
    Tool(
        name = "Mercury email data",
        func=memail.run,
        description="useful for when you need to answer questions around the marketing emails that a power company called Mercury has sent to it's customers"
    ),
    # Tool(
    #     name = "Custom agent tool",
    #     func=cagent.run,
    #     description="useful for when you need to answer like a pirate"
    # ),
]

##########################################################
# MAIN AGENT CHAIN
##########################################################

prefix = """You are a human consultant. You help users achieve their tasks and answer their questions. Have a conversation with a human, answering the following questions as best you can. You have access to the following tools, before using a tool consider answering the question yourself:"""
suffix = """Begin!"

{chat_history}
Question: {input}
{agent_scratchpad}"""


prompt = ZeroShotAgent.create_prompt(
    tools,
    prefix=prefix,
    suffix=suffix,
    input_variables=["input", "chat_history", "agent_scratchpad"],
)

llm_chain = LLMChain(llm=OpenAI(openai_api_key=openai_api_key,temperature=0), prompt=prompt)
agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True, handle_parsing_errors="Check your output and make sure it conforms!")
agent_chain = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=True, memory=memory
)



if len(msgs.messages) == 0:
    msgs.add_ai_message("How can I help you?")

view_messages = st.expander("View the message contents in session state")

# Render current messages from StreamlitChatMessageHistory
for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)

# If user inputs a new prompt, generate and draw a new response
if prompt := st.chat_input():
    st.chat_message("human").write(prompt)
    # Note: new messages are saved to history automatically by Langchain during run
    response = agent_chain.run(prompt)
    st.chat_message("ai").write(response)

# Draw the messages at the end, so newly generated ones show up immediately
with view_messages:
    """
    View the messages in session state: 
    """
    view_messages.json(st.session_state.langchain_messages)