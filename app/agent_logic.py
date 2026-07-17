from langchain_ollama import ChatOllama
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import create_react_agent, AgentExecutor
from langchain.chains import LLMChain
from langchain_core.runnables.history import RunnableWithMessageHistory
from .prompts import react_prompt, offline_prompt
from .utils import has_internet

def run_agent(task, chat_history):
    """Run the smart agent, switching between online and offline modes."""
    llm = ChatOllama(model="gemma3:4b")

    # ---------------------------
    # Build chat history string 
    # ---------------------------
    history_str = ""
    for msg in chat_history.messages:
        if msg.type == "human":
            history_str += f"User: {msg.content}\n"
        else:
            history_str += f"Assistant: {msg.content}\n"

    if has_internet():
        tools = load_tools(["wikipedia", "ddg-search"])
        
        # Create ReAct agent 
        agent = create_react_agent(llm, tools, react_prompt)

        # AgentExecutor with parsing errors handled 
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=False,
            handle_parsing_errors=True
        )

        chain = RunnableWithMessageHistory(
            agent_executor,
            lambda session_id: chat_history,
            input_messages_key="input",
            history_messages_key="chat_history",
        )

        response = chain.invoke(
            {"input": task, "chat_history": history_str},  # pass chat history
            config={"configurable": {"session_id": "chat"}},
        )
        return response.get("output", response.get("text", "No response.")), True

    else:
        # Offline chain with chat memory 
        offline_chain = LLMChain(llm=llm, prompt=offline_prompt)

        chain = RunnableWithMessageHistory(
            offline_chain,
            lambda session_id: chat_history,
            input_messages_key="input",
            history_messages_key="chat_history",
        )

        response = chain.invoke(
            {"input": task, "chat_history": history_str},  # pass chat history
            config={"configurable": {"session_id": "chat"}},
        )
        return response.get("text", "No response."), False
