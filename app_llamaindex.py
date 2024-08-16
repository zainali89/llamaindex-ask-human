from llama_index.core.agent import ReActAgent
from llama_index.agent.openai import OpenAIAgent
from llama_index.llms.openai import OpenAI
from llama_index.core.tools.tool_spec.base import BaseToolSpec

import chainlit as cl
from chainlit.sync import run_sync

class HumanInputChainlit(BaseToolSpec):
    spec_functions = ["_run", "_arun"]
    """Tool that adds the capability to ask user for input."""

    name = "human"
    description = (
        "You can ask a human for guidance when you think you "
        "got stuck or you are not sure what to do next. "
        "The input should be a question for the human."
    )

    def _run(self, query: str, run_manager=None) -> str:
        """Use the Human input tool."""
        res = run_sync(cl.AskUserMessage(content=query).send())
        return res["content"]

    async def _arun(self, query: str, run_manager=None) -> str:
        """Use the Human input tool."""
        res = await cl.AskUserMessage(content=query).send()
        return res["output"]

@cl.on_chat_start
def start():
    llm = OpenAI(model="gpt-4-turbo-preview", api_key="OPENAI_API_KEY")
    agent = OpenAIAgent.from_tools(HumanInputChainlit().to_tool_list(), llm=llm, verbose=True)
    cl.user_session.set("agent", agent)
    print("Agent initialized.")  # Debug output

@cl.on_message
async def main(message: cl.Message):
    agent = cl.user_session.get("agent")
    
    try:
        res = agent.chat(message.content)
        print(f"Agent response: {res}") 
        await cl.Message(content=res).send()
    except Exception as e:
        print(f"Error: {e}")
        await cl.Message(content="An error occurred while processing your request.").send()

