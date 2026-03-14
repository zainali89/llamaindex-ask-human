import os

from dotenv import load_dotenv
from llama_index.agent.openai import OpenAIAgent
from llama_index.llms.openai import OpenAI
from llama_index.core.tools.tool_spec.base import BaseToolSpec

import chainlit as cl
from chainlit.sync import run_sync

load_dotenv()


class HumanInputChainlit(BaseToolSpec):
    """Tool that adds the capability to ask a human user for input."""

    spec_functions = ["ask_human"]

    name = "human"
    description = (
        "You can ask a human for guidance when you think you "
        "got stuck or you are not sure what to do next. "
        "The input should be a question for the human."
    )

    def ask_human(self, query: str) -> str:
        """Ask the human user a question and return their response."""
        res = run_sync(cl.AskUserMessage(content=query).send())
        if res is None:
            return "The user did not respond."
        return res["output"]


@cl.on_chat_start
def start():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable is not set. "
            "Please create a .env file with your API key."
        )

    llm = OpenAI(model="gpt-4-turbo-preview", api_key=api_key)
    human_tool = HumanInputChainlit()
    agent = OpenAIAgent.from_tools(
        human_tool.to_tool_list(), llm=llm, verbose=True
    )
    cl.user_session.set("agent", agent)


@cl.on_message
async def main(message: cl.Message):
    agent = cl.user_session.get("agent")

    try:
        res = agent.chat(message.content)
        await cl.Message(content=str(res)).send()
    except Exception as e:
        await cl.Message(
            content=f"An error occurred while processing your request: {e}"
        ).send()
