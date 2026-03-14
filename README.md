# LlamaIndex Ask-Human Agent

An AI agent built with [LlamaIndex](https://www.llamaindex.ai/) and [Chainlit](https://chainlit.io/) that implements a **human-in-the-loop** pattern. When the agent is unsure or needs clarification, it asks the user for guidance directly in the chat before continuing -- giving you control over the agent's decision-making process.

## Demo

A demo video is included in the repository (`ask-human.mp4`).

## Tech Stack

- **LlamaIndex** -- agent framework and OpenAI tool integration
- **Chainlit** -- chat UI with built-in support for user input prompts
- **OpenAI GPT-4 Turbo** -- underlying LLM

## Setup

### Prerequisites

- Python 3.10+
- An [OpenAI API key](https://platform.openai.com/api-keys)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/zainali89/llamaindex-ask-human.git
   cd llamaindex-ask-human
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file from the example and add your OpenAI API key:
   ```bash
   cp .env.example .env
   ```
   Then edit `.env` and replace the placeholder with your actual key.

## Running

Start the Chainlit app:

```bash
chainlit run app_llamaindex.py -w
```

The `-w` flag enables auto-reload on file changes. Open the URL shown in the terminal (typically `http://localhost:8000`) to interact with the agent.

## How It Works

The agent uses a custom `HumanInputChainlit` tool. When the LLM decides it needs human input, it invokes this tool, which surfaces a prompt in the Chainlit UI. The user's response is fed back to the agent so it can continue reasoning with that new information.

## License

MIT
