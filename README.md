# 📚 Autonomous arXiv Research Assistant

> A multi-agent AI system built with **Microsoft AutoGen**, **OpenAI**, and **Streamlit** that automates academic literature discovery, token-aware context management, and structured paper synthesis from arXiv.

---

### ⚡ Key Capabilities
* **Multi-Agent Orchestration:** Uses `autogen-agentchat` and `autogen-core` to coordinate specialized agents for query formulation, paper retrieval, and executive summarization.
* **Live arXiv Integration:** Queries the arXiv API in real time to fetch relevant preprints, metadata, abstracts, and publication links.
* **Token-Optimized Synthesis:** Leverages `tiktoken` to manage prompt budgets cleanly across long abstracts and multi-paper comparisons.
* **Interactive Streamlit UI:** Provides a clean web interface for researchers and analysts to run natural-language literature queries.

---

### 🏗️ Agent Architecture

```mermaid
flowchart LR
    A[User Research Query] --> B[Streamlit Interface]
    B --> C[AutoGen Orchestrator]
    C --> D[arXiv Retrieval Agent]
    D -->|Raw Metadata & Abstracts| E[Tiktoken Context Manager]
    E --> F[OpenAI Synthesis Agent]
    F --> G[Structured Research Brief]

---
🛠️ Tech Stack
Agent Framework: autogen-agentchat, autogen-core, autogen-ext

LLM & Tokenization: openai, tiktoken

Data Source: arxiv

Frontend: streamlit

🚀 Quickstart

Agent Framework: autogen-agentchat, autogen-core, autogen-ext

LLM & Tokenization: openai, tiktoken

Data Source: arxiv

Frontend: streamlit

# 1. Clone the repository
git clone [https://github.com/oscartandi/arxiv_paper_finder.git](https://github.com/oscartandi/arxiv_paper_finder.git)
cd arxiv_paper_finder

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your OpenAI API key
export OPENAI_API_KEY="your-api-key-here"

# 4. Launch the Streamlit app
streamlit run app.py
