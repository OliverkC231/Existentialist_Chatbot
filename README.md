**Existentialist Chatbot Comparison**

This project compares three different chatbot configurations—each trained or augmented with the writings of major existentialist philosophers—to observe how Retrieval-Augmented Generation (RAG) affects the style, depth, and consistency of LLM responses.

**Models Compared**

The system evaluates three chatbot setups:
Ollama LLM (no RAG)
A baseline chatbot running a local Ollama model without any external knowledge retrieval.
Ollama LLM + RAG
The same Ollama model, but enhanced with a Pinecone vector store and RAG pipeline to ground responses in curated existentialist texts.
OpenAI GPT-5.1 + RAG
A cloud-based chatbot using OpenAI’s GPT-5.1 model with a RAG layer for deeper philosophical grounding and richer text interpretation.

**Goal of the Project**
The aim is to analyze how each model behaves when exposed to existentialist source material:
Do their tones become more “existentialist”?
Do their interpretations become more accurate?
How does retrieval influence style vs. reasoning?
Does GPT-5.1 handle ambiguity differently than local LLMs?
This enables controlled qualitative comparisons of model behavior with and without RAG.

**RUNNING THE SCRIPT**

1. Install a venv
2. Install the requirements
3. Set up API keys
4. Run ingestion files to store vectos in Pinecone
5. Have fun experimenting with the chatbots!

