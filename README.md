# Multi_modal_agent
AI Q&A Agent with Image Understanding

This project integrates Large Language Models (LLMs) and Visual-Language Models (VLMs) to create an intelligent agent capable of answering text-based and image-based questions. It uses Groq’s LLaMA-3.3-70B model for natural language reasoning and BLIP (Bootstrapped Language Image Pretraining) for visual captioning. The system can understand user queries, interpret uploaded images, and generate meaningful responses—all through an interactive Gradio interface.

Features

Text-based Q&A: Ask general knowledge or technical questions.

Image-based Q&A: Upload an image and ask related questions (e.g., “Describe the chart” or “What is happening in this image?”).

Smart Workflow Routing: Dynamically routes the question to the right node (text-only or image-Q&A).

Modular LangGraph Workflow: Uses StateGraph to define the flow of states and logic.

LLM Integration: Utilizes ChatGroq with the LLaMA family of models for accurate text generation.

Visual Understanding: Employs Salesforce/BLIP to convert visual data into descriptive text.

Interactive Gradio UI: Simple and intuitive interface to interact with the agent.

Tech Stack
Component	Purpose
Python 3.10+	Programming language
LangGraph	State management and workflow definition
LangChain Groq	LLaMA-3.3-70B language model access
Transformers (Hugging Face)	BLIP image captioning
PIL (Pillow)	Image processing
Torch	Deep learning inference
Gradio	Frontend interface for user interaction
dotenv	API key management
How It Works

Text Question:
When the user asks a question without uploading an image, the query is routed to the text_only node.
The Groq LLaMA model processes it and returns a detailed response.

Image Question:
When an image is uploaded, the BLIP model generates a caption.
This caption is combined with the user’s question and sent to LLaMA for contextual reasoning.

Workflow:
The logic is handled by a LangGraph StateGraph, which routes between nodes and merges results.
