# 🚀 LLM Benchmarking and Comparison Tool 🌟

Welcome to the **LLM Benchmarking and Comparison Tool**, a powerful application designed to compare the performance of multiple state-of-the-art language models (LLMs) on complex queries or uploaded documents/images. This tool helps you evaluate models like **GPT-4o**, **Claude 3**, **Deepseek**, **Qwen**, **Llama**, **Gemini**, and more! 🤖✨

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [File Structure](#file-structure)
3. [Setup Instructions](#setup-instructions)
   - [Step 1: Create a Virtual Environment](#step-1-create-a-virtual-environment)
   - [Step 2: Install Dependencies](#step-2-install-dependencies)
   - [Step 3: Gather API Keys](#step-3-gather-api-keys)
   - [Step 4: Add API Keys to `.env`](#step-4-add-api-keys-to-env)
   - [Step 5: Run the Application](#step-5-run-the-application)
4. [How It Works](#how-it-works)
5. [Contributing](#contributing)

---

## 🌟 Project Overview

This application allows users to:
- Enter a query or upload a document/image.
- Compare responses from multiple LLMs (e.g., GPT-4o, Claude 3, Deepseek, Qwen, etc.).
- Evaluate models based on relevance scores calculated using **cosine similarity**.
- Generate complex queries using an open-source model from Hugging Face to test edge cases.

The app is built using **Streamlit** for the frontend and integrates APIs from various LLM providers. It also supports file uploads for analyzing documents/images.

---

## 📂 File Structure

Here’s the structure of the project and what each file does:

```
llm-comparison/
├── app.py                     # Main Streamlit app
├── requirements.txt           # List of Python dependencies
├── .env                       # Environment variables (API keys)
├── utils/                     # Utility functions
│   ├── model_utils.py         # Functions for interacting with LLMs
│   └── file_utils.py          # Functions for handling file uploads
└── assets/                    # Icons for models
    ├── gpt-4o.png
    ├── deepseek-r1.png
    ├── qwen-2-5.png
    └── ... (other model icons)
```

---

## 🔧 Setup Instructions

Follow these steps to set up and run the project locally:

---

### **Step 1: Create a Virtual Environment** 🌱

Create a virtual environment to isolate dependencies:

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

---

### **Step 2: Install Dependencies** 🛠️

Install all required dependencies using `pip`:

```bash
pip install -r requirements.txt
```

---

### **Step 3: Gather API Keys** 🔑

To use the LLMs, you’ll need API keys from their respective providers. Follow the links below to get your API keys:

- **OpenAI (GPT Models)**: [https://platform.openai.com/](https://platform.openai.com/)
- **Anthropic (Claude Models)**: [https://www.anthropic.com/](https://www.anthropic.com/)
- **Deepseek**: [https://www.deepseek.com/](https://www.deepseek.com/)
- **Qwen**: [https://www.aliyun.com/](https://www.aliyun.com/)
- **Llama**: [https://llama.meta.com/](https://llama.meta.com/)
- **Gemini**: [https://deepmind.com/](https://deepmind.com/)
- **Grok**: [https://x.ai/](https://x.ai/)
- **Hugging Face**: [https://huggingface.co/](https://huggingface.co/)

---

### **Step 4: Add API Keys to `.env`** 📝

Create a `.env` file in the root directory and add your API keys:

```plaintext
OPENAI_API_KEY=<your-openai-key>
ANTHROPIC_API_KEY=<your-anthropic-key>
DEEPSEEK_API_KEY=<your-deepseek-key>
QWEN_API_KEY=<your-qwen-key>
LLAMA_API_KEY=<your-llama-key>
GEMINI_API_KEY=<your-gemini-key>
GROK_API_KEY=<your-grok-key>
HUGGINGFACE_API_KEY=<your-huggingface-key>
```

⚠️ **Important**: Do not commit this file to version control!

---

### **Step 5: Run the Application** 🏃‍♂️

Start the Streamlit app:

```bash
streamlit run app.py
```

Open your browser and navigate to `http://localhost:8501`. You’ll see the interactive frontend where you can:
- Enter a query or upload a file.
- Compare responses from different LLMs.
- View benchmark scores and model comparisons.

---

## 🎯 How It Works

1. **User Interaction**:
   - The user enters a query or uploads a document/image via the frontend.
   - Optionally, the user can generate a complex query using the Hugging Face model.

2. **Backend Processing**:
   - The query or file content is sent to multiple LLMs for processing.
   - Responses are collected and scored using cosine similarity.

3. **Frontend Feedback**:
   - Results are displayed with icons, answers, and relevance scores.
   - Animations enhance the user experience.

---

## 🤝 Contributing

We welcome contributions! Here’s how you can help:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Commit your changes (`git commit -m "Add awesome feature"`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a pull request.

Let’s make this project even better together! 🌟

---

## 🙏 Acknowledgments

Special thanks to:
- **OpenAI**, **Anthropic**, **Deepseek**, **Qwen**, **Llama**, **Gemini**, and **Grok** for their amazing models.
- **Hugging Face** for providing open-source tools and models.
- **You** for checking out this project! 🚀

---

Happy coding! 🎉
