import os
from openai import OpenAI
from anthropic import Anthropic
from huggingface_hub import InferenceApi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def query_models(query):
    """Query multiple models and return their responses."""
    results = {}
    
    # GPT Models
    results["GPT-4o"] = query_gpt_4o(query)
    results["GPT-o1"] = query_gpt_o1(query)
    results["GPT-4o-mini"] = query_gpt_4o_mini(query)
    results["GPT-o3-mini"] = query_gpt_o3_mini(query)
    results["GPT-o1-mini"] = query_gpt_o1_mini(query)
    
    # Deepseek Models
    results["Deepseek R1"] = query_deepseek_r1(query)
    results["Deepseek V3"] = query_deepseek_v3(query)
    
    # Qwen Model
    results["Qwen 2.5"] = query_qwen_2_5(query)
    
    # Claude Models
    results["Claude 3 Sonnet"] = query_claude_3_sonnet(query)
    results["Claude 3 Opus"] = query_claude_3_opus(query)
    
    # Llama Model
    results["Llama 3.3"] = query_llama_3_3(query)
    
    # Gemini Models
    results["Gemini 1.5 Pro"] = query_gemini_1_5_pro(query)
    results["Gemini 2.0 Flash"] = query_gemini_2_0_flash(query)
    
    # Grok Model
    results["Grok-1"] = query_grok_1(query)
    
    return results

def query_with_file(query, file_content):
    """Query models with file content."""
    results = {}
    
    # GPT Models
    results["GPT-4o"] = query_gpt_4o(f"{query}\n\nContext:\n{file_content}")
    results["GPT-o1"] = query_gpt_o1(f"{query}\n\nContext:\n{file_content}")
    results["GPT-4o-mini"] = query_gpt_4o_mini(f"{query}\n\nContext:\n{file_content}")
    results["GPT-o3-mini"] = query_gpt_o3_mini(f"{query}\n\nContext:\n{file_content}")
    results["GPT-o1-mini"] = query_gpt_o1_mini(f"{query}\n\nContext:\n{file_content}")
    
    # Deepseek Models
    results["Deepseek R1"] = query_deepseek_r1(f"{query}\n\nContext:\n{file_content}")
    results["Deepseek V3"] = query_deepseek_v3(f"{query}\n\nContext:\n{file_content}")
    
    # Qwen Model
    results["Qwen 2.5"] = query_qwen_2_5(f"{query}\n\nContext:\n{file_content}")
    
    # Claude Models
    results["Claude 3 Sonnet"] = query_claude_3_sonnet(f"{query}\n\nContext:\n{file_content}")
    results["Claude 3 Opus"] = query_claude_3_opus(f"{query}\n\nContext:\n{file_content}")
    
    # Llama Model
    results["Llama 3.3"] = query_llama_3_3(f"{query}\n\nContext:\n{file_content}")
    
    # Gemini Models
    results["Gemini 1.5 Pro"] = query_gemini_1_5_pro(f"{query}\n\nContext:\n{file_content}")
    results["Gemini 2.0 Flash"] = query_gemini_2_0_flash(f"{query}\n\nContext:\n{file_content}")
    
    # Grok Model
    results["Grok-1"] = query_grok_1(f"{query}\n\nContext:\n{file_content}")
    
    return results

# Individual model query functions
def query_gpt_4o(query):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": query}]
    )
    return {
        "answer": response.choices[0].message.content,
        "score": calculate_relevance_score(response.choices[0].message.content, query)
    }

def query_gpt_o1(query):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-o1",
        messages=[{"role": "user", "content": query}]
    )
    return {
        "answer": response.choices[0].message.content,
        "score": calculate_relevance_score(response.choices[0].message.content, query)
    }

def query_gpt_4o_mini(query):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": query}]
    )
    return {
        "answer": response.choices[0].message.content,
        "score": calculate_relevance_score(response.choices[0].message.content, query)
    }

def query_gpt_o3_mini(query):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-o3-mini",
        messages=[{"role": "user", "content": query}]
    )
    return {
        "answer": response.choices[0].message.content,
        "score": calculate_relevance_score(response.choices[0].message.content, query)
    }

def query_gpt_o1_mini(query):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-o1-mini",
        messages=[{"role": "user", "content": query}]
    )
    return {
        "answer": response.choices[0].message.content,
        "score": calculate_relevance_score(response.choices[0].message.content, query)
    }

def query_deepseek_r1(query):
    client = InferenceApi(repo_id="deepseek-ai/deepseek-r1", token=os.getenv("DEEPSEEK_API_KEY"))
    response = client(inputs=query)
    return {
        "answer": response[0]["generated_text"],
        "score": calculate_relevance_score(response[0]["generated_text"], query)
    }

def query_deepseek_v3(query):
    client = InferenceApi(repo_id="deepseek-ai/deepseek-v3", token=os.getenv("DEEPSEEK_API_KEY"))
    response = client(inputs=query)
    return {
        "answer": response[0]["generated_text"],
        "score": calculate_relevance_score(response[0]["generated_text"], query)
    }

def query_qwen_2_5(query):
    client = InferenceApi(repo_id="qwen/qwen-2.5", token=os.getenv("QWEN_API_KEY"))
    response = client(inputs=query)
    return {
        "answer": response[0]["generated_text"],
        "score": calculate_relevance_score(response[0]["generated_text"], query)
    }

def query_claude_3_sonnet(query):
    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    response = client.completions.create(
        model="claude-3-sonnet",
        prompt=query
    )
    return {
        "answer": response.completion,
        "score": calculate_relevance_score(response.completion, query)
    }

def query_claude_3_opus(query):
    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    response = client.completions.create(
        model="claude-3-opus",
        prompt=query
    )
    return {
        "answer": response.completion,
        "score": calculate_relevance_score(response.completion, query)
    }

def query_llama_3_3(query):
    client = InferenceApi(repo_id="meta-llama/Llama-3.3", token=os.getenv("LLAMA_API_KEY"))
    response = client(inputs=query)
    return {
        "answer": response[0]["generated_text"],
        "score": calculate_relevance_score(response[0]["generated_text"], query)
    }

def query_gemini_1_5_pro(query):
    client = InferenceApi(repo_id="google/gemini-1.5-pro", token=os.getenv("GEMINI_API_KEY"))
    response = client(inputs=query)
    return {
        "answer": response[0]["generated_text"],
        "score": calculate_relevance_score(response[0]["generated_text"], query)
    }

def query_gemini_2_0_flash(query):
    client = InferenceApi(repo_id="google/gemini-2.0-flash", token=os.getenv("GEMINI_API_KEY"))
    response = client(inputs=query)
    return {
        "answer": response[0]["generated_text"],
        "score": calculate_relevance_score(response[0]["generated_text"], query)
    }

def query_grok_1(query):
    client = InferenceApi(repo_id="xai/grok-1", token=os.getenv("GROK_API_KEY"))
    response = client(inputs=query)
    return {
        "answer": response[0]["generated_text"],
        "score": calculate_relevance_score(response[0]["generated_text"], query)
    }

def calculate_relevance_score(response, query):
    """Calculate a relevance score using cosine similarity."""
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([response, query])
    similarity_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return round(similarity_score * 100, 2)  # Convert to percentage