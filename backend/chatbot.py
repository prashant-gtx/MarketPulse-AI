import requests
import json
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ollama Configuration
OLLAMA_API_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3.2"

# System Prompt for Financial Assistant
SYSTEM_PROMPT = """You are MarketPulse AI, an expert financial analyst and assistant.
Your goal is to provide accurate, insightful, and concise answers to user questions about the Indian Stock Market (NSE/BSE), economy, and finance.

Guidelines:
1. Be professional yet accessible. Avoid overly complex jargon without explanation.
2. If asked about specific stocks, focus on the context provided in the question or recent news.
3. If you don't know the answer, admit it. Do not hallucinate financial data.
4. Keep answers concise (under 150 words) unless detailed analysis is requested.
5. Format your response in clean Markdown.

You likely have access to some context about news or market data passed in the user prompt. Use it effectively."""

def init_gemini():
    """No-op for compatibility, or check Ollama connection."""
    try:
        # Quick check if Ollama is reachable
        # Note: /api/tags or root might be better for health check
        logger.info(f"Checking Ollama connection at {OLLAMA_API_URL}...")
        requests.get("http://localhost:11434/", timeout=2) # Base URL check usually returns 200 OK 'Ollama is running'
        logger.info(f"Ollama appears to be running.")
    except Exception as e:
        logger.warning(f"Could not connect to Ollama at {OLLAMA_API_URL}. Ensure it is running. Error: {e}")

def get_chat_response(query: str, context: str = "") -> str:
    """
    Generates a response using the local Llama 3 model via Ollama.
    """
    try:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query}
        ]
        
        # If context is provided (e.g. from RAG in future), append it
        if context:
            messages.insert(1, {"role": "system", "content": f"Context: {context}"})

        payload = {
            "model": OLLAMA_MODEL,
            "messages": messages,
            "stream": False
        }
        
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=120)
        response.raise_for_status()
        
        data = response.json()
        return data.get("message", {}).get("content", "Error: No response from model.")

    except requests.exceptions.ConnectionError:
        return "Error: Could not connect to Ollama. Is the Ollama app running?"
    except requests.exceptions.ReadTimeout:
        return "Error: The model took too long to respond. Please try again or ask a shorter question."
    except Exception as e:
        logger.error(f"Error in chat generation: {e}")
        return f"I apologize, but I encountered an error processing your request. ({str(e)})"

# Legacy/Unused for now but kept if main.py imports it
def init_summarizer():
    pass

# Imports for Summarization
from scraper import scrape_article_content
from summarizer import summarize_with_t5
from sentiment import analyze_sentiment

def summarize_news(news_link: str):
    """
    Fetches article content, summarizes it using T5, and performs sentiment analysis.
    """
    try:
        # 1. Scrape Content
        content = scrape_article_content(news_link)
        if not content or len(content) < 100:
            return {"summary": "Could not extract sufficient content from this article to summarize.", "sentiment": "neutral"}
            
        # 2. Summarize
        # Note: summarize_with_t5 handles model loading if needed
        summary = summarize_with_t5(content)
        
        # 3. Sentiment Analysis
        sentiment_result = analyze_sentiment(content[:1000]) # Analyze start of content
        sentiment = sentiment_result.get('label', 'neutral')
        
        return {"summary": summary, "sentiment": sentiment}
        
    except Exception as e:
        logger.error(f"Error in summarize_news: {e}")
        return {"summary": f"Failed to generate summary: {str(e)}", "sentiment": "neutral"}
