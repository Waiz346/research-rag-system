"""Multi-provider LLM response generator with grounding, HyDE, and LLM-as-a-Judge."""
import logging
import json
import re
from typing import Any, Dict, List, Optional
import httpx
from src.config import settings

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = (
    "You are an expert scientific research assistant. "
    "Answer the user question based strictly on the provided research context. "
    "Cite sources accurately using [Source X] references.\n\n"
    "CRITICAL INTENT RULES:\n"
    "1. If the user asks a general or conceptual question (e.g. 'What is RAG?'), "
    "but the provided context ONLY discusses a specific specialized variant (e.g. 'AR-RAG for image patches'), "
    "you MUST explicitly clarify this distinction upfront: state clearly that the retrieved papers only cover "
    "a specific sub-variant, describe that variant accurately, and clarify that the foundational definition is absent.\n"
    "2. Never present a narrow application or specialized sub-technique as the overarching definition.\n"
    "3. If the context does not contain enough information to address the query, state that clearly."
)

class LLMGenerator:
    """Interacts with configured LLM providers (Gemini, OpenAI, Ollama, Mock)."""

    def __init__(self, provider: Optional[str] = None):
        self.provider = (provider or settings.LLM_PROVIDER).lower()

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate text from LLM using selected provider."""
        system_content = system_prompt or DEFAULT_SYSTEM_PROMPT

        if self.provider == "mock":
            return self._mock_generate(prompt)
        elif self.provider == "openai":
            return self._openai_generate(prompt, system_content)
        elif self.provider == "gemini":
            return self._gemini_generate(prompt, system_content)
        elif self.provider == "ollama":
            return self._ollama_generate(prompt, system_content)
        else:
            logger.warning(f"Unknown provider '{self.provider}'. Falling back to mock generator.")
            return self._mock_generate(prompt)

    def generate_hyde(self, query: str) -> str:
        """Generate a hypothetical document passage (HyDE) to improve vector retrieval."""
        hyde_prompt = (
            f"Write a short, dense 2-3 sentence hypothetical academic paper excerpt or abstract "
            f"that directly defines and explains the subject of this query: '{query}'. "
            f"Do not write conversational preamble. Output only the passage text."
        )
        try:
            passage = self.generate(hyde_prompt, system_prompt="You are an academic text synthesizer.")
            logger.info(f"Generated HyDE passage: {passage[:120]}...")
            return passage.strip()
        except Exception as e:
            logger.warning(f"Failed to generate HyDE passage: {e}")
            return query

    def judge_evaluation(self, query: str, answer: str, context: str) -> Optional[Dict[str, Any]]:
        """LLM-as-a-Judge: Evaluate Context Relevance, Faithfulness, and Answer Relevance semantically."""
        judge_prompt = f"""You are an expert AI evaluator assessing a Retrieval-Augmented Generation (RAG) system.
Analyze the user query, the retrieved context from research papers, and the generated answer.

User Query: "{query}"

Retrieved Context:
\"\"\"{context}\"\"\"

Generated Answer:
\"\"\"{answer}\"\"\"

Evaluate the response on three metrics (Scores must be between 0.0 and 1.0):
1. context_relevance: Does the retrieved context actually contain information that answers the intent of the query? (Score low if the query asks for a general concept but context only contains a specialized sub-variant).
2. faithfulness: Is the generated answer factually grounded in the provided context without hallucinations?
3. answer_relevance: Does the generated answer directly address the specific intent and scope of the user query? (Score low if it defines a narrow niche variant when a general definition was requested).

Respond ONLY with valid JSON in this exact structure:
{{
  "context_relevance": {{
    "score": 0.0,
    "rationale": "one sentence explanation"
  }},
  "faithfulness": {{
    "score": 0.0,
    "rationale": "one sentence explanation"
  }},
  "answer_relevance": {{
    "score": 0.0,
    "rationale": "one sentence explanation"
  }}
}}"""

        try:
            raw = self.generate(judge_prompt, system_prompt="You are an impartial academic benchmark judge. Output pure JSON.")
            # Extract JSON block
            json_match = re.search(r"\{[\s\S]*\}", raw)
            if json_match:
                parsed = json.loads(json_match.group(0))
                if "context_relevance" in parsed and "faithfulness" in parsed and "answer_relevance" in parsed:
                    return parsed
        except Exception as e:
            logger.warning(f"LLM-as-a-Judge parse error: {e}")
        return None

    def _mock_generate(self, prompt: str) -> str:
        """Deterministic mock generator synthesizing answer from prompt context."""
        logger.info("Using Mock LLM Generator (Offline mode).")
        lines = prompt.splitlines()
        context_lines = [l for l in lines if "Context" in l or "Source" in l or "Title:" in l]
        
        extracted_info = "\n".join(f"- {line.strip()}" for line in context_lines[:4])
        return (
            f"Based on the retrieved research context, here are the key findings:\n\n"
            f"{extracted_info if extracted_info else '- Key methodology and results from retrieved papers.'}\n\n"
            f"Summary: The analyzed research indicates significant advancements in this domain, "
            f"demonstrating improved performance and practical applicability [Source 1]."
        )

    def _openai_generate(self, prompt: str, system_content: str) -> str:
        """Generate response via OpenAI API."""
        try:
            from openai import OpenAI
            client = OpenAI(api_key=settings.OPENAI_API_KEY)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"OpenAI generation error: {e}. Falling back to mock response.")
            return self._mock_generate(prompt)

    def _gemini_generate(self, prompt: str, system_content: str) -> str:
        """Generate response via Google Gemini REST API."""
        api_key = settings.GEMINI_API_KEY
        if not api_key:
            logger.warning("GEMINI_API_KEY not set. Using mock generator.")
            return self._mock_generate(prompt)

        model = settings.GEMINI_MODEL or "gemini-1.5-pro"
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        payload = {
            "system_instruction": {
                "parts": [{"text": system_content}]
            },
            "contents": [
                {"role": "user", "parts": [{"text": prompt}]}
            ],
            "generationConfig": {
                "temperature": 0.2
            }
        }
        try:
            with httpx.Client(timeout=45.0) as client:
                res = client.post(url, json=payload)
                res.raise_for_status()
                data = res.json()
                return data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as e:
            logger.error(f"Gemini API error ({model}): {e}. Falling back to mock response.")
            return self._mock_generate(prompt)

    def _ollama_generate(self, prompt: str, system_content: str) -> str:
        """Generate response via local Ollama instance."""
        url = f"{settings.OLLAMA_BASE_URL}/api/generate"
        payload = {
            "model": "llama3",
            "prompt": f"{system_content}\n\n{prompt}",
            "stream": False
        }
        try:
            with httpx.Client(timeout=60.0) as client:
                res = client.post(url, json=payload)
                res.raise_for_status()
                return res.json().get("response", "")
        except Exception as e:
            logger.error(f"Ollama generation error: {e}. Falling back to mock response.")
            return self._mock_generate(prompt)
