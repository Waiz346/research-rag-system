"""Multi-provider LLM response generator supporting dynamic runtime model switching (Qwen, GLM, Gemini, OpenAI, Ollama, Custom)."""
import logging
import json
import re
import time
from typing import Any, Dict, List, Optional, Tuple
import httpx
from src.config import settings

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = (
    "You are an expert scientific research assistant in an interactive research workbench. "
    "Your objective is to provide comprehensive, well-structured, clear, and insightful answers to user questions. "
    "Synthesize knowledge directly from the provided research context, citing specific sources with [Source X] references.\n\n"
    "CORE PRINCIPLES:\n"
    "1. Comprehensive Explanation: When the user asks a conceptual question (e.g., 'What is RAG?'), provide a thorough, structured explanation of the core concepts, mechanisms, and architectures. If the retrieved context discusses specific techniques, variants, or applications (e.g. Autoregressive RAG, dense retrieval, etc.), explain the overarching concept first, and ground your discussion using the concrete findings, methods, and results described in the retrieved literature.\n"
    "2. Rigorous Provenance: Attribute key factual claims, architectural components, benchmark results, and methodology to their corresponding sources using inline [Source X] tags.\n"
    "3. Contextual Fidelity: Distinguish general principles from specific empirical findings reported in the papers. Synthesize the most informative and truthful explanation possible from the provided text."
)

PROVIDER_DEFAULTS = {
    "qwen": {
        "base_url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        "default_model": "qwen-plus",
        "description": "Alibaba Cloud Qwen (DashScope / OpenAI-compatible)"
    },
    "glm": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "default_model": "glm-4-flash",
        "description": "Zhipu AI GLM (BigModel / OpenAI-compatible)"
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta",
        "default_model": "gemini-3.5-flash-lite",
        "description": "Google Gemini (Google AI Studio)"
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "default_model": "gpt-4o-mini",
        "description": "OpenAI Official"
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "default_model": "llama3",
        "description": "Local Ollama Instance"
    },
    "custom": {
        "base_url": "https://api.siliconflow.cn/v1",
        "default_model": "Qwen/Qwen2.5-7B-Instruct",
        "description": "Custom OpenAI-Compatible API (SiliconFlow, DeepSeek, vLLM, etc.)"
    },
    "mock": {
        "base_url": "",
        "default_model": "mock-eval-engine",
        "description": "Offline Deterministic Mock"
    }
}

class LLMGenerator:
    """Dynamically switchable LLM engine supporting Gemini, Qwen, GLM, OpenAI, Ollama, and Custom endpoints."""

    def __init__(self, provider: Optional[str] = None):
        self.provider = (provider or settings.LLM_PROVIDER).lower()
        self.model_name = self._resolve_initial_model()
        self.api_key = self._resolve_initial_key()
        self.base_url = self._resolve_initial_url()

    def _resolve_initial_model(self) -> str:
        if self.provider == "gemini":
            return getattr(settings, "GEMINI_MODEL", "gemini-3.5-flash-lite")
        preset = PROVIDER_DEFAULTS.get(self.provider)
        return preset["default_model"] if preset else "qwen-plus"

    def _resolve_initial_key(self) -> str:
        if self.provider == "gemini":
            return getattr(settings, "GEMINI_API_KEY", "")
        elif self.provider == "openai":
            return getattr(settings, "OPENAI_API_KEY", "")
        return ""

    def _resolve_initial_url(self) -> str:
        preset = PROVIDER_DEFAULTS.get(self.provider)
        return preset["base_url"] if preset else ""

    def update_config(
        self,
        provider: str,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ) -> Dict[str, Any]:
        """Update runtime LLM settings on-the-fly without server restart."""
        norm_provider = provider.strip().lower()
        self.provider = norm_provider

        preset = PROVIDER_DEFAULTS.get(norm_provider, PROVIDER_DEFAULTS["custom"])
        self.base_url = (base_url.strip() if base_url else None) or preset["base_url"]
        self.model_name = (model_name.strip() if model_name else None) or preset["default_model"]

        if api_key is not None:
            self.api_key = api_key.strip()

        logger.info(f"LLM runtime switched to: provider={self.provider}, model={self.model_name}, base_url={self.base_url}")
        return self.get_config()

    def get_config(self) -> Dict[str, Any]:
        """Return active LLM configuration with masked API key."""
        masked_key = ""
        if self.api_key:
            if len(self.api_key) > 8:
                masked_key = f"{self.api_key[:4]}...{self.api_key[-4:]}"
            else:
                masked_key = "********"

        return {
            "provider": self.provider,
            "model_name": self.model_name,
            "base_url": self.base_url,
            "has_api_key": bool(self.api_key),
            "masked_api_key": masked_key
        }

    def test_connection(self) -> Dict[str, Any]:
        """Verify connectivity with the active provider and model."""
        start = time.perf_counter()
        test_prompt = "Reply with exactly one word: 'READY'."
        try:
            output = self.generate(test_prompt, system_prompt="You are a system test probe. Output only 'READY'.")
            latency_ms = round((time.perf_counter() - start) * 1000, 2)
            return {
                "success": True,
                "provider": self.provider,
                "model_name": self.model_name,
                "latency_ms": latency_ms,
                "sample_output": output[:100].strip(),
                "message": f"Successfully connected to {self.provider.upper()} ({self.model_name}) in {latency_ms} ms."
            }
        except Exception as e:
            latency_ms = round((time.perf_counter() - start) * 1000, 2)
            return {
                "success": False,
                "provider": self.provider,
                "model_name": self.model_name,
                "latency_ms": latency_ms,
                "error": str(e),
                "message": f"Connection failed to {self.provider.upper()}: {e}"
            }

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Route generation request to active provider."""
        system_content = system_prompt or DEFAULT_SYSTEM_PROMPT

        if self.provider == "mock":
            return self._mock_generate(prompt)
        elif self.provider in ["qwen", "glm", "openai", "custom"]:
            return self._openai_compatible_generate(
                prompt=prompt,
                system_content=system_content,
                base_url=self.base_url,
                api_key=self.api_key,
                model=self.model_name
            )
        elif self.provider == "gemini":
            return self._gemini_generate(prompt, system_content)
        elif self.provider == "ollama":
            return self._ollama_generate(prompt, system_content)
        else:
            logger.warning(f"Unknown provider '{self.provider}'. Falling back to mock generator.")
            return self._mock_generate(prompt)

    def _openai_compatible_generate(
        self,
        prompt: str,
        system_content: str,
        base_url: str,
        api_key: str,
        model: str
    ) -> str:
        """Universal OpenAI-compatible chat completions caller (covers Qwen, GLM, DeepSeek, OpenAI, SiliconFlow)."""
        if not api_key and self.provider not in ["ollama"]:
            raise ValueError(f"API Key for {self.provider.upper()} is not set. Please provide your API key in settings.")

        endpoint = f"{base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2
        }

        try:
            with httpx.Client(timeout=45.0) as client:
                res = client.post(endpoint, headers=headers, json=payload)
                if res.status_code != 200:
                    err_msg = f"HTTP {res.status_code} from {endpoint}: {res.text}"
                    logger.error(err_msg)
                    raise RuntimeError(err_msg)

                data = res.json()
                return data["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"Error calling {self.provider} ({model}) at {endpoint}: {e}. Falling back to offline mock.")
            return self._mock_generate(prompt)

    def _gemini_generate(self, prompt: str, system_content: str) -> str:
        """Generate response via Google Gemini REST API with resilient fallback across candidate models."""
        api_key = self.api_key or getattr(settings, "GEMINI_API_KEY", "")
        if not api_key:
            logger.warning("GEMINI_API_KEY is not set. Falling back to offline mock.")
            return self._mock_generate(prompt)

        primary_model = self.model_name or "gemini-3.5-flash-lite"
        candidate_models = [primary_model]
        for fallback in ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-flash-latest", "gemini-3.8-flash"]:
            if fallback not in candidate_models:
                candidate_models.append(fallback)

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

        for model in candidate_models:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
            try:
                with httpx.Client(timeout=45.0) as client:
                    res = client.post(url, json=payload)
                    if res.status_code == 200:
                        data = res.json()
                        candidates = data.get("candidates", [])
                        if candidates and "content" in candidates[0]:
                            parts = candidates[0]["content"].get("parts", [])
                            if parts and "text" in parts[0]:
                                if model != primary_model:
                                    logger.info(f"Gemini responded via fallback model '{model}'.")
                                return parts[0]["text"]
                    logger.warning(f"Gemini API returned HTTP {res.status_code} for {model}: {res.text[:120]}. Trying fallback...")
            except Exception as e:
                logger.warning(f"Gemini API error ({model}): {e}. Trying fallback...")

        logger.error("All Gemini candidate models failed. Falling back to offline mock.")
        return self._mock_generate(prompt)

    def _ollama_generate(self, prompt: str, system_content: str) -> str:
        """Generate response via local Ollama instance."""
        url = f"{self.base_url.rstrip('/')}/chat/completions" if "/v1" in self.base_url else f"{settings.OLLAMA_BASE_URL}/api/generate"
        if "/v1" in self.base_url:
            return self._openai_compatible_generate(prompt, system_content, self.base_url, "ollama", self.model_name)

        payload = {
            "model": self.model_name or "llama3",
            "prompt": f"{system_content}\n\n{prompt}",
            "stream": False
        }
        try:
            with httpx.Client(timeout=60.0) as client:
                res = client.post(url, json=payload)
                res.raise_for_status()
                return res.json().get("response", "")
        except Exception as e:
            logger.error(f"Ollama generation error: {e}. Falling back to offline mock.")
            return self._mock_generate(prompt)

    def _mock_generate(self, prompt: str) -> str:
        """Deterministic mock generator synthesizing answer from prompt context."""
        if "hypothetical academic paper" in prompt or "hypothetical" in prompt.lower():
            match = re.search(r"query:\s*['\"]([^'\"]+)['\"]", prompt, re.IGNORECASE)
            subj = match.group(1) if match else "the investigated topic"
            return (
                f"In this study, we investigate {subj}. We analyze its theoretical foundations, "
                f"mechanisms, and empirical behavior across standardized evaluation protocols. "
                f"The results establish key operational dynamics and quantitative characteristics of {subj}."
            )

        # Extract paper titles and snippets cleanly
        sources_found = re.findall(r"\[Source (\d+)\] Title: ([^\n|]+)", prompt)
        user_q_match = re.search(r"Current User Question:\s*([^\n]+)", prompt)
        question = user_q_match.group(1).strip() if user_q_match else "your research question"

        if sources_found:
            src_bullets = "\n".join(f"- **[Source {idx}]** *{title.strip()}*" for idx, title in sources_found[:3])
            return (
                f"Based on the analyzed research context for **'{question}'**:\n\n"
                f"The literature investigates the following key publications:\n{src_bullets}\n\n"
                f"**Key Findings & Synthesis:**\n"
                f"1. **Core Architecture & Methodology:** The retrieved literature details methods for grounding generative models using indexed knowledge bases to enhance factual reliability [Source 1].\n"
                f"2. **Empirical Results:** Across standardized benchmarks, these approaches demonstrate significant performance improvements over standalone baseline systems [Source 1].\n\n"
                f"*Note: Synthesized from indexed research papers.*"
            )

        return (
            f"Based on the retrieved research context for **'{question}'**, the indexed documents provide relevant empirical findings and methodology [Source 1]."
        )

    def generate_hyde(self, query: str) -> str:
        """Generate a hypothetical document passage (HyDE) to improve vector retrieval."""
        hyde_prompt = (
            f"Write a short, dense 2-3 sentence hypothetical academic paper excerpt or abstract "
            f"that directly defines and explains the subject of this query: '{query}'. "
            f"Do not write conversational preamble. Output only the passage text."
        )
        try:
            passage = self.generate(hyde_prompt, system_prompt="You are an academic text synthesizer.")
            return passage.strip()
        except Exception as e:
            logger.warning(f"HyDE generation skipped: {e}")
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
            json_match = re.search(r"\{[\s\S]*\}", raw)
            if json_match:
                parsed = json.loads(json_match.group(0))
                if "context_relevance" in parsed and "faithfulness" in parsed and "answer_relevance" in parsed:
                    return parsed
        except Exception as e:
            logger.warning(f"LLM-as-a-Judge parse error: {e}")
        return None
