"""
LLM Client — Gemini 2.0 Flash via LangChain
"""
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="langchain_google_genai")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain_google_genai")

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage

from src.config import GEMINI_TEMPERATURE, GEMINI_MAX_TOKENS, GOOGLE_API_KEY
from src.utils.logger import get_logger
from src.utils.cache import make_cache_key, get_cached, set_cached
from src.utils.retry import safe_llm_call

logger = get_logger(__name__)

# Confirmed working model for this API key
GEMINI_CHAT_MODEL = "gemini-2.0-flash"

FALLBACK_RESPONSE = (
    "The AI service is temporarily unavailable. "
    "Please verify your GOOGLE_API_KEY and retry."
)


def get_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=GEMINI_CHAT_MODEL,
        temperature=GEMINI_TEMPERATURE,
        max_output_tokens=GEMINI_MAX_TOKENS,
        google_api_key=GOOGLE_API_KEY,
    )


def call_llm(
    system_prompt: str,
    user_prompt: str,
    cache_prefix: str = "llm",
    use_cache: bool = True,
) -> str:
    cache_key = make_cache_key(cache_prefix, system_prompt, user_prompt)

    if use_cache:
        cached = get_cached(cache_key)
        if cached:
            logger.debug(f"LLM Cache HIT: {cache_key}")
            return cached

    def _invoke():
        llm = get_llm()
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
        response = llm.invoke(messages)
        return response.content

    result = safe_llm_call(_invoke, fallback=FALLBACK_RESPONSE)

    if use_cache and result != FALLBACK_RESPONSE:
        set_cached(cache_key, result)

    return result