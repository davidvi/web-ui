import logging
import requests

logger = logging.getLogger(__name__)

# Cache for OpenRouter models
_openrouter_models_cache = None

PROVIDER_DISPLAY_NAMES = {
    "openai": "OpenAI",
    "azure_openai": "Azure OpenAI",
    "anthropic": "Anthropic",
    "deepseek": "DeepSeek",
    "google": "Google",
    "alibaba": "Alibaba",
    "moonshot": "MoonShot",
    "unbound": "Unbound AI",
    "ibm": "IBM",
    "grok": "Grok",
    "openrouter": "OpenRouter",
}

# Predefined model names for common providers
model_names = {
    "anthropic": ["claude-3-5-sonnet-20241022", "claude-3-5-sonnet-20240620", "claude-3-opus-20240229"],
    "openai": ["gpt-4o", "gpt-4", "gpt-3.5-turbo", "o3-mini"],
    "deepseek": ["deepseek-chat", "deepseek-reasoner"],
    "google": ["gemini-2.0-flash", "gemini-2.0-flash-thinking-exp", "gemini-1.5-flash-latest",
               "gemini-1.5-flash-8b-latest", "gemini-2.0-flash-thinking-exp-01-21", "gemini-2.0-pro-exp-02-05",
               "gemini-2.5-pro-preview-03-25", "gemini-2.5-flash-preview-04-17"],
    "ollama": ["qwen2.5:7b", "qwen2.5:14b", "qwen2.5:32b", "qwen2.5-coder:14b", "qwen2.5-coder:32b", "llama2:7b",
               "deepseek-r1:14b", "deepseek-r1:32b"],
    "azure_openai": ["gpt-4o", "gpt-4", "gpt-3.5-turbo"],
    "mistral": ["pixtral-large-latest", "mistral-large-latest", "mistral-small-latest", "ministral-8b-latest"],
    "alibaba": ["qwen-plus", "qwen-max", "qwen-vl-max", "qwen-vl-plus", "qwen-turbo", "qwen-long"],
    "moonshot": ["moonshot-v1-32k-vision-preview", "moonshot-v1-8k-vision-preview"],
    "unbound": ["gemini-2.0-flash", "gpt-4o-mini", "gpt-4o", "gpt-4.5-preview"],
    "grok": [
        "grok-3",
        "grok-3-fast",
        "grok-3-mini",
        "grok-3-mini-fast",
        "grok-2-vision",
        "grok-2-image",
        "grok-2",
    ],
    "siliconflow": [
        "deepseek-ai/DeepSeek-R1",
        "deepseek-ai/DeepSeek-V3",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "deepseek-ai/DeepSeek-V2.5",
        "deepseek-ai/deepseek-vl2",
        "Qwen/Qwen2.5-72B-Instruct-128K",
        "Qwen/Qwen2.5-72B-Instruct",
        "Qwen/Qwen2.5-32B-Instruct",
        "Qwen/Qwen2.5-14B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen/Qwen2.5-Coder-32B-Instruct",
        "Qwen/Qwen2.5-Coder-7B-Instruct",
        "Qwen/Qwen2-7B-Instruct",
        "Qwen/Qwen2-1.5B-Instruct",
        "Qwen/QwQ-32B-Preview",
        "Qwen/Qwen2-VL-72B-Instruct",
        "Qwen/Qwen2.5-VL-32B-Instruct",
        "Qwen/Qwen2.5-VL-72B-Instruct",
        "TeleAI/TeleChat2",
        "THUDM/glm-4-9b-chat",
        "Vendor-A/Qwen/Qwen2.5-72B-Instruct",
        "internlm/internlm2_5-7b-chat",
        "internlm/internlm2_5-20b-chat",
        "Pro/Qwen/Qwen2.5-7B-Instruct",
        "Pro/Qwen/Qwen2-7B-Instruct",
        "Pro/Qwen/Qwen2-1.5B-Instruct",
        "Pro/THUDM/chatglm3-6b",
        "Pro/THUDM/glm-4-9b-chat",
    ],
    "ibm": ["ibm/granite-vision-3.1-2b-preview", "meta-llama/llama-4-maverick-17b-128e-instruct-fp8",
            "meta-llama/llama-3-2-90b-vision-instruct"],
    "modelscope":[
        "Qwen/Qwen2.5-Coder-32B-Instruct",
        "Qwen/Qwen2.5-Coder-14B-Instruct",
        "Qwen/Qwen2.5-Coder-7B-Instruct",
        "Qwen/Qwen2.5-72B-Instruct",
        "Qwen/Qwen2.5-32B-Instruct",
        "Qwen/Qwen2.5-14B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen/QwQ-32B-Preview",
        "Qwen/Qwen2.5-VL-3B-Instruct",
        "Qwen/Qwen2.5-VL-7B-Instruct",
        "Qwen/Qwen2.5-VL-32B-Instruct",
        "Qwen/Qwen2.5-VL-72B-Instruct",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "deepseek-ai/DeepSeek-R1",
        "deepseek-ai/DeepSeek-V3",
        "Qwen/Qwen3-1.7B",
        "Qwen/Qwen3-4B",
        "Qwen/Qwen3-8B",
        "Qwen/Qwen3-14B",
        "Qwen/Qwen3-30B-A3B",
        "Qwen/Qwen3-32B",
        "Qwen/Qwen3-235B-A22B",
    ],
    "openrouter": [],
}


def fetch_openrouter_models():
    """
    Fetch available models from OpenRouter API.
    Returns a list of model IDs, with caching to avoid repeated API calls.
    """
    global _openrouter_models_cache
    
    # Return cached models if available
    if _openrouter_models_cache is not None:
        return _openrouter_models_cache
    
    try:
        response = requests.get("https://openrouter.ai/api/v1/models", timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Extract model IDs from the response
        # Response format: {"data": [{"id": "model-id", ...}, ...]}
        if "data" in data and isinstance(data["data"], list):
            models = [model["id"] for model in data["data"] if "id" in model]
            # Sort models for better UX
            models.sort()
            _openrouter_models_cache = models
            logger.info(f"Successfully fetched {len(models)} models from OpenRouter")
            return models
        else:
            logger.warning("OpenRouter API response format unexpected")
            return []
    except requests.exceptions.RequestException as e:
        logger.warning(f"Failed to fetch OpenRouter models: {e}")
        return []
    except Exception as e:
        logger.warning(f"Unexpected error fetching OpenRouter models: {e}")
        return []
