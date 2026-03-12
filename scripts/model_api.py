# model_api.py

from ner_annotator import API_MODELS

DEFAULT_MODEL_NAME = "DeepSeek-R1"
DEFAULT_LANG_FACTOR = "fr"
DEFAULT_SYSTEM_PROMPT = ""   # <= make it an empty string, NOT None

def call_model(prompt: str, text: str) -> str:
    # If your base prompt contains {text}, use replace; otherwise append the input.
    if "{text}" in prompt:
        user_prompt = prompt.replace("{text}", text)
    else:
        user_prompt = f"{prompt}\n\nHere is the input text:\n{text}"

    api_model = API_MODELS[DEFAULT_MODEL_NAME]
    content, _token_probs = api_model["function"](
        api_model=api_model,
        lang_factor=DEFAULT_LANG_FACTOR,
        user_prompt=user_prompt,
        system_prompt=DEFAULT_SYSTEM_PROMPT or ""   # <= ensure string
    )
    return content
