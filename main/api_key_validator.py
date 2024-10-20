import openai
import anthropic

# Helper method to validate OpenAI API key
def _check_openai_api_key(api_key):
    client = openai.OpenAI(api_key=api_key)
    try:
        client.models.list()
    except openai.AuthenticationError:
        return False
    else:
        return True

# Helper method to validate Anthropic API key
def _check_anthropic_api_key(api_key):
    client = anthropic.Client(api_key=api_key)
    try:
        client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=1,
            messages=[
                {"role": "user", "content": "Test"}
                ]
            )
    except anthropic.AuthenticationError:
        return False
    else:
        return True

# Validate both API keys
def validate_api_keys(openai_key, anthropic_key):
    if openai_key is None or anthropic_key is None:
        return False
    try:
        # Validate OpenAI API key
        openai_valid = _check_openai_api_key(openai_key)
        if not openai_valid:
            return False
        # Validate Anthropic API key
        anthropic_valid = _check_anthropic_api_key(anthropic_key)
        if not anthropic_valid:
            return False

        return True
    except Exception as e:
        print(f"Error caught: {e}")
        return False