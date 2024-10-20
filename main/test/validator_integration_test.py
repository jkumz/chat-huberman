from api_key_validator import validate_api_keys
import os
from dotenv import load_dotenv, find_dotenv

# For local testing use env file, otherwise use environment variables in GitHub Actions
if not os.environ.get("OPENAI_API_KEY") and not os.environ.get("ANTHROPIC_API_KEY"):
    load_dotenv(find_dotenv(filename=".test.env"))

# Import methods to unit test - not integrated
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

"""Test case for validate_api_keys"""
def test_validate_api_keys_normal():
    assert validate_api_keys(OPENAI_API_KEY, ANTHROPIC_API_KEY)

"""Test case for incorrect API keys entered"""
def test_validate_api_keys_incorrect():
    assert not validate_api_keys("incorrect_openai_key", ANTHROPIC_API_KEY)
    assert not validate_api_keys(OPENAI_API_KEY, "incorrect_anthropic_key")

"""Test case for invalid API keys entered"""
def test_invalid_api_keys():
    assert not validate_api_keys("invalid_openai_key", "invalid_anthropic_key")

"""Test case for missing API keys"""
def test_missing_api_keys():
    assert not validate_api_keys("", "")
    assert not validate_api_keys(OPENAI_API_KEY, "")
    assert not validate_api_keys("", ANTHROPIC_API_KEY)

"""Test case for None API keys (no input)"""
def test_none_api_keys():
    assert not validate_api_keys(None, None)
    assert not validate_api_keys(OPENAI_API_KEY, None)
    assert not validate_api_keys(None, ANTHROPIC_API_KEY)