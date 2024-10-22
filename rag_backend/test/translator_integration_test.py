import os
import sys
from dotenv import load_dotenv, find_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from rag_backend.query_translator import QueryTranslator as qt

# For local testing use env file, otherwise use environment variables in GitHub Actions
if not os.environ.get("OPENAI_API_KEY"):
    load_dotenv(find_dotenv(filename=".rag_engine.env"))

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

def test_qt_should_use_multi_query():
    translator = qt(OPENAI_API_KEY)
    assert translator.should_use_multi_query("How does the dopamine system differ between neurodivergent and non-neurodivergent individuals?")
    assert not translator.should_use_multi_query("What is the purpose of the hippocampus?")

def test_qt_multi_query_generation():
    translator = qt(OPENAI_API_KEY)
    queries = translator.multi_query_generation().invoke({"user_input": "How does the dopamine system differ between neurodivergent and non-neurodivergent individuals?"})
    assert len(queries) == 5