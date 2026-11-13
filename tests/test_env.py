import os
from dotenv import load_dotenv

load_dotenv()

def test_environment_configuration():
    assert os.getenv("GROQ_API_KEY")
    assert os.getenv("LOCAL_ENDPOINT_URL")
    assert os.getenv("EMBEDDING_MODEL")

    print("Environment configuration verified.")