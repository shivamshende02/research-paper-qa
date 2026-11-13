import json

from groq import Groq

from src.services.vector_ingestion import IngestionSettings

settings = IngestionSettings()

client = Groq(
    api_key=settings.GROQ_API_KEY.get_secret_value()
)

input_path = "data/processed/ee431476-669e-4980-9a8c-b52d82571e43.json"

with open(input_path, "r", encoding="utf-8") as file:
    document = json.load(file)

abstract = next(
    section for section in document["sections"]
    if "abstract" in section["header"].lower()
)

response = client.chat.completions.create(
    model=settings.GROQ_MODEL,
    messages=[
        {
            "role": "system",
            "content": (
                "Summarize this research paper section. "
                "Preserve technical terminology, methods, and findings. "
                "Do not invent information. Aim for 150-300 tokens."
            ),
        },
        {
            "role": "user",
            "content": abstract["content"],
        },
    ],
    temperature=0.1,
    max_tokens=350,
)

summary = response.choices[0].message.content

print("\nGROQ SUMMARY TEST SUCCESSFUL")
print("\nSummary:\n", summary)
print("\nModel:", response.model)
print("Prompt tokens:", response.usage.prompt_tokens)
print("Completion tokens:", response.usage.completion_tokens)