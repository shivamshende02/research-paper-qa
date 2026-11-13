
"""Standalone test for the Vaani retrieval pipeline."""

from src.services.retrieval import ResearchPaperRetriever


def main():
    retriever = ResearchPaperRetriever()

    print("\nVaani Research Paper Retrieval")
    print("-" * 45)

    while True:
        query = input("\nEnter your research question (or 'exit'): ").strip()

        if query.lower() == "exit":
            break

        if not query:
            print("Please enter a question.")
            continue

        try:
            result = retriever.retrieve(query)

            print("\nStatus:", result["status"])
            print("Reason:", result.get("reason", "N/A"))

            if result.get("top_score") is not None:
                print("Summary score:", round(result["top_score"], 4))

            print("\nAnswer:")
            print(result["answer"])

            if result["status"] == "success":
                print("\nRetrieval details:")
                print("Document UUID:", result["document_uuid"])
                print("Section:", result["section"])
                print("Page:", result["page"])
                print("Retrieved chunks:", result["retrieved_chunks"])
                print("Reranked chunks:", result["reranked_chunks"])

        except Exception as exc:
            print(f"\nRetrieval failed: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
