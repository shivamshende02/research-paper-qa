from src.results_logger import (
    initialize_results_csv,
    ResultsLogger,
)

CSV_PATH = "outputs/results.csv"

# Initialize CSV
initialize_results_csv(CSV_PATH)

# Sample evaluation result
result = {
    "request_id": "req_001",
    "scenario": "steady",
    "source": "real",
    "start_time": "2026-11-11T19:30:00",
    "first_token_ms": 450.5,
    "total_ms": 1250.8,
    "status": "ok",
    "retries": 0,
    "tokens_in": 1200,
    "tokens_out": 250,
    "retrieval_setting": "500_tokens_top5_rerank",
    "answer_grade": "correct",
}

# Append result
ResultsLogger.log_result(CSV_PATH, result)

print("Result logged successfully.")