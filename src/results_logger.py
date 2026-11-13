
"""Thread-safe CSV results logging for LLM evaluation."""

import csv
import logging
import os
import threading
from typing import Any


logger = logging.getLogger(__name__)

CSV_HEADERS = [
    "request_id",
    "scenario",
    "source",
    "start_time",
    "first_token_ms",
    "total_ms",
    "status",
    "retries",
    "tokens_in",
    "tokens_out",
    "retrieval_setting",
    "answer_grade",
]

# Shared lock protects CSV operations across threads in this process.
_FILE_LOCK = threading.RLock()


def initialize_results_csv(file_path: str) -> None:
    """
    Initialize the results CSV if it does not already exist.

    Args:
        file_path: Destination CSV file path.

    Creates parent directories when necessary.
    Existing files are preserved without modification.

    Raises:
        OSError: If the directory or file cannot be created.
    """
    if not file_path:
        raise ValueError("file_path cannot be empty.")

    try:
        directory = os.path.dirname(os.path.abspath(file_path))

        with _FILE_LOCK:
            os.makedirs(directory, exist_ok=True)

            if not os.path.exists(file_path):
                with open(
                    file_path,
                    mode="x",
                    newline="",
                    encoding="utf-8",
                ) as csv_file:
                    writer = csv.writer(csv_file)
                    writer.writerow(CSV_HEADERS)

                logger.info("Initialized results CSV: %s", file_path)

    except OSError:
        logger.exception("Failed to initialize CSV: %s", file_path)
        raise


class ResultsLogger:
    """Persist LLM evaluation results in a structured CSV file."""

    @staticmethod
    def validate_data(data: dict[str, Any]) -> None:
        """
        Validate that dictionary keys exactly match CSV_HEADERS.

        Expected values:
            request_id, scenario, source, status,
            retrieval_setting, answer_grade: str
            start_time: str (timestamp)
            first_token_ms, total_ms: int | float | None
            retries, tokens_in, tokens_out: int

        None is also accepted for unavailable measurements.

        Raises:
            TypeError: If data is not a dictionary.
            ValueError: If keys are missing or unexpected.
        """
        if not isinstance(data, dict):
            raise TypeError("data must be a dictionary.")

        actual_keys = set(data.keys())
        expected_keys = set(CSV_HEADERS)

        missing = expected_keys - actual_keys
        extra = actual_keys - expected_keys

        if missing or extra:
            errors = []

            if missing:
                errors.append(f"Missing keys: {sorted(missing)}")

            if extra:
                errors.append(f"Unexpected keys: {sorted(extra)}")

            raise ValueError(
                "Invalid CSV result schema. " + "; ".join(errors)
            )

    @staticmethod
    def log_result(file_path: str, data: dict[str, Any]) -> None:
        """
        Append one evaluation result to the CSV file.

        Args:
            file_path: Destination CSV file.
            data: Result dictionary containing exactly CSV_HEADERS.

        Values are serialized by csv.writer. Numeric measurements
        should be supplied as numbers, and timestamps as strings.

        Raises:
            ValueError: If dictionary keys do not match the schema.
            OSError: If the file cannot be accessed or written.
        """
        if not file_path:
            raise ValueError("file_path cannot be empty.")

        ResultsLogger.validate_data(data)

        row = [data[column] for column in CSV_HEADERS]

        try:
            with _FILE_LOCK:
                # Initialize automatically if the file is absent.
                if not os.path.exists(file_path):
                    initialize_results_csv(file_path)

                with open(
                    file_path,
                    mode="a",
                    newline="",
                    encoding="utf-8",
                ) as csv_file:
                    writer = csv.writer(csv_file)
                    writer.writerow(row)

        except OSError:
            logger.exception(
                "Failed to append result to CSV: %s", file_path
            )
            raise
