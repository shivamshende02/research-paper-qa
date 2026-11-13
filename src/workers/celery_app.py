
"""Celery application configuration."""

from celery import Celery

from src.core.config import settings

celery_app = Celery(
    "vaani",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    include=["src.workers.tasks"],
)

celery_app.conf.update(
    # Serialization
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],

    # Timezone
    timezone="Asia/Kolkata",
    enable_utc=True,

    # Reliability
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,

    # Task execution limits
    task_soft_time_limit=300,
    task_time_limit=360,

    # Result handling
    result_expires=3600,
    task_track_started=True,

    # Worker behavior
    broker_connection_retry_on_startup=True,
)
