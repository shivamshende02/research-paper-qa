
"""Initialize database tables for the Vaani prototype."""

from sqlalchemy import inspect

from src.db.session import Base, engine
from src.models import Document  # Registers the model with Base.metadata


def main():
    Base.metadata.create_all(bind=engine)

    inspector = inspect(engine)
    print("Database tables:", inspector.get_table_names())

    columns = inspector.get_columns("documents")
    print("\nDocument columns:")
    for column in columns:
        print(f"  {column['name']}: {column['type']}")

    print("\nDATABASE INITIALIZATION SUCCESSFUL")


if __name__ == "__main__":
    main()
