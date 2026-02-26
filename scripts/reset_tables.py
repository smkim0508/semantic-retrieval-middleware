from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import time
# import all tables here
from db.model import VectorDB
from db.model import MainDB_Base

from dotenv import load_dotenv
import os

# import creation and deletion scripts
from scripts.create_tables import create_all_tables
from scripts.delete_tables import delete_all_tables

# NOTE: trouble shooting: if script imports do not work, use PYTHONPATH=. to explicitly include root of project
if __name__ == "__main__":
    # one-off script to reset db, by deleting then creating all tables
    load_dotenv()
    # one-off script to reset tables
    MAIN_DB_USER = os.getenv("MAIN_DB_USER")
    MAIN_DB_PW = os.getenv("MAIN_DB_PW")
    MAIN_DB_HOST = os.getenv("MAIN_DB_HOST")
    MAIN_DB_PORT = os.getenv("MAIN_DB_PORT")
    MAIN_DB_NAME = os.getenv("MAIN_DB_NAME")

    # MAIN_DB_URL = f"postgresql+psycopg2://{MAIN_DB_USER}:{MAIN_DB_PW}@{MAIN_DB_HOST}:{MAIN_DB_PORT}/{MAIN_DB_NAME}"
    MAIN_DB_URL = f"postgresql+psycopg2://{MAIN_DB_USER}:{MAIN_DB_PW}@{MAIN_DB_HOST}:{MAIN_DB_PORT}/{MAIN_DB_NAME}?sslmode=require"

    assert MAIN_DB_URL, "MAIN_DB_URL is not set"

    try:
        engine = create_engine(MAIN_DB_URL)
    except Exception as e:
        print(f"Error connecting to database: {e}")
        exit(1)

    # NOTE: this is the actual logic to reset: edit as needed
    delete_all_tables(engine)
    create_all_tables(engine)
    print(f"All tables reset successfully!")
