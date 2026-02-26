from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import time
# import all tables here
from db.model import VectorDB
from db.model import MainDB_Base

from dotenv import load_dotenv
import os

# helper to delete all tables
def delete_all_tables(engine):
    print(list(MainDB_Base.metadata.tables.keys()))
    print(f"WARNING: THIS WILL DELETE **ALL** TABLES IN THE MAIN DB IN 5 SECONDS, PLEASE DOUBLE CHECK!!")
    time.sleep(5)
    print(f"Dropping all tables now...")

    # Set statement timeout to prevent hanging
    with engine.connect() as conn:
        conn.execute(text("SET statement_timeout = '30s';"))
        conn.commit()

    try:
        MainDB_Base.metadata.drop_all(engine, checkfirst=True)
    except Exception as e:
        print(f"Error: {e}")
        print("Trying alternative method...")
        
        # Fallback: drop each table individually with CASCADE
        for table_name in reversed(list(MainDB_Base.metadata.tables.keys())):
            try:
                with engine.connect() as conn:
                    conn.execute(text(f"DROP TABLE IF EXISTS {table_name} CASCADE;"))
                    conn.commit()
                print(f"SUCCESS: Dropped {table_name}")
            except Exception as e2:
                print(f"WARNING: Error dropping {table_name}: {e2}")

# helper to delete a single table
def delete_table(table_name, engine):
    print(f"WARNING: THIS WILL DELETE TABLE *{table_name}* IN THE MAIN DB IN 5 SECONDS, PLEASE DOUBLE CHECK!!")
    time.sleep(5)
    
    try:
        print(f"Dropping table {table_name} now...")
        table = MainDB_Base.metadata.tables[table_name]
    except KeyError:
        raise ValueError(f"Table '{table_name}' not found in metadata")

    table.drop(engine)
    print(f"Dropped table {table_name}")

if __name__ == "__main__":
    # one-off script to delete tables
    load_dotenv()
    MAIN_DB_USER = os.getenv("MAIN_DB_USER")
    MAIN_DB_PASSWORD = os.getenv("MAIN_DB_PASSWORD")
    MAIN_DB_HOST = os.getenv("MAIN_DB_HOST")
    MAIN_DB_PORT = os.getenv("MAIN_DB_PORT")
    MAIN_DB_NAME = os.getenv("MAIN_DB_NAME")

    # MAIN_DB_URL = f"postgresql+psycopg2://{MAIN_DB_USER}:{MAIN_DB_PASSWORD}@{MAIN_DB_HOST}:{MAIN_DB_PORT}/{MAIN_DB_NAME}"
    MAIN_DB_URL = f"postgresql+psycopg2://{MAIN_DB_USER}:{MAIN_DB_PASSWORD}@{MAIN_DB_HOST}:{MAIN_DB_PORT}/{MAIN_DB_NAME}?sslmode=require"

    assert MAIN_DB_URL, "MAIN_DB_URL is not set"

    try:
        engine = create_engine(MAIN_DB_URL)
    except Exception as e:
        print(f"Error connecting to database: {e}")
        exit(1)

    # deletes all tables or a single table
    delete_all_tables(engine)
    # delete_table("structured_memory", engine)
