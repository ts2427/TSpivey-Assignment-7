import pandas as pd
import sqlite3
import os

def save_to_csv(df_databreach, output_dir='output'):
    """
    Saves the databreach dataframe to a CSV file.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    filepath = os.path.join(output_dir, 'databreach.csv')
    df_databreach.to_csv(filepath, index=False)
    print(f"Saved databreach to {filepath}")


def load_to_database(df_databreach, db_name='databreach.db'):
    """
    Loads the databreach dataframe into a SQLite database.
    """
    conn = sqlite3.connect(db_name)
    df_databreach.to_sql('databreach', conn, if_exists='replace', index=False)
    print(f"Loaded databreach into database table")
    conn.close()
    print(f"Database saved as {db_name}")


def load_from_database(db_name='databreach.db'):
    """
    Loads the databreach table from the SQLite database.
    """
    conn = sqlite3.connect(db_name)
    df = pd.read_sql_query("SELECT * FROM databreach", conn)
    conn.close()
    return df


def get_table_info(db_name='databreach.db'):
    """
    Prints information about all tables in the database.
    """
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    
    # Get all table names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    for table in tables:
        table_name = table[0]
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cursor.fetchone()[0]
        print(f"{table_name}: {count:,} rows")
    
    conn.close()


def load_sec_reference(db_name='databreach.db'):
    """
    Loads SEC company reference data into the database.
    """
    sec_data = pd.read_csv('CIK_NAME_TICKER_EXCHANGE.csv')
    
    conn = sqlite3.connect(db_name)
    sec_data.to_sql('sec_company_reference', conn, if_exists='replace', index=False)
    conn.close()
    
    print(f"Loaded {len(sec_data):,} SEC companies to sec_company_reference table")