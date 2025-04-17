import pandas as pd
import psycopg2

def load_notes(input_cfg):
    mode = input_cfg["mode"]

    if mode == "csv":
        path = input_cfg["csv"]["path"]
        column = input_cfg["csv"]["column"]
        return pd.read_csv(path)[column].dropna().tolist()

    elif mode == "db":
        db = input_cfg["db"]
        conn = psycopg2.connect(
            host=db["host"],
            port=db["port"],
            user=db["user"],
            password=db["password"],
            database=db["database"]
        )
        query = f'SELECT "{db["column"]}" FROM "{db["schema"]}"."{db["table"]}"'
        df = pd.read_sql(query, conn)
        conn.close()
        return df[db["column"]].dropna().tolist()

    else:
        raise ValueError("Unsupported input mode")
