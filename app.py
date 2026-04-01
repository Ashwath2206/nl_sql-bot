
import streamlit as st
import sqlite3
import pandas as pd
import anthropic
import os

# --- API Client ---
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

SCHEMA_CONTEXT = """
user_loan contains customer loan details.
payments contains payment transaction history.
"""

@st.cache_resource
def load_database():
    conn = sqlite3.connect(":memory:")
    df = pd.read_csv("sql_bot_test.csv")
    df.columns = [c.strip().lower() for c in df.columns]
    df.to_sql("user_loan", conn, if_exists="replace", index=False)
    df2 = pd.read_csv("sql_bot_train_payments.csv")
    df2.columns = [c.strip().lower() for c in df2.columns]
    df2.to_sql("payments", conn, if_exists="replace", index=False)
    return conn

def get_schema(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    schema = ""
    for (table_name,) in tables:
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        col_list = ", ".join([f"{col[1]} ({col[2]})" for col in columns])
        schema += f"Table: {table_name}\nColumns: {col_list}\n\n"
    return schema

def build_prompt(question, schema):
    return f"""You are a SQL expert. Given the database schema below, write a SQL query that answers the question.

SCHEMA:
{schema}

SCHEMA CONTEXT:
{SCHEMA_CONTEXT}

RULES:
- Return ONLY the SQL query, nothing else
- No explanation, no markdown, no backticks
- Use proper JOINs when needed
- Use lower() to avoid case sensitivity

QUESTION: {question}

SQL:"""

def ask_claude(question, conn):
    schema = get_schema(conn)
    prompt = build_prompt(question, schema)
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text.strip()

def run_query(sql, conn):
    try:
        result = pd.read_sql_query(sql, conn)
        return result, None
    except Exception as e:
        return None, str(e)

# --- UI ---
st.set_page_config(page_title="SQL Bot", page_icon="🤖")
st.title("🤖 SQL Bot")
st.caption("Ask a question about your data!")

conn = load_database()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            st.code(msg["sql"], language="sql")
            if msg["result"] is not None:
                st.dataframe(msg["result"])
            elif msg["error"]:
                st.error(f"Query error: {msg['error']}")
        else:
            st.markdown(msg["content"])

user_input = st.chat_input("e.g. How many users have missed payments?")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            sql = ask_claude(user_input, conn)
            st.code(sql, language="sql")
            result, error = run_query(sql, conn)
            if result is not None:
                st.dataframe(result)
            else:
                st.error(f"Query error: {error}")

    st.session_state.messages.append({
        "role": "assistant",
        "sql": sql,
        "result": result,
        "error": error
    })
