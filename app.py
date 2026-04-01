import streamlit as st
import sqlite3
import pandas as pd
import anthropic
import os

# --- Page Config ---
st.set_page_config(
    page_title="SQL Bot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Dark Modern Theme ---
st.markdown("""
    <style>
        /* Main background */
        .stApp {
            background-color: #0e1117;
            color: #ffffff;
        }

        /* Sidebar */
        [data-testid="stSidebar"] {
            background-color: #161b22;
            border-right: 1px solid #30363d;
        }

        /* Chat input */
        /* Chat messages */
        [data-testid="stChatMessage"] {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
        color: #c9d1d9 !important;
        }

/* Fix user message text color */
[data-testid="stChatMessage"] p {
    color: #c9d1d9 !important;
}

        /* Chat messages */
        [data-testid="stChatMessage"] {
            background-color: #161b22;
            border: 1px solid #30363d;
            border-radius: 10px;
            padding: 10px;
            margin-bottom: 10px;
        }

        /* Code blocks */
        .stCode {
            background-color: #0d1117 !important;
            border: 1px solid #30363d;
            border-radius: 8px;
        }

        /* Dataframe */
        [data-testid="stDataFrame"] {
            border: 1px solid #30363d;
            border-radius: 8px;
        }

        /* Spinner */
        .stSpinner {
            color: #a371f7 !important;
        }

        /* Title */
        h1 {
            color: #a371f7 !important;
        }
    </style>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.markdown("""
        <div style='text-align: center; padding: 20px 0;'>
            <div style='font-size: 48px;'>🤖</div>
            <h2 style='color: #a371f7; margin: 0;'>SQL Bot</h2>
            <p style='color: #8b949e; font-size: 13px;'>Powered by Claude AI</p>
        </div>
        <hr style='border-color: #30363d;'>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div style='padding: 10px 0;'>
            <p style='color: #8b949e; font-size: 13px;'>📂 <b style='color:#58a6ff'>Connected Tables</b></p>
            <p style='color: #c9d1d9; font-size: 13px;'>• user_loan</p>
            <p style='color: #c9d1d9; font-size: 13px;'>• payments</p>
        </div>
        <hr style='border-color: #30363d;'>
    """, unsafe_allow_html=True)

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# --- API Client ---
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

SCHEMA_CONTEXT = """
## Database: Loan Management System

### Table: user_loan
Purpose: Stores all loan applications and their lifecycle status.

Columns:
- id              : Unique loan ID
- user_id         : The borrower (links to users table)
- created_at      : When the loan record was created
- applied_at      : When the user submitted the application
- amount          : Loan amount in rupees
- purpose         : Reason for loan (e.g. Medical, Education, Business)
- loan_type       : Type of loan (personal, home, business, vehicle)
- status          : Current state → pending, approved, disbursed, completed, rejected 
- emi_date (ignore)        : Day of month when EMI is due (e.g. 5 means 5th of every month)
- client_id (ignore)      : Which business client/partner issued this loan
- disbursed_at    : When money was actually sent to borrower
- loan_product_id (ignore) : Links to the loan product/plan chosen
- lender_id       : Which lender/bank is funding this loan
- esign_id       (ignore) : Electronic signature document ID
- esign_upload_status (ignore) : Whether esign doc is uploaded → uploaded, pending, failed
- agreement_status    : Loan agreement state → pending,done
- completed_at    : When the loan was fully repaid

### Table: payments
Purpose: Every EMI or payment made against a loan.

Columns:
- id               : Unique payment ID
- user_id          : Borrower who made the payment
- user_loan_id     : Which loan this payment belongs to (FK → user_loan.id)
- amount           : Payment amount in rupees
- status           : Payment state → success, failed, pending
- mode             : Payment method (UPI, NEFT, NACH, cash)
- ref_number       : Bank/payment gateway reference number
- payment_deadline : Date by which payment was due
- payment_date     : Actual date payment was made
- client_id        : Business client associated with this payment
- type             : Payment type → emi, prepayment, foreclosure, penalty
- overdue          : Number of days payment is overdue (0 if on time)
- penalty          : Penalty amount charged for late payment
- extension_deadline  : New deadline if extension was granted
- extension_status    : Whether extension was approved
- extension_count     : How many times extension has been requested
- bounced_charge   : Charge applied if payment bounced

### Relationships:
- payments.user_loan_id → user_loan.id  (one loan has many payments)
- payments.user_id      → user_loan.user_id

### Business Rules:
- A loan is "active" when status IN ('disbursed')
- A payment is "on time" when all the emis less than or equal to today have payment_date <= payment_deadline
- DPD days past due = (payment_date - payment_deadline) If a loan has any of the payment with DPD > 0 
- A loan is "closed" when status IN ('completed')
- Loan DPD - max of individual payments DPD of a user_loan_id
- Current DPD - max of individual payments DPD of a user_loan_id which is still not marked as done today

Ex: one user_loan_id has 3 payments deadline 
Payment deadline  Payment date           Status     DPDs
Jan 1 2026        Feb 6 2026             DONE       36 
Feb 1 2026        Feb 6 2026             DONE       5 
Mar 1 2026        null                   PENDING    30

Loan dpd is 36 days and current dpd is 5 days
"""

@st.cache_resource
def load_database():
    conn = sqlite3.connect(":memory:", check_same_thread=False)
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
