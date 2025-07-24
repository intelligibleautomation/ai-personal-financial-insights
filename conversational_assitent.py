from datetime import datetime
from typing import Optional

import psycopg2
from dateutil.parser import isoparse
from flask import Flask, request, jsonify
import re
import logging
import os
import google.generativeai as genai

# 🔐 Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Configure logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("BigAmountAlert")

# Initialize Flask app
app = Flask(__name__)


@app.route('/parse_transaction', methods=['POST'])
def parse_transaction():
    """
    Parses a transaction from the user's utterance using the Gemini API.

    Returns:
        JSON: Parsed transaction details.
    """
    data = request.json
    utterance = data.get("utterance")

    if not utterance:
        return jsonify({"error": "'utterance' is required."}), 400

    # Current date for resolving relative dates
    current_date = datetime.now().strftime("%Y-%m-%d")

    # Gemini API prompt
    prompt = f"""
    Extract the transaction details from the following user utterance and return them as a JSON object.
    The details should include:
    - Date (YYYY-MM-DD format, resolve relative dates like 'yesterday' based on {current_date})
    - Amount (float, positive for income, negative for expense)
    - Type (income or expense)
    - Category (e.g., Groceries, Salary, Utilities)
    - Subcategory (optional, more specific category)
    - Description (original transaction description)

    User Utterance:
    {utterance}
    """
    try:
        log.info("🔍 Sending prompt to Gemini API...")
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        log.info(f"🤖 Gemini API raw response: {response.text}")

        # Parse the response
        transaction_details = response.text.strip()
        return jsonify({"transaction_details": transaction_details})
    except Exception as e:
        log.error(f"❌ Error parsing transaction: {e}")
        return jsonify({"error": "Failed to parse transaction. Please try again later."}), 500



# -------------------------------
# 🗄️ Fetch Transactions (Dummy Data)
# -------------------------------
def fetch_transactions():
    """
    Returns dummy transactions.

    Returns:
        list: List of dummy transaction dictionaries.
    """
    transactions = [
        {"date": "2025-07-01", "category": "Groceries", "amount": 4000},
        {"date": "2025-07-05", "category": "Rent", "amount": 12000},
        {"date": "2025-07-10", "category": "Dining", "amount": 2000},
        {"date": "2025-07-15", "category": "Utilities", "amount": 3000},
        {"date": "2025-07-15", "category": "income", "amount": 100000},
    ]
    return transactions


# -------------------------------
# 💬 Gemini Affordability Engine
# -------------------------------
def check_affordability(question, transactions):
    """
    Uses the Gemini API to check affordability based on transactions and a user question.

    Args:
        question (str): The user's question about affordability.
        transactions (list): List of transaction dictionaries.

    Returns:
        str: The response from the Gemini API.
    """
    model = genai.GenerativeModel("gemini-1.5-flash")

    prompt = f"""
    You are a smart personal finance assistant. Based on the user's recent transactions, answer the question below.

    Transactions:
    {transactions}

    Question:
    {question}

    Respond with:
    - Whether the purchase is affordable
    - What adjustments the user can make to afford it
    - Keep it short and actionable
    """

    response = model.generate_content(prompt)
    return response.text


# -------------------------------
# ⚠️ Alert Endpoint
# -------------------------------
@app.route('/alert', methods=['POST'])
def alert_big_purchase():
    """
    Extracts the purchase amount from the question, compares it with the threshold,
    and returns an alert if the amount is below the threshold.

    Returns:
        JSON: Response with alert message.
    """
    data = request.json
    question = data.get("question")
    threshold_amount = data.get("threshold_amount")

    if not question or not threshold_amount:
        return jsonify({"error": "Both 'question' and 'threshold_amount' are required."}), 400

    # Extract amount from the question
    match = re.search(r"₹?(\d+(?:,\d{3})*(?:\.\d{1,2})?)", question)
    if not match:
        return jsonify({"error": "No valid amount found in the question."}), 400

    purchase_amount = float(match.group(1).replace(",", ""))
    log.info(f"Extracted purchase amount: ₹{purchase_amount}")

    if purchase_amount < threshold_amount:
        log.warning(f"⚠️ Alert: Purchase amount ₹{purchase_amount} is below the threshold ₹{threshold_amount}.")
        return jsonify({
            "alert": f"Purchase amount ₹{purchase_amount} is below the threshold ₹{threshold_amount}.",
            "proceed": "Do you want to check the affordability?"
        })
    else:
        log.warning(f"⚠️ Alert: Purchase amount ₹{purchase_amount} is above the threshold ₹{threshold_amount}.")
        return jsonify({
            "alert": f"Purchase amount ₹{purchase_amount} is above the threshold ₹{threshold_amount}.",
            "proceed": "Do you still want to proceed with the purchase?"

        })


# -------------------------------
# 💬 Affordability Endpoint
# -------------------------------
@app.route('/affordability', methods=['POST'])
def affordability_check():
    """
    Performs an affordability check using the Gemini API.

    Returns:
        JSON: Response with affordability check result.
    """
    data = request.json
    question = data.get("question")

    if not question:
        return jsonify({"error": "'question' is required."}), 400

    # Fetch transactions and check affordability
    transactions = fetch_transactions()
    affordability_response = check_affordability(question, transactions)
    return jsonify({"affordability_response": affordability_response})


def get_db_connection() -> psycopg2.extensions.connection:
    """
    Establish and return a PostgreSQL database connection.

    Returns:
        psycopg2.extensions.connection: Database connection object.
    """
    try:
        conn = psycopg2.connect(
            dbname=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT"),
        )
        # Check connection to ensure it's valid
        with conn.cursor() as cursor:
            cursor.execute("SELECT 1")
        log.info("✅ Successfully connected to PostgreSQL")
        return conn
    except psycopg2.OperationalError as e:
        log.error(f"❌ Database connection failed: {str(e)}")
        raise


def execute_query(query: str, params: Optional[tuple] = None) -> bool:
    """
    Execute a SQL query with optional parameters.

    Args:
        query (str): SQL query to execute.
        params (Optional[tuple]): Parameters for the query.

    Returns:
        bool: True if the query executed successfully, False otherwise.
    """
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute(query, params)
            conn.commit()
            log.info(f"✅ Query executed successfully: {query[:50]}...")  # Log a snippet of the query
            return True
    except psycopg2.Error as e:
        log.error(f"❌ Query execution failed: {str(e)} for query: {query}")
        # Rollback in case of error
        if conn:
            conn.rollback()
        return False
    finally:
        if conn:
            conn.close()
            log.info("🔒 Database connection closed")


def add_transaction_to_db(
        user_id: str, date: str, amount: float, type: str, category: str, subcategory: Optional[str], description: str,
        merchant: str, location: str, payment_method: str
) -> bool:
    """
    Add a new transaction to the PostgreSQL database.
    """
    # Ensure trans_type is capitalized for consistency with DB schema (if needed)
    trans_type = type.capitalize()
    # Handle ISO 8601 and DD-MM-YYYY formats
    try:
        date_obj = isoparse(date).date()  # Parse ISO 8601 format
    except ValueError:
        date_obj = datetime.strptime(date, "%d-%m-%Y").date()  # Fallback to DD-MM-YYYY format
    query = """
    INSERT INTO user_transactions (user_id, date, amount, type, category, subcategory, description, merchant, location, payment_method)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    params = (user_id, date_obj, amount, trans_type, category, subcategory, description, merchant, location, payment_method)
    log.info(
        f"Attempting to add transaction to DB: Date={date_obj}, Amount={amount}, Type={trans_type}, Category={category}")
    return execute_query(query, params)

@app.route('/add_transaction', methods=['POST'])
def add_transaction():
    """
    API endpoint to add a new transaction to the database.

    Expects a JSON payload with the following fields:
    - user_id (str): User ID associated with the transaction.
    - date (str): Transaction date in ISO 8601 or DD-MM-YYYY format.
    - amount (float): Transaction amount.
    - type (str): Type of transaction (Income/Expense).
    - category (str): Transaction category.
    - subcategory (Optional[str]): Transaction subcategory.
    - description (str): Transaction description.
    - merchant (str): Merchant associated with the transaction.
    - location (str): Location of the transaction.
    - payment_method (str): Payment method used.

    Returns:
        JSON: Success or error message.
    """
    data = request.json

    # Validate required fields
    required_fields = ["user_id", "date", "amount", "type", "category", "description", "merchant", "location", "payment_method"]
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        return jsonify({"error": f"Missing required fields: {', '.join(missing_fields)}"}), 400

    # Extract fields from the JSON payload
    user_id = data["user_id"]
    date = data["date"]
    amount = data["amount"]
    trans_type = data["type"]
    category = data["category"]
    subcategory = data.get("subcategory")  # Optional field
    description = data["description"]
    merchant = data["merchant"]
    location = data["location"]
    payment_method = data["payment_method"]

    # Call the database function
    success = add_transaction_to_db(
        user_id=user_id,
        date=date,
        amount=amount,
        type=trans_type,
        category=category,
        subcategory=subcategory,
        description=description,
        merchant=merchant,
        location=location,
        payment_method=payment_method
    )

    if success:
        return jsonify({"message": "Transaction added successfully."}), 201
    else:
        return jsonify({"error": "Failed to add transaction to the database."}), 500


# -------------------------------
# 🚀 Run Flask App
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0')
