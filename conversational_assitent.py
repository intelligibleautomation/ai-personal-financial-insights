from datetime import datetime
from typing import Optional

import psycopg2
from dateutil.parser import isoparse
from flask import Flask, request, jsonify
import re
import logging
import os
import google.generativeai as genai
from financial_health_score import (
    load_config,
    load_and_process_transactions,
    compute_metrics,
    calculate_financial_health_score,
)

# 🔐 Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Configure logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("BigAmountAlert")

# Initialize Flask app
app = Flask(__name__)


@app.route("/parse_transaction", methods=["POST"])
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
        return (
            jsonify({"error": "Failed to parse transaction. Please try again later."}),
            500,
        )


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
@app.route("/alert", methods=["POST"])
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
        return (
            jsonify({"error": "Both 'question' and 'threshold_amount' are required."}),
            400,
        )

    # Extract amount from the question
    match = re.search(r"₹?(\d+(?:,\d{3})*(?:\.\d{1,2})?)", question)
    if not match:
        return jsonify({"error": "No valid amount found in the question."}), 400

    purchase_amount = float(match.group(1).replace(",", ""))
    log.info(f"Extracted purchase amount: ₹{purchase_amount}")

    if purchase_amount < threshold_amount:
        log.warning(
            f"⚠️ Alert: Purchase amount ₹{purchase_amount} is below the threshold ₹{threshold_amount}."
        )
        return jsonify(
            {
                "alert": f"Purchase amount ₹{purchase_amount} is below the threshold ₹{threshold_amount}.",
                "proceed": "Do you want to check the affordability?",
            }
        )
    else:
        log.warning(
            f"⚠️ Alert: Purchase amount ₹{purchase_amount} is above the threshold ₹{threshold_amount}."
        )
        return jsonify(
            {
                "alert": f"Purchase amount ₹{purchase_amount} is above the threshold ₹{threshold_amount}.",
                "proceed": "Do you still want to proceed with the purchase?",
            }
        )


# -------------------------------
# 💬 Affordability Endpoint
# -------------------------------
@app.route("/affordability", methods=["POST"])
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
            log.info(
                f"✅ Query executed successfully: {query[:50]}..."
            )  # Log a snippet of the query
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
    user_id: str,
    date: str,
    amount: float,
    type: str,
    category: str,
    subcategory: Optional[str],
    description: str,
    merchant: str,
    location: str,
    payment_method: str,
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
        date_obj = datetime.strptime(
            date, "%d-%m-%Y"
        ).date()  # Fallback to DD-MM-YYYY format
    query = """
    INSERT INTO user_transactions (user_id, date, amount, type, category, subcategory, description, merchant, location, payment_method)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    params = (
        user_id,
        date_obj,
        amount,
        trans_type,
        category,
        subcategory,
        description,
        merchant,
        location,
        payment_method,
    )
    log.info(
        f"Attempting to add transaction to DB: Date={date_obj}, Amount={amount}, Type={trans_type}, Category={category}"
    )
    return execute_query(query, params)


@app.route("/add_transaction", methods=["POST"])
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
    required_fields = [
        "user_id",
        "date",
        "amount",
        "type",
        "category",
        "description",
        "merchant",
        "location",
        "payment_method",
    ]
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        return (
            jsonify({"error": f"Missing required fields: {', '.join(missing_fields)}"}),
            400,
        )

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
        payment_method=payment_method,
    )

    if success:
        return jsonify({"message": "Transaction added successfully."}), 201
    else:
        return jsonify({"error": "Failed to add transaction to the database."}), 500


# --------------------------------------------
# 📊 Get Statistics from Database for UI
# --------------------------------------------


@app.route("/get_statistics", methods=["GET"])
def get_statistics():
    """
    API endpoint to retrieve transaction statistics for the user.

    Query Parameters:
        user_id (str): Required. User ID to filter transactions.
        start_date (str): Optional. Start date for filtering (YYYY-MM-DD format).
        end_date (str): Optional. End date for filtering (YYYY-MM-DD format).

    Returns:
        JSON: Transaction statistics including total income, total expenses, balance, and discretionary ratio.
    """
    try:
        user_id = request.args.get("user_id")
        if not user_id:
            log.warning("❌ Missing user_id parameter in request")
            return jsonify({"error": "Missing user_id parameter"}), 400

        start_date = request.args.get("start_date")
        end_date = request.args.get("end_date")

        log.info(
            f"🔍 Processing statistics request for user_id={user_id}, start_date={start_date}, end_date={end_date}"
        )

        # Validate date format if provided
        if start_date:
            try:
                datetime.strptime(start_date, "%Y-%m-%d")
            except ValueError:
                log.warning(f"❌ Invalid start_date format: {start_date}")
                return (
                    jsonify({"error": "Invalid start_date format. Use YYYY-MM-DD."}),
                    400,
                )

        if end_date:
            try:
                datetime.strptime(end_date, "%Y-%m-%d")
            except ValueError:
                log.warning(f"❌ Invalid end_date format: {end_date}")
                return (
                    jsonify({"error": "Invalid end_date format. Use YYYY-MM-DD."}),
                    400,
                )

        # Retrieve statistics from the database
        stats = retrieve_transaction_statistics(user_id, start_date, end_date)

        # Check if there was an error
        if not stats:
            log.error("❌ No statistics returned from database function")
            return jsonify({"error": "Failed to retrieve statistics"}), 500

        if "error" in stats:
            log.error(f"❌ Database error: {stats['error']}")
            return jsonify(stats), 500

        log.info(f"✅ Successfully retrieved statistics for user {user_id}")
        return jsonify(stats), 200

    except Exception as e:
        log.error(f"❌ Unexpected error in get_statistics: {str(e)}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


def get_discretionary_categories():
    """
    Load discretionary categories from the configuration file.

    Returns:
        list: List of discretionary categories.
    """
    try:
        from financial_health_score import load_config

        _, discretionary_categories = load_config("categories_config.json")
        log.info(f"✅ Loaded discretionary categories: {discretionary_categories}")
        return discretionary_categories
    except Exception as e:
        log.error(f"❌ Error loading discretionary categories: {str(e)}")
        # Return default categories if config fails to load
        default_categories = ["entertainment", "shopping", "dining", "luxury", "travel"]
        log.info(f"🔧 Using default discretionary categories: {default_categories}")
        return default_categories


def retrieve_transaction_statistics(
    user_id: str, start_date: Optional[str] = None, end_date: Optional[str] = None
) -> dict:
    """
    Retrieve transaction statistics for a given user from the PostgreSQL database.

    Args:
        user_id (str): User ID to filter transactions.
        start_date (Optional[str]): Start date for filtering transactions (YYYY-MM-DD format).
        end_date (Optional[str]): End date for filtering transactions (YYYY-MM-DD format).

    Returns:
        dict: Dictionary containing total income, total expenses, balance, and discretionary ratio.
    """
    conn = None
    try:
        # Load discretionary categories from config
        discretionary_categories = get_discretionary_categories()
        log.info(f"🔧 Using discretionary categories: {discretionary_categories}")

        # Check if database connection is possible
        try:
            conn = get_db_connection()
        except Exception as db_error:
            log.error(f"❌ Database connection failed: {str(db_error)}")
            return {"error": f"Database connection failed: {str(db_error)}"}

        with conn.cursor() as cursor:
            # Build dynamic query with optional date filtering
            query_params = [user_id]
            date_filter = ""

            if start_date and end_date:
                date_filter = " AND date BETWEEN %s AND %s"
                query_params.extend([start_date, end_date])
            elif start_date:
                date_filter = " AND date >= %s"
                query_params.append(start_date)
            elif end_date:
                date_filter = " AND date <= %s"
                query_params.append(end_date)

            # Build safe parameterized query for discretionary categories
            # Create placeholders for the IN clause
            discretionary_placeholders = ", ".join(
                ["%s"] * len(discretionary_categories)
            )

            # Build the final parameter list: user_id + date params + discretionary categories
            discretionary_params = [cat.lower() for cat in discretionary_categories]
            final_params = query_params + discretionary_params

            # Build the query properly with correct parameterization
            query = f"""
            SELECT 
                COALESCE(SUM(CASE WHEN type = 'Income' THEN amount ELSE 0 END), 0) AS total_income,
                COALESCE(SUM(CASE WHEN type = 'Expense' THEN amount ELSE 0 END), 0) AS total_expenses,
                COALESCE(SUM(CASE WHEN type = 'Expense' AND LOWER(category) IN ({discretionary_placeholders}) THEN amount ELSE 0 END), 0) AS discretionary_spending
            FROM user_transactions
            WHERE user_id = %s{date_filter}
            """

            log.info(
                f"🔍 Executing query with params: user_id={user_id}, categories={discretionary_categories}"
            )
            log.debug(f"📝 SQL Query: {query}")
            log.debug(f"📝 Parameters: {final_params}")

            cursor.execute(query, tuple(final_params))
            result = cursor.fetchone()

            if result:
                total_income, total_expenses, discretionary_spending = result

                # Ensure values are not None (COALESCE should handle this, but double-check)
                total_income = total_income or 0
                total_expenses = total_expenses or 0
                discretionary_spending = discretionary_spending or 0

                balance = total_income - total_expenses

                # Calculate discretionary ratio
                discretionary_ratio = (
                    discretionary_spending / total_expenses if total_expenses > 0 else 0
                )

                stats = {
                    "total_income": float(total_income),
                    "total_expenses": float(total_expenses),
                    "balance": float(balance),
                    "discretionary_spending": float(discretionary_spending),
                    "discretionary_ratio": round(discretionary_ratio, 4),
                    "date_range": {"start_date": start_date, "end_date": end_date},
                }

                log.info(f"✅ Successfully calculated statistics: {stats}")
                return stats
            else:
                log.warning("⚠️ No transactions found for this user")
                return {"error": "No transactions found for this user."}

    except psycopg2.Error as e:
        error_msg = f"Database error occurred: {str(e)}"
        log.error(f"❌ {error_msg}")
        return {"error": error_msg}
    except Exception as e:
        error_msg = f"Failed to process transaction statistics: {str(e)}"
        log.error(f"❌ {error_msg}")
        return {"error": error_msg}
    finally:
        if conn:
            conn.close()
            log.info("🔒 Database connection closed")


# -------------------------------
# 🚀 Run Flask App
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0')
