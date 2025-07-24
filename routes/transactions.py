from datetime import datetime
from typing import Optional

from dateutil.parser import isoparse
from flask import Blueprint, request, jsonify

from database import execute_query
import logging

bp = Blueprint("transactions", __name__, url_prefix="/transactions")
log = logging.getLogger("Chatbot")
logging.basicConfig(level=logging.INFO)

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
    params = (
        user_id, date_obj, amount, trans_type, category, subcategory, description, merchant, location, payment_method)
    log.info(
        f"Attempting to add transaction to DB: Date={date_obj}, Amount={amount}, Type={trans_type}, Category={category}")
    return execute_query(query, params)


@bp.route('/add_transaction', methods=['POST'])
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
    required_fields = ["user_id", "date", "amount", "type", "category", "description", "merchant", "location",
                       "payment_method"]
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