import json
import os
from datetime import datetime, timedelta

from flask import Blueprint, request, jsonify
import logging
from fuzzywuzzy import fuzz

from database import get_db_connection
from routes.decorators import retry_with_exponential_backoff
from dotenv import load_dotenv
import google.generativeai as genai
import psycopg2 # Import psycopg2 for PostgreSQL interaction
from psycopg2 import extras # For dictionary cursor

bp = Blueprint("chatbot", __name__, url_prefix="/chatbot")
log = logging.getLogger("Chatbot")
logging.basicConfig(level=logging.INFO)

# --- Gemini API Configuration ---
load_dotenv() # Load .env variables
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    log.critical("GEMINI_API_KEY environment variable not set. Chatbot functionality requiring Gemini will be disabled.")
    genai.configure(api_key="PLACEHOLDER_KEY_IF_NOT_SET") # Configure with a placeholder
else:
    genai.configure(api_key=GEMINI_API_KEY)

GEMINI_MODEL = "gemini-2.5-flash-preview-05-20"
try:
    if GEMINI_API_KEY:
        model = genai.GenerativeModel(GEMINI_MODEL)
    else:
        model = None
except Exception as e:
    log.error(f"Error initializing Gemini model '{GEMINI_MODEL}': {e}")
    model = None


async def get_user_transactions(user_id: str, num_transactions: int = 10):
    """
    Fetches the most recent transactions for a given user from the PostgreSQL database.
    Args:
        user_id (str): The ID of the user whose transactions to fetch.
        num_transactions (int): The maximum number of recent transactions to fetch.
    Returns:
        list: A list of dictionaries, where each dictionary represents a transaction.
              Returns an empty list if no transactions are found or an error occurs.
    """
    transactions = []
    conn = None
    try:
        conn = get_db_connection()
        # Use DictCursor to fetch rows as dictionaries
        with conn.cursor(cursor_factory=extras.DictCursor) as cur:
            # Assuming 'transactions' is your table name and it has 'user_id', 'amount', 'type', 'category', 'transaction_date', 'description'
            query = """
                SELECT date, amount, type, category, subcategory, description
                FROM user_transactions
                WHERE user_id = 101
                ORDER BY date DESC
                LIMIT 5
            """
            cur.execute(query, (user_id, num_transactions))
            for row in cur.fetchall():
                # Convert transaction_date to string if it's a datetime object
                transaction_date_str = row['date'].strftime("%Y-%m-%d") if isinstance(row['date'], datetime) else str(row['date'])
                transactions.append({
                    "amount": float(row['amount']),
                    "type": row['type'],
                    "category": row['category'],
                    "date": transaction_date_str,
                    "description": row['description']
                })
        log.info(f"Fetched {len(transactions)} transactions for user {user_id}.")
    except Exception as e:
        log.error(f"Error fetching transactions for user {user_id}: {e}")
    finally:
        if conn:
            conn.close()
    return transactions

# --- Dynamic Keyword Configuration ---
FINANCIAL_ADVICE_KEYWORDS = ["advice", "suggest", "recommend", "financial advice", "tip", "help me manage money"]
ADD_TRANSACTION_KEYWORDS = ["spent", "add", "record", "log", "submit", "expense", "income", "spending", "spend"]
FUZZY_MATCH_THRESHOLD = 80


# --- API Endpoint for Chatbot Interactions ---
@bp.route('/chatbot-interact', methods=['POST'])
@retry_with_exponential_backoff(max_retries=3)
async def chatbot_interact():
    if model is None:
        return jsonify({"error": "Gemini model not initialized. Please check GEMINI_API_KEY and server logs."}), 500

    data = request.get_json()
    user_utterance = data.get('utterance')
    user_id = '101' # User ID is crucial for fetching personalized data

    if not user_utterance:
        return jsonify({"error": "No utterance provided"}), 400
    if not user_id:
        return jsonify({"error": "No user_id provided. Cannot personalize response."}), 400

    user_utterance_lower = user_utterance.lower()

    # Detect "add transaction" intent
    is_add_transaction_related = any(
        fuzz.partial_ratio(keyword, user_utterance_lower) >= FUZZY_MATCH_THRESHOLD
        for keyword in ADD_TRANSACTION_KEYWORDS
    )

    # Detect "financial advice" intent
    is_financial_advice_related = any(
        fuzz.partial_ratio(keyword, user_utterance_lower) >= FUZZY_MATCH_THRESHOLD
        for keyword in FINANCIAL_ADVICE_KEYWORDS
    )

    if is_add_transaction_related:
        log.info(f"User utterance '{user_utterance}' indicates intent to add a transaction.")
        try:
            parsed_response = await parse_transaction_logic(user_utterance)
            if parsed_response.get('intent') == "record_expense_or_income":
                amount_val = parsed_response.get('amount', 'an amount')
                type_val = parsed_response.get('type', 'transaction')
                category_val = parsed_response.get('category', 'a category')
                date_val = parsed_response.get('date', 'today')
                parsed_response['response_text'] = (
                    f"Okay, I've parsed your request to record a {type_val} of {amount_val} for "
                    f"{category_val} on {date_val}. This can now be saved to your database."
                )
            # If parsing was unclear, parse_transaction_logic already sets response_text
            return jsonify(parsed_response), 200
        except Exception as e:
            log.error(f"Error during transaction parsing for '{user_utterance}': {e}")
            return jsonify({"error": "An internal server error occurred while parsing the transaction."}), 500

    if is_financial_advice_related:
        log.info(f"User utterance '{user_utterance}' indicates intent to get financial advice.")
        try:
            # --- FETCH TRANSACTIONS FROM DB ---
            user_transactions = await get_user_transactions(user_id=user_id, num_transactions=10)
            transactions_json_str = json.dumps(user_transactions, indent=2)

            financial_advice_prompt = f"""
               You are a financial assistant. Based on the user's recent spending patterns, provide actionable financial advice.

               Here are the user's most recent transactions:
               {transactions_json_str}

               User Utterance: "{user_utterance}"

               If no transactions are provided (the list is empty), state that you need more data about their spending to give personalized advice.
               Otherwise, provide advice that is specific, actionable, and encouraging. Focus on categories where spending is high, or patterns are clear.
               For example, if entertainment spending is high, suggest budgeting for it or exploring free activities.
               Keep the advice concise and easy to understand.

               Return your response in the following JSON format:
               {{
                   "intent": "financial_advice",
                   "data": {{"recent_transactions_summary": "Summary or key takeaways from transactions analyzed."}},
                   "response_text": "Your generated financial advice."
               }}
               """
            generation_config = {
                "response_mime_type": "application/json",
                "temperature": 0.7,
            }

            response = await model.generate_content_async(
                contents=[{"role": "user", "parts": [{"text": financial_advice_prompt}]}],
                generation_config=generation_config
            )

            gemini_raw_text = response.text.strip()
            log.info(f"🤖 Gemini API raw financial advice response: {gemini_raw_text}")

            try:
                parsed_response = json.loads(gemini_raw_text)
            except json.JSONDecodeError as e:
                log.error(f"Gemini did not return valid JSON for financial advice: {e} - Raw text: {gemini_raw_text}")
                return jsonify({"error": "Failed to interpret Gemini's financial advice (invalid JSON format).",
                                "raw_gemini_output": gemini_raw_text}), 500

            if 'response_text' not in parsed_response:
                parsed_response['response_text'] = "I'm sorry, I couldn't generate a coherent financial advice response."

            # Ensure 'data' field is a dictionary and add the actual transactions for context (optional)
            if 'data' not in parsed_response or not isinstance(parsed_response['data'], dict):
                 parsed_response['data'] = {}
            parsed_response['data']['actual_transactions_fetched'] = user_transactions

            return jsonify(parsed_response), 200

        except Exception as e:
            log.error(f"An error occurred during Gemini API call for financial advice: {e}")
            return jsonify({"error": f"An internal server error occurred: {e}"}), 500

    # If no specific intent is matched, return a general response
    return jsonify({
        "intent": "unclear",
        "data": None,
        "response_text": "I'm sorry, I didn't understand your request. Can you please rephrase it or ask about adding a transaction or financial advice?"
    }), 400


async def parse_transaction_logic(utterance: str):
    """
    Uses the Gemini model to parse transaction details from a user utterance.
    """
    current_date = datetime.now().strftime("%Y-%m-%d")

    prompt = f"""
    Extract the transaction details from the following user utterance and return them as a JSON object.
    If a field cannot be extracted, set it to null.
    The details should include:
    - Date (YYYY-MM-DD format, resolve relative dates like 'today', 'yesterday', 'last Friday' based on {current_date})
    - Amount (float, always positive).
    - Type (income or expense). Infer based on context if not explicit (e.g., "spent" implies expense, "received" implies income).
    - Category (e.g., Groceries, Salary, Utilities, Dining, Transport, Rent, Entertainment, Shopping, Medical). Use these specific categories if possible.
    - Subcategory (optional, more specific category like 'Coffee', 'Public Transport', 'Prescription').
    - Description (original transaction description or a concise summary).

    Example of desired JSON format:
    {{
        "date": "YYYY-MM-DD",
        "amount": 123.45,
        "type": "expense",
        "category": "Groceries",
        "subcategory": null,
        "description": "Weekly grocery shopping"
    }}
    Always provide a valid JSON object. Do not include any text outside the JSON block.
    If the utterance does not contain clear transaction details (e.g., missing amount, or just a general chat),
    return a JSON with "intent": "unclear" and a helpful "response_text".

    User Utterance:
    {utterance}
    """
    try:
        log.info(f"🔍 Sending parsing prompt to Gemini API for utterance: '{utterance}'")
        if model is None:
            return {
                "intent": "unclear",
                "data": None,
                "response_text": "Chatbot model is not initialized. Cannot parse transaction."
            }

        generation_config = {
            "response_mime_type": "application/json",
            "temperature": 0.1, # Keep temperature low for structured extraction
        }

        response = await model.generate_content_async(
            contents=[{"role": "user", "parts": [{"text": prompt}]}],
            generation_config=generation_config
        )

        transaction_details_str = response.text.strip()
        log.info(f"🤖 Gemini API raw parsing response (cleaned): {transaction_details_str}")

        try:
            parsed_json = json.loads(transaction_details_str)
        except json.JSONDecodeError as e:
            log.error(
                f"Gemini did not return valid JSON for parsing: {transaction_details_str} - Error: {e}")
            return {
                "intent": "unclear",
                "data": None,
                "response_text": "I received a response from Gemini, but couldn't understand its format for the transaction. Could you try rephrasing with more specific details?"
            }

        # Check if Gemini explicitly returned an "unclear" intent
        if parsed_json.get('intent') == "unclear":
            return parsed_json # Return it as is if Gemini decided it's unclear

        # Basic validation and type conversion for extracted fields
        if 'amount' in parsed_json and isinstance(parsed_json['amount'], (int, float)):
            parsed_json['amount'] = abs(parsed_json['amount'])
        else:
            log.warning(f"Amount not found or invalid type in parsed JSON: {parsed_json.get('amount')}. Setting to None.")
            # If a critical field like 'amount' is missing or invalid, consider it unclear
            return {
                "intent": "unclear",
                "data": None,
                "response_text": "It looks like the amount for your transaction is missing or unclear. Could you provide a specific numerical amount?"
            }

        if 'date' in parsed_json and parsed_json['date']:
            try:
                # Attempt to parse and reformat date to ensure YYYY-MM-DD
                parsed_date = datetime.strptime(parsed_json['date'], "%Y-%m-%d").strftime("%Y-%m-%d")
                parsed_json['date'] = parsed_date
            except ValueError:
                log.warning(f"Could not parse date '{parsed_json['date']}'. Setting to current date.")
                parsed_json['date'] = current_date
        else:
            parsed_json['date'] = current_date # Default to current date if missing or invalid

        if 'type' not in parsed_json or parsed_json['type'] not in ['income', 'expense']:
            log.warning(f"Type not found or invalid in parsed JSON: {parsed_json.get('type')}. Defaulting to 'expense'.")
            parsed_json['type'] = 'expense' # Default to expense if type is ambiguous

        # Set the intent for successful parsing
        parsed_json['intent'] = "record_expense_or_income"

        return parsed_json
    except Exception as e:
        log.error(f"❌ Error during transaction parsing logic: {e}")
        # Return a generic error response for unexpected exceptions
        return {
            "intent": "unclear",
            "data": None,
            "response_text": "An unexpected error occurred while trying to parse your transaction. Please try again."
        }