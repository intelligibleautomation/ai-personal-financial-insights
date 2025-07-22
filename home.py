import os
from logging import Logger
import psycopg2
from typing import Any, Optional, Dict, Literal, TypedDict
from datetime import datetime
from dotenv import load_dotenv
from utils.logging_utils import setup_logging
import json  # Import json for parsing Gemini's output
from dateutil.parser import isoparse  # Import for ISO 8601 parsing

# Load environment variables
log: Logger = setup_logging("AI Financial Assistant")
load_dotenv()
log.info("✨ Environment variables loaded")

# Validate required environment variables
REQUIRED_ENV_VARS = ["DB_NAME", "DB_USER", "DB_PASSWORD", "DB_HOST", "DB_PORT", "GEMINI_API_KEY"]
for var in REQUIRED_ENV_VARS:
    if not os.getenv(var):
        raise EnvironmentError(f"❌ Missing required environment variable: {var}")


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
        date: str, amount: float, trans_type: str, category: str, subcategory: Optional[str], description: str
) -> bool:
    """
    Add a new transaction to the PostgreSQL database.

    Args:
        date (str): Transaction date in ISO 8601 or DD-MM-YYYY format.
        amount (float): Transaction amount.
        trans_type (str): Type of transaction (Income/Expense).
        category (str): Transaction category.
        subcategory (Optional[str]): Transaction subcategory.
        description (str): Transaction description.

    Returns:
        bool: True if the transaction was added successfully, False otherwise.
    """
    # Ensure trans_type is capitalized for consistency with DB schema (if needed)
    trans_type = trans_type.capitalize()

    # Handle ISO 8601 and DD-MM-YYYY formats
    try:
        date_obj = isoparse(date).date()  # Parse ISO 8601 format
    except ValueError:
        date_obj = datetime.strptime(date, "%d-%m-%Y").date()  # Fallback to DD-MM-YYYY format

    query = """
    INSERT INTO transactions (date, amount, type, category, subcategory, description)
    VALUES (%s, %s, %s, %s, %s, %s)
    """
    params = (date_obj, amount, trans_type, category, subcategory, description)
    log.info(
        f"Attempting to add transaction to DB: Date={date_obj}, Amount={amount}, Type={trans_type}, Category={category}")
    return execute_query(query, params)


# Define the schema for the desired output from Gemini
class TransactionDetails(TypedDict):
    date: str
    amount: float
    type: Literal["income", "expense"]  # Ensure case matches what you expect (e.g., 'income' or 'Income')
    category: str
    subcategory: Optional[str]
    description: str


def get_gemini_model() -> Any:
    """
    Configure and return the Gemini AI model with a specific response schema for transactions.

    Returns:
        Any: Gemini AI model object.
    """
    try:
        import google.generativeai as genai
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))  # type: ignore

        # Define the JSON schema for transaction details
        transaction_schema = {
            "type": "object",
            "properties": {
                "date": {"type": "string", "format": "date-time",
                         "description": "Transaction date in YYYY-MM-DD format. Resolve relative dates like 'yesterday' to today's date if possible, otherwise use the mentioned date."},
                "amount": {"type": "number", "description": "The numerical amount of the transaction."},
                "type": {"type": "string", "enum": ["income", "expense"],
                         "description": "Type of transaction: 'income' or 'expense'."},
                "category": {"type": "string",
                             "description": "Primary category of the transaction (e.g., Groceries, Salary, Utilities, Rent, Transport, Entertainment, Shopping, Health, Education, Travel, Food & Dining)."},
                "subcategory": {"type": "string", "nullable": True,
                                "description": "More specific subcategory if available (e.g., Coffee, Petrol, Movies). Can be null."},
                "description": {"type": "string", "description": "A brief description of the transaction."},
            },
            "required": ["date", "amount", "type", "category", "description"]
        }

        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            generation_config={
                "response_mime_type": "application/json",
                "response_schema": transaction_schema
            }
        )
        log.info("🤖 Gemini AI configured successfully with transaction schema")
        return model
    except ImportError as e:
        log.error(f"❌ Failed to import Gemini AI library: {str(e)}")
        raise
    except Exception as e:
        log.error(f"❌ Failed to configure Gemini AI: {str(e)}")
        raise


def parse_transaction_details(response_text: str) -> Dict[str, Any]:
    """
    Parse transaction details from the Gemini AI JSON response.

    Args:
        response_text (str): JSON string response from Gemini AI.

    Returns:
        Dict[str, Any]: Parsed transaction details.
    """
    try:
        # Gemini is instructed to return valid JSON, so directly parse it
        details: TransactionDetails = json.loads(response_text)

        # Basic validation/defaulting for optional fields if they come back as None
        details['subcategory'] = details.get('subcategory') or None
        details['description'] = details.get('description') or f"{details['type']} for {details['amount']}"

        log.info(f"✅ Successfully parsed Gemini response: {details}")
        return details
    except json.JSONDecodeError as e:
        log.error(f"❌ Failed to parse JSON from Gemini response: {e}. Response was: {response_text}")
        raise ValueError("Invalid JSON response from Gemini AI.")
    except KeyError as e:
        log.error(f"❌ Missing expected key in Gemini response: {e}. Response was: {response_text}")
        raise ValueError(f"Incomplete transaction details from Gemini AI: missing {e}")
    except Exception as e:
        log.error(f"❌ An unexpected error occurred during parsing: {str(e)}")
        raise


def handle_transaction_with_gemini(utterance: str) -> bool:
    """
    Use Gemini AI to process the utterance, extract transaction details, and store them in PostgreSQL.

    Args:
        utterance (str): Input text containing transaction details.

    Returns:
        bool: True if the transaction is successfully stored, False otherwise.
    """
    try:
        # Get the Gemini AI model
        model = get_gemini_model()

        # Define the prompt for Gemini
        # We explicitly ask for the current date to help Gemini resolve relative dates
        current_date = datetime.now().strftime("%Y-%m-%d")
        prompt = f"""
        Extract the following transaction details from the user's utterance and return them in JSON format according to the provided schema.
        If a date is relative (e.g., "yesterday", "tomorrow", "next week"), resolve it based on the current date: {current_date}.
        If an explicit category is not given, infer one from the description (e.g., "groceries" -> "Food & Dining").
        Ensure the 'type' is either 'income' or 'expense'.

        Utterance: "{utterance}"
        """

        log.info(f"Sending prompt to Gemini: {prompt}")
        response = model.generate_content(prompt)

        # Accessing response.text directly as we configured for JSON MIME type
        gemini_response_text = response.text
        log.info(f"🤖 Gemini AI raw response: {gemini_response_text}")

        # Parse the response to extract transaction details
        transaction_details = parse_transaction_details(gemini_response_text)

        # Store transaction in the database
        return add_transaction_to_db(
            date=transaction_details["date"],
            amount=transaction_details["amount"],
            trans_type=transaction_details["type"],
            category=transaction_details["category"],
            subcategory=transaction_details["subcategory"],
            description=transaction_details["description"],
        )
    except Exception as e:
        log.error(f"❌ Failed to handle transaction with Gemini AI: {str(e)}")
        return False


# --- Database Initialization (for testing) ---
# def setup_database_schema():
#     """
#     Creates the 'transactions' table if it doesn't exist.
#     Call this once before running the main application.
#     """
#     conn = None
#     try:
#         conn = get_db_connection()
#         cursor = conn.cursor()
#         cursor.execute("""
#             CREATE TABLE IF NOT EXISTS transactions (
#                 id SERIAL PRIMARY KEY,
#                 date DATE NOT NULL,
#                 amount NUMERIC(10, 2) NOT NULL,
#                 type VARCHAR(50) NOT NULL,
#                 category VARCHAR(100) NOT NULL,
#                 subcategory VARCHAR(100),
#                 description TEXT
#             );
#         """)
#         conn.commit()
#         log.info("✅ 'transactions' table checked/created successfully.")
#     except psycopg2.Error as e:
#         log.error(f"❌ Failed to set up database schema: {str(e)}")
#     finally:
#         if conn:
#             conn.close()

if __name__ == "__main__":
    # Ensure your PostgreSQL database is running and accessible
    # and the environment variables (DB_NAME, DB_USER, etc.) are set correctly.

    # 1. First, create the table if it doesn't exist (run once)
    # setup_database_schema()

    # 2. Now, test with various utterances
    print("\n--- Test Case 1 ---")
    handle_transaction_with_gemini("Add a transaction of 5000 for entertainment on 2025-10-01")