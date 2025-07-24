import json
import os
from datetime import datetime

from flask import Blueprint, request, jsonify
import logging
from fuzzywuzzy import fuzz
from routes.decorators import retry_with_exponential_backoff
from dotenv import load_dotenv
import google.generativeai as genai

bp = Blueprint("chatbot", __name__, url_prefix="/chatbot")
log = logging.getLogger("Chatbot")
logging.basicConfig(level=logging.INFO)

# --- Gemini API Configuration ---
load_dotenv()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set. Please set it before running the Flask app.")

genai.configure(api_key=GEMINI_API_KEY)

GEMINI_MODEL = "gemini-2.5-flash-preview-05-20"
try:
    model = genai.GenerativeModel(GEMINI_MODEL)
except Exception as e:
    log.error(f"Error initializing Gemini model: {e}")
    model = None

# --- Dynamic Keyword Configuration ---
TRANSACTION_KEYWORDS = ["spend", "spent", "paid", "purchased", "used", "allocated", "record", "log", "bought", "buy"]
FINANCIAL_GOALS_KEYWORDS = ["save", "invest", "budget", "retirement", "goal", "plan", "wealth", "debt", "loan",
                            "mortgage"]
FUZZY_MATCH_THRESHOLD = 80
GOAL_CREATION_KEYWORDS = ["create goal", "set goal", "goal setting", "new goal", "plan goal"]


# --- API Endpoint for Chatbot Interactions ---
@bp.route('/chatbot-interact', methods=['POST'])
@retry_with_exponential_backoff(max_retries=3)
async def chatbot_interact():
    if model is None:
        return jsonify({"error": "Gemini model not initialized. Check server logs."}), 500

    data = request.get_json()
    user_utterance = data.get('utterance')

    if not user_utterance:
        return jsonify({"error": "No utterance provided"}), 400

    user_utterance_lower = user_utterance.lower()

    is_transaction_related = any(
        fuzz.partial_ratio(keyword, user_utterance_lower) >= FUZZY_MATCH_THRESHOLD
        for keyword in TRANSACTION_KEYWORDS
    )

    is_financial_goals_related = any(
        fuzz.partial_ratio(keyword, user_utterance_lower) >= FUZZY_MATCH_THRESHOLD
        for keyword in FINANCIAL_GOALS_KEYWORDS
    )

    is_goal_creation_related = any(
        fuzz.partial_ratio(keyword, user_utterance_lower) >= FUZZY_MATCH_THRESHOLD
        for keyword in GOAL_CREATION_KEYWORDS
    )

    if is_goal_creation_related:
        log.info(f"User utterance '{user_utterance}' indicates goal creation intent.")
        return jsonify({
            "intent": "create_goal",
            "data": {
                "form": {
                    "fields": [
                        {"name": "goal_name", "type": "text", "label": "Goal Name"},
                        {"name": "target_amount", "type": "number", "label": "Target Amount"},
                        {"name": "deadline", "type": "date", "label": "Deadline"},
                        {"name": "priority", "type": "select", "label": "Priority",
                         "options": ["High", "Medium", "Low"]}
                    ]
                }
            },
            "response_text": "Please fill out the form to set your goal."
        }), 200

    if is_transaction_related:
        log.info(f"User utterance '{user_utterance}' contains transaction keywords. Calling internal parsing logic.")
        try:
            parsed_response_from_gemini = await parse_transaction_logic(user_utterance)
            if parsed_response_from_gemini.get('intent') == "record_expense_or_income":
                data_dict = parsed_response_from_gemini.get('data', {})
                amount_val = data_dict.get('amount', 'an amount')
                type_val = data_dict.get('type', 'transaction')
                category_val = data_dict.get('category', 'a category')
                date_val = data_dict.get('date', 'today')
                parsed_response_from_gemini[
                    'response_text'] = f"Okay, recorded your {amount_val} {type_val} for {category_val} on {date_val}."

            return jsonify(parsed_response_from_gemini), 200
        except Exception as e:
            log.error(f"Error during internal parsing logic: {e}")
            return jsonify({"error": "An internal server error occurred while parsing the transaction."}), 500

    elif is_financial_goals_related:
        log.info(f"User utterance '{user_utterance}' contains financial goals/advice keywords. Generating advice.")
        try:
            financial_goals_prompt = f"""
            You are a personal financial assistant. The user is asking about financial goals or advice.
            Provide actionable and concise advice based on the user's query.

            User Utterance: "{user_utterance}"

            Return your response in the following JSON format:
            {{
                "intent": "financial_goals_or_advice",
                "data": null,
                "response_text": "Your generated advice or response."
            }}
            """

            generation_config = {
                "response_mime_type": "application/json",
                "temperature": 0.7,
            }

            response = await model.generate_content_async(
                contents=[{"role": "user", "parts": [{"text": financial_goals_prompt}]}],
                generation_config=generation_config
            )

            gemini_raw_text = response.text.strip()
            log.info(f"🤖 Gemini API raw financial goals response: {gemini_raw_text}")

            try:
                parsed_gemini_response = json.loads(gemini_raw_text)
            except json.JSONDecodeError as e:
                log.error(f"Gemini did not return valid JSON for financial goals: {e} - Raw text: {gemini_raw_text}")
                return jsonify({"error": "Failed to interpret Gemini's financial goals advice (invalid JSON format).",
                                "raw_gemini_output": gemini_raw_text}), 500

            if 'response_text' not in parsed_gemini_response:
                parsed_gemini_response['response_text'] = "I'm sorry, I couldn't generate a coherent response."

            return jsonify(parsed_gemini_response), 200

        except Exception as e:
            log.error(f"An error occurred during Gemini API call for financial goals: {e}")
            return jsonify({"error": f"An internal server error occurred: {e}"}), 500

    log.info(f"User utterance '{user_utterance}' does not match any specific category. Getting general advice.")
    try:
        general_advice_prompt = f"""
        You are a helpful personal financial assistant. Based on the user's request, provide a concise and relevant response.
        If the user is asking a general financial question (e.g., "how to save money?", "what's a good investment?"), generate advice.
        If it's small talk (greet, goodbye, mood), respond appropriately.
        If it's unclear, ask for clarification.

        Return your response in the following JSON format:
        {{
            "intent": "get_financial_advice" or "small_talk" or "unclear",
            "data": null,
            "response_text": "Your generated text response."
        }}
        User Utterance: "{user_utterance}"
        """

        generation_config = {
            "response_mime_type": "application/json",
            "temperature": 0.7,
        }

        response = await model.generate_content_async(
            contents=[{"role": "user", "parts": [{"text": general_advice_prompt}]}],
            generation_config=generation_config
        )

        gemini_raw_text = response.text.strip()
        log.info(f"🤖 Gemini API raw general advice response: {gemini_raw_text}")

        try:
            parsed_gemini_response = json.loads(gemini_raw_text)
        except json.JSONDecodeError as e:
            log.error(f"Gemini did not return valid JSON for general advice: {e} - Raw text: {gemini_raw_text}")
            return jsonify({"error": "Failed to interpret Gemini's general advice (invalid JSON format).",
                            "raw_gemini_output": gemini_raw_text}), 500

        if 'response_text' not in parsed_gemini_response:
            parsed_gemini_response['response_text'] = "I'm sorry, I couldn't generate a coherent response."

        return jsonify(parsed_gemini_response), 200

    except Exception as e:
        log.error(f"An error occurred during Gemini API call for general advice: {e}")
        return jsonify({"error": f"An internal server error occurred: {e}"}), 500


async def parse_transaction_logic(utterance: str):
    current_date = datetime.now().strftime("%Y-%m-%d")

    prompt = f"""
    Extract the transaction details from the following user utterance and return them as a JSON object.
    If a field cannot be extracted, set it to null.
    The details should include:
    - Date (YYYY-MM-DD format, resolve relative dates like 'today', 'yesterday', 'last Friday' based on {current_date})
    - Amount (float, always positive).
    - Type (income or expense). Infer based on context if not explicit.
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

    User Utterance:
    {utterance}
    """
    try:
        log.info(f"🔍 Sending parsing prompt to Gemini API for utterance: '{utterance}'")
        if model is None:
            raise Exception("Gemini model not initialized for parsing.")

        generation_config = {
            "response_mime_type": "application/json",
            "temperature": 0.1,
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
                f"Gemini did not return valid JSON for parsing even after stripping: {transaction_details_str} - Error: {e}")
            return {
                "intent": "unclear",
                "data": None,
                "response_text": "I received a response from Gemini, but couldn't understand its format for the transaction. Could you try rephrasing?"
            }

        if 'amount' in parsed_json and isinstance(parsed_json['amount'], (int, float)):
            parsed_json['amount'] = abs(parsed_json['amount'])
        else:
            log.warning(f"Amount not found or invalid type in parsed JSON: {parsed_json}")
            parsed_json['amount'] = None

        parsed_json['intent'] = "record_expense_or_income"

        return parsed_json
    except Exception as e:
        log.error(f"❌ Error during transaction parsing logic: {e}")
        raise