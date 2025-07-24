from flask import Blueprint, request, jsonify
import logging

bp = Blueprint("financial_score", __name__, url_prefix="/financial_score")
log = logging.getLogger("FinancialScore")
logging.basicConfig(level=logging.INFO)


@bp.route('/calculate', methods=['POST'])
def financial_health_score():
    """
    Calculates the financial health score for a user.
    """
    data = request.json
    income = data.get("income")
    expense = data.get("expense")
    savings = data.get("savings")
    discretionary_ratio = data.get("discretionary_ratio")
    ai_risk_score = data.get("ai_risk_score", 5)

    if None in [income, expense, savings, discretionary_ratio]:
        return jsonify({"error": "Missing required fields: income, expense, savings, discretionary_ratio"}), 400

    # Add logging to debug
    log.info(f"Received data: {data}")

    weights = data.get("weights", {
        "expense": 0.25,
        "savings": 0.25,
        "discretionary": 0.2,
        "net_flow": 0.1,
        "ai_risk": 0.2
    })

    expense_ratio = expense / income if income > 0 else 1
    expense_score = (
        30 if expense_ratio <= 0.5 else
        20 if expense_ratio <= 0.7 else
        10 if expense_ratio <= 0.9 else
        0
    )

    savings_rate = savings / income if income > 0 else -1
    savings_score = (
        30 if savings_rate >= 0.3 else
        20 if savings_rate >= 0.2 else
        10 if savings_rate >= 0.1 else
        0
    )

    discretionary_score = (
        20 if discretionary_ratio <= 0.1 else
        10 if discretionary_ratio <= 0.2 else
        5 if discretionary_ratio <= 0.3 else
        0
    )

    net_flow_score = 10 if savings >= 0 else 0
    ai_risk_score = max(0, min(ai_risk_score, 10))

    final_score = round(
        (expense_score * weights["expense"] +
         savings_score * weights["savings"] +
         discretionary_score * weights["discretionary"] +
         net_flow_score * weights["net_flow"] +
         ai_risk_score * 3 * weights["ai_risk"]), 2
    )

    classification = (
        "🌟 Excellent" if final_score >= 85 else
        "👍 Good" if final_score >= 70 else
        "🟡 Moderate" if final_score >= 50 else
        "⚠️ Needs Attention"
    )

    response = {
        "final_score": final_score,
        "classification": classification,
        "breakdown": {
            "expense_score": expense_score,
            "savings_score": savings_score,
            "discretionary_score": discretionary_score,
            "net_flow_score": net_flow_score,
            "ai_risk_score": ai_risk_score
        }
    }

    # Log the response for debugging
    log.info(f"Response: {response}")

    return jsonify(response), 200
