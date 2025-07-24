from flask import Blueprint, request, jsonify
import logging

bp = Blueprint("alerts", __name__, url_prefix="/alerts")
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("Alerts")

@bp.route('/big_purchase', methods=['POST'])
def alert_big_purchase():
    """
    Extracts the purchase amount from the question, compares it with the threshold,
    and returns an alert if the amount is below the threshold.
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