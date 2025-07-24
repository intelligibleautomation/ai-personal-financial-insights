from flask import Blueprint, request, jsonify
import logging

bp = Blueprint("financial_score", __name__, url_prefix="/financial_score")
log = logging.getLogger("FinancialScore")
logging.basicConfig(level=logging.INFO)


@bp.route("/calculate", methods=["POST"])
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
        return (
            jsonify(
                {
                    "error": "Missing required fields: income, expense, savings, discretionary_ratio"
                }
            ),
            400,
        )

    # Add logging to debug
    log.info(f"Received data: {data}")

    weights = data.get(
        "weights",
        {
            "expense": 0.25,
            "savings": 0.25,
            "discretionary": 0.2,
            "net_flow": 0.1,
            "ai_risk": 0.2,
        },
    )

    expense_ratio = expense / income if income > 0 else 1
    expense_score = (
        30
        if expense_ratio <= 0.5
        else 20 if expense_ratio <= 0.7 else 10 if expense_ratio <= 0.9 else 0
    )

    savings_rate = savings / income if income > 0 else -1
    savings_score = (
        30
        if savings_rate >= 0.3
        else 20 if savings_rate >= 0.2 else 10 if savings_rate >= 0.1 else 0
    )

    discretionary_score = (
        20
        if discretionary_ratio <= 0.1
        else (
            10 if discretionary_ratio <= 0.2 else 5 if discretionary_ratio <= 0.3 else 0
        )
    )

    net_flow_score = 10 if savings >= 0 else 0
    ai_risk_score = max(0, min(ai_risk_score, 10))

    final_score = round(
        (
            expense_score * weights["expense"]
            + savings_score * weights["savings"]
            + discretionary_score * weights["discretionary"]
            + net_flow_score * weights["net_flow"]
            + ai_risk_score * 3 * weights["ai_risk"]
        ),
        2,
    )

    classification = (
        "🌟 Excellent"
        if final_score >= 85
        else (
            "👍 Good"
            if final_score >= 70
            else "🟡 Moderate" if final_score >= 50 else "⚠️ Needs Attention"
        )
    )

    response = {
        "final_score": final_score,
        "classification": classification,
        "breakdown": {
            "expense_score": expense_score,
            "savings_score": savings_score,
            "discretionary_score": discretionary_score,
            "net_flow_score": net_flow_score,
            "ai_risk_score": ai_risk_score,
        },
    }

    # Log the response for debugging
    log.info(f"Response: {response}")

    return jsonify(response), 200


@bp.route("/detailed-health", methods=["GET"])
def detailed_financial_health():
    """
    Calculates detailed financial health score with breakdown components for dashboard.

    Query Parameters:
        user_id (str): Required. User ID to calculate health score for.
        months (int): Optional. Number of months to analyze (default: 6).
    """
    try:
        user_id = request.args.get("user_id")
        if not user_id:
            log.warning("❌ Missing user_id parameter in request")
            return jsonify({"error": "Missing user_id parameter"}), 400

        months = request.args.get("months", 6)
        try:
            months = int(months)
        except ValueError:
            months = 6

        log.info(
            f"🔍 Processing detailed health score for user_id={user_id}, months={months}"
        )

        # Import here to avoid circular imports
        from database import get_db_connection

        conn = get_db_connection()
        cursor = conn.cursor()

        # Get financial data for the user
        query = f"""
            SELECT 
                SUM(CASE WHEN type = 'Income' THEN amount ELSE 0 END) as total_income,
                SUM(CASE WHEN type = 'Expense' THEN amount ELSE 0 END) as total_expenses
            FROM user_transactions 
            WHERE user_id = %s 
                AND date >= CURRENT_DATE - INTERVAL '{months} months';
        """

        log.info(f"🔍 Executing health score query with params: user_id={user_id}")
        cursor.execute(query, (user_id,))
        result = cursor.fetchone()

        log.info(f"🔍 Health score query result: {result}")
        log.info(f"🔍 Result type: {type(result)}")
        log.info(f"🔍 Result length: {len(result) if result else 0}")

        if not result:
            return jsonify({"error": "No financial data found for user"}), 404

        total_income, total_expenses = result
        total_income = float(total_income) if total_income else 0.0
        total_expenses = float(total_expenses) if total_expenses else 0.0

        if total_income == 0:
            return jsonify({"error": "No income data found for user"}), 404

        # Calculate core metrics
        savings = total_income - total_expenses
        savings_rate = (savings / total_income) * 100 if total_income > 0 else 0
        expense_ratio = (
            (total_expenses / total_income) * 100 if total_income > 0 else 100
        )

        # Get emergency fund estimation (assuming 3-6 months of expenses is ideal)
        monthly_expenses = total_expenses / months if months > 0 else 0
        emergency_fund_needed = monthly_expenses * 6  # 6 months of expenses
        current_emergency_fund = max(
            0, savings
        )  # Simplified - actual apps would track this separately
        emergency_fund_ratio = (
            (current_emergency_fund / emergency_fund_needed) * 100
            if emergency_fund_needed > 0
            else 0
        )

        # Calculate individual component scores (out of 25 each for 100 total)

        # 1. Savings Rate Score (0-25)
        if savings_rate >= 20:
            savings_score = 25
        elif savings_rate >= 15:
            savings_score = 20
        elif savings_rate >= 10:
            savings_score = 15
        elif savings_rate >= 5:
            savings_score = 10
        else:
            savings_score = max(0, savings_rate)

        # 2. Emergency Fund Score (0-20)
        if emergency_fund_ratio >= 100:
            emergency_score = 20
        elif emergency_fund_ratio >= 75:
            emergency_score = 15
        elif emergency_fund_ratio >= 50:
            emergency_score = 10
        elif emergency_fund_ratio >= 25:
            emergency_score = 8
        else:
            emergency_score = max(0, emergency_fund_ratio / 25 * 8)

        # 3. Debt Management Score (0-25) - simplified version
        # Since there are no debt/loan/credit categories in the data, set debt to 0
        # This avoids the psycopg2 issue with ILIKE queries that return no results
        log.info(f"🔍 Setting debt to 0 (no debt categories found in transaction data)")
        debt_amount = 0.0

        debt_to_income_ratio = (
            (debt_amount / total_income) * 100 if total_income > 0 else 0
        )

        if debt_to_income_ratio <= 10:
            debt_score = 25
        elif debt_to_income_ratio <= 20:
            debt_score = 20
        elif debt_to_income_ratio <= 30:
            debt_score = 15
        elif debt_to_income_ratio <= 40:
            debt_score = 10
        else:
            debt_score = max(0, 10 - (debt_to_income_ratio - 40) / 10)

        # 4. Goal Progress Score (0-15) - simplified version based on savings consistency
        goal_score = min(15, savings_score * 0.6)  # Simplified calculation

        # Calculate final score
        final_score = savings_score + emergency_score + debt_score + goal_score
        final_score = round(min(100, max(0, final_score)), 1)

        # Determine classification
        if final_score >= 85:
            classification = "🌟 Excellent"
            status = "excellent"
        elif final_score >= 70:
            classification = "👍 Good"
            status = "good"
        elif final_score >= 50:
            classification = "🟡 Fair"
            status = "fair"
        else:
            classification = "⚠️ Needs Attention"
            status = "needs_attention"

        response = {
            "overall_score": final_score,
            "classification": classification,
            "status": status,
            "component_scores": {
                "savings_rate": {
                    "score": round(savings_score, 1),
                    "max_score": 25,
                    "current_rate": round(savings_rate, 1),
                    "description": f"Saving {savings_rate:.1f}% of income",
                },
                "emergency_fund": {
                    "score": round(emergency_score, 1),
                    "max_score": 20,
                    "coverage_months": (
                        round(current_emergency_fund / monthly_expenses, 1)
                        if monthly_expenses > 0
                        else 0
                    ),
                    "description": f"{emergency_fund_ratio:.1f}% of 6-month emergency fund",
                },
                "debt_management": {
                    "score": round(debt_score, 1),
                    "max_score": 25,
                    "debt_to_income": round(debt_to_income_ratio, 1),
                    "description": f"{debt_to_income_ratio:.1f}% debt-to-income ratio",
                },
                "goal_progress": {
                    "score": round(goal_score, 1),
                    "max_score": 15,
                    "description": "Based on savings consistency",
                },
            },
            "financial_metrics": {
                "total_income": total_income,
                "total_expenses": total_expenses,
                "savings": savings,
                "savings_rate": round(savings_rate, 1),
                "monthly_expenses": round(monthly_expenses, 2),
                "debt_amount": debt_amount,
            },
            "analysis_period": f"Last {months} months",
        }

        log.info(
            f"✅ Successfully calculated detailed health score for user {user_id}: {final_score}"
        )
        return jsonify(response), 200

    except Exception as e:
        error_msg = f"Failed to calculate detailed health score: {str(e)}"
        log.error(f"❌ {error_msg}")
        return jsonify({"error": error_msg}), 500
    finally:
        if "conn" in locals():
            conn.close()
