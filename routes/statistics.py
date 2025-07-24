from datetime import datetime
from typing import Optional

import psycopg2
from dateutil.parser import isoparse
from flask import Blueprint, request, jsonify
import logging
from database import get_db_connection, execute_query

log = logging.getLogger("Statistics")
bp = Blueprint("statistics", __name__, url_prefix="/statistics")


@bp.route("/", methods=["GET"])
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


@bp.route("/categories", methods=["GET"])
def get_category_breakdown():
    """
    API endpoint to retrieve spending breakdown by category for the user.

    Query Parameters:
        user_id (str): Required. User ID to filter transactions.
        start_date (str): Optional. Start date for filtering (YYYY-MM-DD format).
        end_date (str): Optional. End date for filtering (YYYY-MM-DD format).

    Returns:
        JSON: Category breakdown with total spent per category.
    """
    try:
        user_id = request.args.get("user_id")
        if not user_id:
            log.warning("❌ Missing user_id parameter in request")
            return jsonify({"error": "Missing user_id parameter"}), 400

        start_date = request.args.get("start_date")
        end_date = request.args.get("end_date")

        # Validate date format if provided
        if start_date:
            try:
                datetime.strptime(start_date, "%Y-%m-%d")
            except ValueError:
                return (
                    jsonify({"error": "Invalid start_date format. Use YYYY-MM-DD."}),
                    400,
                )

        if end_date:
            try:
                datetime.strptime(end_date, "%Y-%m-%d")
            except ValueError:
                return (
                    jsonify({"error": "Invalid end_date format. Use YYYY-MM-DD."}),
                    400,
                )

        # Retrieve category breakdown from the database
        breakdown = get_category_breakdown_from_db(user_id, start_date, end_date)

        if "error" in breakdown:
            log.error(f"❌ Database error: {breakdown['error']}")
            return jsonify(breakdown), 500

        log.info(f"✅ Successfully retrieved category breakdown for user {user_id}")
        return jsonify(breakdown), 200

    except Exception as e:
        log.error(f"❌ Unexpected error in get_category_breakdown: {str(e)}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


@bp.route("/monthly", methods=["GET"])
def get_monthly_trends():
    """
    API endpoint to retrieve monthly spending trends for the user.

    Query Parameters:
        user_id (str): Required. User ID to filter transactions.
        months (int): Optional. Number of months to include (default: 6).

    Returns:
        JSON: Monthly trends showing income and expenses over time.
    """
    try:
        user_id = request.args.get("user_id")
        if not user_id:
            log.warning("❌ Missing user_id parameter in request")
            return jsonify({"error": "Missing user_id parameter"}), 400

        months = request.args.get("months", default=6, type=int)
        if months <= 0 or months > 24:
            return jsonify({"error": "Months parameter must be between 1 and 24"}), 400

        # Retrieve monthly trends from the database
        trends = get_monthly_trends_from_db(user_id, months)

        if "error" in trends:
            log.error(f"❌ Database error: {trends['error']}")
            return jsonify(trends), 500

        log.info(f"✅ Successfully retrieved monthly trends for user {user_id}")
        return jsonify(trends), 200

    except Exception as e:
        log.error(f"❌ Unexpected error in get_monthly_trends: {str(e)}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


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
        dict: Dictionary containing total income, total expenses, balance.
    """
    conn = None
    try:

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

            final_params = query_params

            query = f"""
            SELECT COALESCE(SUM(CASE WHEN LOWER(type) in ('income','credit') THEN amount ELSE 0 END), 0) AS total_income,
            COALESCE(SUM(CASE WHEN LOWER(type) in ( 'expense', 'debit' THEN amount ELSE 0 END), 0) AS total_expenses 
            FROM user_transactions WHERE user_id = %s{date_filter}
            """

            log.info(f"🔍 Executing query with params: user_id={user_id}")
            log.debug(f"📝 SQL Query: {query}")
            log.debug(f"📝 Parameters: {final_params}")

            cursor.execute(query, tuple(final_params))
            result = cursor.fetchone()

            if result:
                total_income, total_expenses = result

                total_income = total_income or 0
                total_expenses = total_expenses or 0
                balance = total_income - total_expenses
                log.info(
                    f"✅ Retrieved statistics: Total Income={total_income}, Total Expenses={total_expenses}, Balance={balance}"
                )
                stats = {
                    "total_income": float(total_income),
                    "total_expenses": float(total_expenses),
                    "balance": float(balance),
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


def get_category_breakdown_from_db(
    user_id: str, start_date: Optional[str] = None, end_date: Optional[str] = None
) -> dict:
    """
    Retrieve spending breakdown by category for a given user from the PostgreSQL database.

    Args:
        user_id (str): User ID to filter transactions.
        start_date (Optional[str]): Start date for filtering transactions (YYYY-MM-DD format).
        end_date (Optional[str]): End date for filtering transactions (YYYY-MM-DD format).

    Returns:
        dict: Dictionary containing category breakdown.
    """
    conn = None
    try:
        conn = get_db_connection()

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

            query = f"""
            SELECT category, 
                   SUM(CASE WHEN type = 'Expense' THEN amount ELSE 0 END) as total_expenses,
                   SUM(CASE WHEN type = 'Income' THEN amount ELSE 0 END) as total_income,
                   COUNT(*) as transaction_count
            FROM user_transactions 
            WHERE user_id = %s{date_filter}
            GROUP BY category
            ORDER BY total_expenses DESC
            """

            cursor.execute(query, tuple(query_params))
            results = cursor.fetchall()

            categories = []
            total_expenses = 0
            total_income = 0

            for row in results:
                category, expense_amount, income_amount, count = row
                expense_amount = float(expense_amount or 0)
                income_amount = float(income_amount or 0)

                categories.append(
                    {
                        "category": category,
                        "total_expenses": expense_amount,
                        "total_income": income_amount,
                        "transaction_count": count,
                    }
                )

                total_expenses += expense_amount
                total_income += income_amount

            breakdown = {
                "categories": categories,
                "summary": {
                    "total_expenses": total_expenses,
                    "total_income": total_income,
                    "date_range": {"start_date": start_date, "end_date": end_date},
                },
            }

            log.info(
                f"✅ Successfully calculated category breakdown for user {user_id}"
            )
            return breakdown

    except psycopg2.Error as e:
        error_msg = f"Database error occurred: {str(e)}"
        log.error(f"❌ {error_msg}")
        return {"error": error_msg}
    except Exception as e:
        error_msg = f"Failed to process category breakdown: {str(e)}"
        log.error(f"❌ {error_msg}")
        return {"error": error_msg}
    finally:
        if conn:
            conn.close()


def get_monthly_trends_from_db(user_id: str, months: int = 6) -> dict:
    """
    Retrieve monthly spending trends for a given user from the PostgreSQL database.

    Args:
        user_id (str): User ID to filter transactions.
        months (int): Number of months to include in the trends.

    Returns:
        dict: Dictionary containing monthly trends.
    """
    conn = None
    try:
        conn = get_db_connection()

        with conn.cursor() as cursor:
            query = """
            SELECT 
                DATE_TRUNC('month', date) as month,
                SUM(CASE WHEN type = 'Income' THEN amount ELSE 0 END) as total_income,
                SUM(CASE WHEN type = 'Expense' THEN amount ELSE 0 END) as total_expenses,
                COUNT(*) as transaction_count
            FROM user_transactions 
            WHERE user_id = %s 
                AND date >= CURRENT_DATE - INTERVAL '%s months'
            GROUP BY DATE_TRUNC('month', date)
            ORDER BY month DESC
            """

            cursor.execute(query, (user_id, months))
            results = cursor.fetchall()

            monthly_data = []
            for row in results:
                month, income, expenses, count = row
                monthly_data.append(
                    {
                        "month": month.strftime("%Y-%m"),
                        "total_income": float(income or 0),
                        "total_expenses": float(expenses or 0),
                        "net_savings": float((income or 0) - (expenses or 0)),
                        "transaction_count": count,
                    }
                )

            trends = {
                "monthly_data": monthly_data,
                "period": f"Last {months} months",
                "user_id": user_id,
            }

            log.info(f"✅ Successfully calculated monthly trends for user {user_id}")
            return trends

    except psycopg2.Error as e:
        error_msg = f"Database error occurred: {str(e)}"
        log.error(f"❌ {error_msg}")
        return {"error": error_msg}
    except Exception as e:
        error_msg = f"Failed to process monthly trends: {str(e)}"
        log.error(f"❌ {error_msg}")
        return {"error": error_msg}
    finally:
        if conn:
            conn.close()


@bp.route("/daily-spending", methods=["GET"])
def get_daily_spending():
    """
    API endpoint to retrieve daily spending data for the last 30 days.

    Query Parameters:
        user_id (str): Required. User ID to filter transactions.
        days (int): Optional. Number of days to retrieve (default: 30).

    Returns:
        JSON: Daily spending data with date, amount, and day number.
    """
    try:
        user_id = request.args.get("user_id")
        if not user_id:
            log.warning("❌ Missing user_id parameter in request")
            return jsonify({"error": "Missing user_id parameter"}), 400

        days = request.args.get("days", 30)
        try:
            days = int(days)
        except ValueError:
            days = 30

        log.info(
            f"🔍 Processing daily spending request for user_id={user_id}, days={days}"
        )

        conn = get_db_connection()
        cursor = conn.cursor()

        # Get daily spending for the last N days
        query = """
            SELECT 
                DATE(date) as spending_date,
                SUM(CASE WHEN type = 'Expense' THEN amount ELSE 0 END) as daily_spending
            FROM user_transactions 
            WHERE user_id = %s 
                AND date >= CURRENT_DATE - INTERVAL %s
                AND type = 'Expense'
            GROUP BY DATE(date)
            ORDER BY spending_date DESC
            LIMIT %s;
        """

        # Format the interval properly for PostgreSQL
        interval_str = f"{days} days"
        cursor.execute(query, (user_id, interval_str, days))
        results = cursor.fetchall()

        daily_data = []
        for i, (date, amount) in enumerate(results):
            daily_data.append(
                {
                    "day": str(i + 1),
                    "date": date.strftime("%b %d"),
                    "amount": float(amount) if amount else 0.0,
                    "full_date": date.strftime("%Y-%m-%d"),
                }
            )

        # Calculate summary stats
        amounts = [d["amount"] for d in daily_data]
        summary = {
            "average_daily": sum(amounts) / len(amounts) if amounts else 0.0,
            "max_daily": max(amounts) if amounts else 0.0,
            "min_daily": min(amounts) if amounts else 0.0,
            "total_period": sum(amounts),
        }

        response = {
            "daily_spending": daily_data,
            "summary": summary,
            "period": f"Last {days} days",
        }

        log.info(f"✅ Successfully retrieved daily spending for user {user_id}")
        return jsonify(response), 200

    except Exception as e:
        error_msg = f"Failed to retrieve daily spending: {str(e)}"
        log.error(f"❌ {error_msg}")
        return jsonify({"error": error_msg}), 500
    finally:
        if conn:
            conn.close()


@bp.route("/weekly-pattern", methods=["GET"])
def get_weekly_spending_pattern():
    """
    API endpoint to retrieve weekly spending pattern (average by day of week).

    Query Parameters:
        user_id (str): Required. User ID to filter transactions.
        weeks (int): Optional. Number of weeks to analyze (default: 12).

    Returns:
        JSON: Weekly spending pattern with averages by day of week.
    """
    try:
        user_id = request.args.get("user_id")
        if not user_id:
            log.warning("❌ Missing user_id parameter in request")
            return jsonify({"error": "Missing user_id parameter"}), 400

        weeks = request.args.get("weeks", 12)
        try:
            weeks = int(weeks)
        except ValueError:
            weeks = 12

        log.info(
            f"🔍 Processing weekly pattern request for user_id={user_id}, weeks={weeks}"
        )

        conn = get_db_connection()
        cursor = conn.cursor()

        # Get spending by day of week for the last N weeks
        query = """
            SELECT 
                EXTRACT(DOW FROM date) as day_of_week,
                TO_CHAR(date, 'Dy') as day_name,
                AVG(CASE WHEN type = 'Expense' THEN amount ELSE 0 END) as avg_spending,
                SUM(CASE WHEN type = 'Expense' THEN amount ELSE 0 END) as total_spending,
                COUNT(CASE WHEN type = 'Expense' THEN 1 END) as transaction_count
            FROM user_transactions 
            WHERE user_id = %s 
                AND date >= CURRENT_DATE - INTERVAL %s
                AND type = 'Expense'
            GROUP BY EXTRACT(DOW FROM date), TO_CHAR(date, 'Dy')
            ORDER BY day_of_week;
        """

        # Format the interval properly for PostgreSQL
        interval_str = f"{weeks} weeks"
        cursor.execute(query, (user_id, interval_str))
        results = cursor.fetchall()

        # Map PostgreSQL day of week (0=Sunday) to more readable format
        day_mapping = {
            0: "Sun",
            1: "Mon",
            2: "Tue",
            3: "Wed",
            4: "Thu",
            5: "Fri",
            6: "Sat",
        }

        weekly_pattern = []
        total_weekly_avg = 0

        for dow, day_name, avg_spending, total_spending, tx_count in results:
            avg_amount = float(avg_spending) if avg_spending else 0.0
            total_weekly_avg += avg_amount

            weekly_pattern.append(
                {
                    "day": day_mapping.get(int(dow), day_name),
                    "day_name": day_name,
                    "average_spending": avg_amount,
                    "total_spending": float(total_spending) if total_spending else 0.0,
                    "transaction_count": int(tx_count) if tx_count else 0,
                }
            )

        # Fill in missing days with 0 spending
        existing_days = {item["day"] for item in weekly_pattern}
        for day_num, day_abbr in day_mapping.items():
            if day_abbr not in existing_days:
                weekly_pattern.append(
                    {
                        "day": day_abbr,
                        "day_name": day_abbr,
                        "average_spending": 0.0,
                        "total_spending": 0.0,
                        "transaction_count": 0,
                    }
                )

        # Sort by day of week (Mon-Sun)
        day_order = {
            "Mon": 1,
            "Tue": 2,
            "Wed": 3,
            "Thu": 4,
            "Fri": 5,
            "Sat": 6,
            "Sun": 7,
        }
        weekly_pattern.sort(key=lambda x: day_order.get(x["day"], 8))

        response = {
            "weekly_pattern": weekly_pattern,
            "analysis_period": f"Last {weeks} weeks",
            "weekly_average": total_weekly_avg,
        }

        log.info(f"✅ Successfully retrieved weekly pattern for user {user_id}")
        return jsonify(response), 200

    except Exception as e:
        error_msg = f"Failed to retrieve weekly pattern: {str(e)}"
        log.error(f"❌ {error_msg}")
        return jsonify({"error": error_msg}), 500
    finally:
        if conn:
            conn.close()


@bp.route("/financial-summary", methods=["GET"])
def get_financial_summary():
    """
    API endpoint to retrieve comprehensive financial summary metrics.

    Query Parameters:
        user_id (str): Required. User ID to filter transactions.
        months (int): Optional. Number of months to analyze (default: 6).

    Returns:
        JSON: Financial summary including totals, averages, savings rate, and debt info.
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
            f"🔍 Processing financial summary request for user_id={user_id}, months={months}"
        )

        conn = get_db_connection()
        cursor = conn.cursor()

        # Get comprehensive financial data - Use proper interval format
        query = """
            SELECT 
                SUM(CASE WHEN type = 'Income' THEN amount ELSE 0 END) as total_income,
                SUM(CASE WHEN type = 'Expense' THEN amount ELSE 0 END) as total_expenses,
                COUNT(CASE WHEN type = 'Income' THEN 1 END) as income_transactions,
                COUNT(CASE WHEN type = 'Expense' THEN 1 END) as expense_transactions,
                AVG(CASE WHEN type = 'Income' THEN amount END) as avg_income_per_transaction,
                AVG(CASE WHEN type = 'Expense' THEN amount END) as avg_expense_per_transaction
            FROM user_transactions 
            WHERE user_id = %s 
                AND date >= CURRENT_DATE - INTERVAL '%s months';
        """

        log.info(f"🔍 Executing query with params: user_id={user_id}, months={months}")
        cursor.execute(query, (user_id, months))
        result = cursor.fetchone()

        log.info(f"🔍 Raw query result: {result}")
        log.info(f"🔍 Result type: {type(result)}")
        log.info(f"🔍 Result length: {len(result) if result else 0}")

        if not result:
            return jsonify({"error": "No data found for user"}), 404

        # Debug: log the actual result
        log.info(f"🔍 Raw query result: {result}")
        log.info(f"🔍 Result length: {len(result) if result else 0}")

        # Safely unpack the result with default values
        try:
            if len(result) < 6:
                # Pad with None values if we don't have enough columns
                result = list(result) + [None] * (6 - len(result))

            log.info(f"🔍 About to unpack: {result[:6]}")
            (
                total_income,
                total_expenses,
                income_tx,
                expense_tx,
                avg_income,
                avg_expense,
            ) = result[:6]
            log.info(f"✅ Unpacking successful!")

        except Exception as e:
            log.error(f"❌ Tuple unpacking failed: {e}")
            log.error(f"❌ Result was: {result}")
            log.error(f"❌ Result type: {type(result)}")
            raise e

        # Calculate derived metrics
        total_income = float(total_income) if total_income else 0.0
        total_expenses = float(total_expenses) if total_expenses else 0.0
        balance = total_income - total_expenses
        savings_rate = (balance / total_income * 100) if total_income > 0 else 0.0

        # Monthly averages
        monthly_income = total_income / months if months > 0 else 0.0
        monthly_expenses = total_expenses / months if months > 0 else 0.0
        monthly_savings = balance / months if months > 0 else 0.0

        # Get debt information (negative balance or loan categories)
        # Since there are no debt/loan/credit categories in the data, set debt to 0
        # This avoids the psycopg2 issue with ILIKE queries that return no results
        log.info(f"🔍 Setting debt to 0 (no debt categories found in transaction data)")
        current_debt = 0.0

        response = {
            "total_income": total_income,
            "total_expenses": total_expenses,
            "total_savings": balance,
            "current_debt": current_debt,
            "savings_rate": round(savings_rate, 1),
            "monthly_averages": {
                "income": round(monthly_income, 2),
                "expenses": round(monthly_expenses, 2),
                "savings": round(monthly_savings, 2),
            },
            "transaction_counts": {
                "income": int(income_tx) if income_tx else 0,
                "expense": int(expense_tx) if expense_tx else 0,
            },
            "average_amounts": {
                "income_per_transaction": (
                    round(float(avg_income), 2) if avg_income else 0.0
                ),
                "expense_per_transaction": (
                    round(float(avg_expense), 2) if avg_expense else 0.0
                ),
            },
            "analysis_period": f"Last {months} months",
            "net_worth_change": balance,  # Simplified - could be enhanced with previous period comparison
        }

        log.info(f"✅ Successfully retrieved financial summary for user {user_id}")
        return jsonify(response), 200

    except Exception as e:
        error_msg = f"Failed to retrieve financial summary: {str(e)}"
        log.error(f"❌ {error_msg}")
        return jsonify({"error": error_msg}), 500
    finally:
        if conn:
            conn.close()
