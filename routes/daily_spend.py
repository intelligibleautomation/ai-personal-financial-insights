from flask import Blueprint, jsonify
import pandas as pd
from database import get_db_connection
from datetime import datetime, timedelta
from flask import Blueprint, request, jsonify


bp = Blueprint("daily_spend", __name__, url_prefix="/daily-spend")


@bp.route('/trends', methods=['GET'])
def get_daily_spending_trends():
    try:
        # Connect to the database
        conn = get_db_connection()
        user_id = request.args.get("user_id")

        # Validate that user_id is provided
        if not user_id:
            return jsonify({"error": "user_id query parameter is required."}), 400
        # SQL query to fetch transaction data
        query = """
               SELECT
                   date,
                   amount
               FROM
                   user_transactions
               WHERE
                   user_id = 1
               ORDER BY
                   date;
               """

        # Load data into a pandas DataFrame
        df = pd.read_sql_query(query.format(user_id), conn)

        # Ensure the 'date' column is in datetime format
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

        # Drop rows with invalid dates or amounts
        df.dropna(subset=['date', 'amount'], inplace=True)

        # Convert 'amount' to numeric
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
        df.dropna(subset=['amount'], inplace=True)

        # Calculate daily spending trends
        daily_spending_trend = df.groupby('date')['amount'].sum().reset_index()
        daily_spending_trend.columns = ['date', 'total_spending']

        # Calculate summary statistics
        daily_average_spending = daily_spending_trend['total_spending'].mean()
        highest_spending_day = daily_spending_trend.loc[daily_spending_trend['total_spending'].idxmax()]
        lowest_spending_day = daily_spending_trend.loc[daily_spending_trend['total_spending'].idxmin()]

        # Filter data for the last 30 days
        last_30_days = datetime.now() - timedelta(days=30)
        last_30_days_data = daily_spending_trend[daily_spending_trend['date'] >= last_30_days]

        # Prepare the response
        response = {
            "daily_average_spending": round(daily_average_spending, 2),
            "highest_spending_day": {
                "date": highest_spending_day['date'].strftime('%Y-%m-%d'),
                "total_spending": round(highest_spending_day['total_spending'], 2)
            },
            "lowest_spending_day": {
                "date": lowest_spending_day['date'].strftime('%Y-%m-%d'),
                "total_spending": round(lowest_spending_day['total_spending'], 2)
            },
            "last_30_days_spending_pattern": last_30_days_data.to_dict(orient='records')
        }

        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500