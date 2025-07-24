from flask import Blueprint, jsonify
import pandas as pd
from database import get_db_connection

bp = Blueprint("spending_by_category", __name__, url_prefix="/spending-by-category")


@bp.route('/breakdown', methods=['GET'])
def get_spending_by_category():
    try:
        # Connect to the database
        conn = get_db_connection()

        # SQL query to fetch transaction data
        query = """
        SELECT
            category,
            subcategory,
            amount
        FROM
            user_transactions
        ORDER BY
            date DESC
        LIMIT 10
        """

        # Load data into a pandas DataFrame
        df = pd.read_sql_query(query, conn)

        # Drop rows with invalid amounts
        df.dropna(subset=['amount'], inplace=True)

        # Convert 'amount' to numeric
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
        df.dropna(subset=['amount'], inplace=True)

        # Calculate total spending
        total_spending = df['amount'].sum()

        # Group by category and calculate spending
        category_data = df.groupby('category')['amount'].sum().reset_index()
        category_data['percentage'] = (category_data['amount'] / total_spending * 100).apply(round, ndigits=2)

        # Group by subcategory for detailed breakdown
        subcategory_data = df.groupby(['category', 'subcategory'])['amount'].sum().reset_index()

        # Prepare the response
        response = {
            "total_spending": round(total_spending, 2),  # Use built-in round()
            "categories": []
        }

        for _, row in category_data.iterrows():
            category = row['category']
            category_total = round(row['amount'], 2)  # Use built-in round()
            category_percentage = round(row['percentage'], 2)  # Use built-in round()

            # Filter subcategories for the current category
            subcategories = subcategory_data[subcategory_data['category'] == category]
            subcategories_list = [
                {
                    "subcategory": sub_row['subcategory'],
                    "amount": round(sub_row['amount'], 2),  # Use built-in round()
                    "percentage": round((sub_row['amount'] / category_total * 100), 2)  # Use built-in round()
                }
                for _, sub_row in subcategories.iterrows()
            ]

            response["categories"].append({
                "category": category,
                "total_spending": category_total,
                "percentage_of_total": category_percentage,
                "subcategories": subcategories_list
            })

        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500