import os

import pandas as pd
import google.generativeai as genai
import json

# Setup Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-flash")


def generate_feedback(income, expense, savings, discretionary_ratio):
    prompt = f"""
    Monthly Income: ₹{income}
    Monthly Expense: ₹{expense}
    Savings: ₹{savings}
    Discretionary Ratio: {discretionary_ratio:.2f}
    Suggest 3 actions to improve financial health.
    """
    return model.generate_content(prompt).text.strip()


def load_config(config_path="categories_config.json"):
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config['income_categories'], config['discretionary_categories']
    except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
        raise ValueError(f"Failed to load category config: {e}")


def load_and_process_transactions(filepath, income_categories, discretionary_categories):
    # Load income and discretionary categories from config
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        raise ValueError("File not found. Please check your file path.")
    except Exception as e:
        raise ValueError(f"Error loading data: {e}")

    required_columns = {'amount', 'category', 'date'}
    if not required_columns.issubset(df.columns):
        raise ValueError("CSV missing required columns: 'date', 'amount', 'category'")

    df['date'] = pd.to_datetime(df['date'])
    df['category'] = df['category'].str.lower()
    df['type'] = df['category'].apply(lambda x: 'credit' if x in income_categories else 'debit')
    df['month'] = df['date'].dt.to_period('M')

    return df, income_categories, discretionary_categories


def compute_metrics(df, discretionary_categories):
    monthly_income = df[df['type'] == 'credit'].groupby('month')['amount'].sum().mean()
    monthly_expense = df[df['type'] == 'debit'].groupby('month')['amount'].sum().mean()
    total_income = df[df['type'] == 'credit']['amount'].sum()
    total_expense = df[df['type'] == 'debit']['amount'].sum()
    total_savings = total_income - total_expense
    discretionary_spend = df[(df['type'] == 'debit') &
                             (df['category'].isin(discretionary_categories))]['amount'].sum()
    discretionary_ratio = discretionary_spend / total_expense if total_expense > 0 else 0
    return monthly_income, monthly_expense, total_savings, discretionary_ratio


def calculate_financial_health_score(income, expense, savings, discretionary_ratio, ai_risk_score=5, weights=None, return_breakdown=False):
    """
    Calculates a financial health score based on various metrics.

    Parameters:
    - income: Monthly income (float)
    - expense: Monthly expense (float)
    - savings: Total savings (float)
    - discretionary_ratio: % of expenses spent on discretionary items (float, 0–1)
    - ai_risk_score: External risk assessment score (int, 0–10)
    - weights: Optional dict to customize weights for each component
    - return_breakdown: Whether to return detailed component scores (bool)

    Returns:
    - Final score (float) or (score, breakdown_dict) if return_breakdown=True
    """

    # Default weights
    default_weights = {
        "expense": 0.25,
        "savings": 0.25,
        "discretionary": 0.2,
        "net_flow": 0.1,
        "ai_risk": 0.2
    }
    w = weights if weights else default_weights

    # Expense-to-Income Ratio
    expense_ratio = expense / income if income > 0 else 1
    expense_score = (
        30 if expense_ratio <= 0.5 else
        20 if expense_ratio <= 0.7 else
        10 if expense_ratio <= 0.9 else
        0
    )

    # Savings Rate
    savings_rate = savings / income if income > 0 else -1
    savings_score = (
        30 if savings_rate >= 0.3 else
        20 if savings_rate >= 0.2 else
        10 if savings_rate >= 0.1 else
        0
    )

    # Discretionary Spend
    discretionary_score = (
        20 if discretionary_ratio <= 0.1 else
        10 if discretionary_ratio <= 0.2 else
        5 if discretionary_ratio <= 0.3 else
        0
    )

    # Net Flow Score
    net_flow_score = 10 if savings >= 0 else 0

    # Risk Score (constrained)
    ai_risk_score = max(0, min(ai_risk_score, 10))

    # Weighted Final Score (all components are normalized to scale of 0–30 for uniformity)
    final_score = round(
        (expense_score * w["expense"] +
         savings_score * w["savings"] +
         discretionary_score * w["discretionary"] +
         net_flow_score * w["net_flow"] +
         ai_risk_score * 3 * w["ai_risk"]), 2  # Multiply risk score (0–10) to fit 0–30 scale
    )

    # Optional classification
    classification = (
        "🌟 Excellent" if final_score >= 85 else
        "👍 Good" if final_score >= 70 else
        "🟡 Moderate" if final_score >= 50 else
        "⚠️ Needs Attention"
    )

    breakdown = {
        "expense_score": expense_score,
        "savings_score": savings_score,
        "discretionary_score": discretionary_score,
        "net_flow_score": net_flow_score,
        "ai_risk_score": ai_risk_score,
        "final_score": final_score,
        "classification": classification
    }

    return (final_score, breakdown) if return_breakdown else final_score


def main(filepath):
    income_categories, discretionary_categories = load_config("categories_config.json")
    df, income_categories, discretionary_categories = load_and_process_transactions(filepath, income_categories,discretionary_categories)
    income, expense, savings, discretionary_ratio = compute_metrics(df, discretionary_categories)
    feedback = generate_feedback(income, expense, savings, discretionary_ratio)
    score, breakdown = calculate_financial_health_score(
        income,
        expense,
        savings,
        discretionary_ratio,
        ai_risk_score=2,
        return_breakdown=True
    )
    print("\n--- Financial Health Score Breakdown ---")
    for key, value in breakdown.items():
        print(f"{key.replace('_', ' ').title()}: {value}")
    print("\n--- Financial Health Report ---")
    print(f"Monthly Income: ₹{income:,.2f}")
    print(f"Monthly Expense: ₹{expense:,.2f}")
    print(f"Total Savings: ₹{savings:,.2f}")
    print(f"Discretionary Spend Ratio: {discretionary_ratio:.2%}")
    print(f"\n📊 Gemini Financial Insights: {feedback}")


if __name__ == "__main__":
    filepath = "transactions.csv"  # Replace with your file path
    main(filepath)
