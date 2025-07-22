TRANSACTION_TYPES = ['Income', 'Expense', 'To Receive', 'To Pay']

CATEGORIES = {
    'Expense': {
        'Food': ['Groceries', 'Dining Out', 'Snacks'],
        'Transportation': ['Fuel', 'Public Transit', 'Maintenance'],
        'Housing': ['Rent', 'Utilities', 'Maintenance'],
        'Entertainment': ['Movies', 'Games', 'Events'],
        'Shopping': ['Clothes', 'Electronics', 'Home Items'],
        'Healthcare': ['Medical', 'Pharmacy', 'Insurance'],
        'Gift': ['Birthday', 'Wedding', 'Holiday', 'Other'],
        'Other': ['Miscellaneous', 'Unspecified']
    },
    'Income': {
        'Salary': ['Regular', 'Bonus', 'Overtime'],
        'Investment': ['Dividends', 'Interest', 'Capital Gains'],
        'Other': ['Gifts', 'Refunds', 'Miscellaneous']
    },
    'To Receive': {
        'Pending Income': ['Salary', 'Investment', 'Other']
    },
    'To Pay': {
        'Bills': ['Utilities', 'Rent', 'Other'],
        'Debt': ['Credit Card', 'Loan', 'Other']
    }
}