import pandas as pd


def calculate_amortization_table(payment: float, annual_rate: float, loan_amount: float, term: int):
    """
    Calculate the amortization table for a given loan.

    Parameters:
    payment (float): The fixed monthly payment amount.
    annual_rate (float): The annual interest rate as a decimal.
    loan_amount (float): The initial amount of the loan.
    term (int): The term of the loan in years.

    Returns:
    tuple: A tuple containing:
        - total_interests (float): The total interest paid over the life of the loan.
        - total_principal (float): The total principal paid over the life of the loan.
        - total_payment (float): The total amount paid over the life of the loan.
        - table (pd.DataFrame): The amortization table as a DataFrame.
        - months (int): The actual number of months the loan was paid.

    Raises:
    ValueError: If any input values are not positive and greater than zero.
    ValueError: If the monthly payment is less than or equal to the monthly interest.
    """
    # Validate inputs
    if payment <= 0 or annual_rate < 0 or loan_amount <= 0 or term <= 0:
        raise ValueError("All input values must be positive and greater than zero.")

    table = pd.DataFrame(columns=['Month', 'Payment', 'Principal', 'Interests', 'Balance'])
    total_interests = 0
    total_principal = 0
    monthly_interest_rate = annual_rate / 12

    balance = loan_amount

    def set_values(month_value, payment_value, interests_value, principal_value, balance_value):
        table.loc[month_value] = [
            month_value,
            round(payment_value, 2),
            round(principal_value, 2),
            round(interests_value, 2),
            round(balance_value, 2)
        ]

    month = 1
    while balance > 0:
        interests = monthly_interest_rate * balance
        principal = payment - interests

        if principal <= 0:
            raise ValueError(f'Payment must be greater than interests, must raise by: ${1 + principal * -1:.2f}')

        if balance - principal <= 0:
            final_payment = balance + interests
            set_values(month, final_payment, interests, balance, 0)
            total_interests += interests
            total_principal += balance
            break

        set_values(month, payment, interests, principal, balance - principal)
        total_interests += interests
        total_principal += principal
        balance -= principal
        month += 1

    total_payment = round(total_interests + total_principal, 2)

    return round(total_interests, 2), round(total_principal, 2), total_payment, table, month
