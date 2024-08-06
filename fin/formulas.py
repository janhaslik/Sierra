def ebitda(operating_income: float, amortization: float, depreciation: float) -> float:
    """
    Calculate Earnings Before Interest, Taxes, Depreciation, and Amortization (EBITDA).

    Args:
        operating_income (float): Operating income of the company.
        amortization (float): Amortization expense.
        depreciation (float): Depreciation expense.

    Returns:
        float: EBITDA value.

    Raises:
        TypeError: If any input is not a float.
    """
    if not all(isinstance(arg, float) for arg in [operating_income, amortization, depreciation]):
        raise TypeError("All arguments must be of type float")
    return operating_income + amortization + depreciation


def free_cashflow(ebit: float, capex: float, taxes: float, depreciation_amortization: float,
                  depreciation: float, amortization: float, increase_ncwc: float) -> float:
    """
    Calculate Free Cash Flow (FCF).

    Args:
        ebit (float): Earnings Before Interest and Taxes.
        capex (float): Capital expenditures.
        taxes (float): Taxes paid.
        depreciation_amortization (float): Combined depreciation and amortization.
        depreciation (float): Depreciation expense.
        amortization (float): Amortization expense.
        increase_ncwc (float): Increase in net working capital.

    Returns:
        float: Free Cash Flow value.

    Raises:
        TypeError: If any input is not a float.
    """
    if not all(isinstance(arg, float) for arg in [ebit, capex, taxes, depreciation, amortization, increase_ncwc]):
        raise TypeError("All arguments must be of type float")
    if depreciation_amortization is None:
        depreciation_amortization = amortization - depreciation
    return ebit - taxes - capex + depreciation_amortization - increase_ncwc


def eps(earnings: float, shares_outstanding: float) -> float:
    """
    Calculate Earnings Per Share (EPS).

    Args:
        earnings (float): Net earnings.
        shares_outstanding (float): Number of shares outstanding.

    Returns:
        float: EPS value.

    Raises:
        TypeError: If any input is not a float.
        ZeroDivisionError: If shares_outstanding is zero.
    """
    if not all(isinstance(arg, float) for arg in [earnings, shares_outstanding]):
        raise TypeError("All arguments must be of type float")
    if shares_outstanding == 0:
        raise ZeroDivisionError("Shares outstanding cannot be zero")
    return earnings / shares_outstanding


def roa(net_income: float, avg_total_assets: float) -> float:
    """
    Calculate Return on Assets (ROA).

    Args:
        net_income (float): Net income.
        avg_total_assets (float): Average total assets.

    Returns:
        float: ROA value.

    Raises:
        TypeError: If any input is not a float.
        ZeroDivisionError: If avg_total_assets is zero.
    """
    if not all(isinstance(arg, float) for arg in [net_income, avg_total_assets]):
        raise TypeError("All arguments must be of type float")
    if avg_total_assets == 0:
        raise ZeroDivisionError("Average total assets cannot be zero")
    return (net_income / avg_total_assets) * 100


def roe(net_income: float, avg_total_equity: float) -> float:
    """
    Calculate Return on Equity (ROE).

    Args:
        net_income (float): Net income.
        avg_total_equity (float): Average total equity.

    Returns:
        float: ROE value.

    Raises:
        TypeError: If any input is not a float.
        ZeroDivisionError: If avg_total_equity is zero.
    """
    if not all(isinstance(arg, float) for arg in [net_income, avg_total_equity]):
        raise TypeError("All arguments must be of type float")
    if avg_total_equity == 0:
        raise ZeroDivisionError("Average total equity cannot be zero")
    return (net_income / avg_total_equity) * 100


def roi(net_return: float, cost_of_investment: float) -> float:
    """
    Calculate Return on Investment (ROI).

    Args:
        net_return (float): Net return from investment.
        cost_of_investment (float): Cost of the investment.

    Returns:
        float: ROI value.

    Raises:
        TypeError: If any input is not a float.
        ZeroDivisionError: If cost_of_investment is zero.
    """
    if not all(isinstance(arg, float) for arg in [net_return, cost_of_investment]):
        raise TypeError("All arguments must be of type float")
    if cost_of_investment == 0:
        raise ZeroDivisionError("Cost of investment cannot be zero")
    return (net_return / cost_of_investment) * 100


def quick_ratio(liquid_assets: float, current_liabilities: float) -> float:
    """
    Calculate Quick Ratio.

    Args:
        liquid_assets (float): Liquid assets.
        current_liabilities (float): Current liabilities.

    Returns:
        float: Quick ratio value.

    Raises:
        TypeError: If any input is not a float.
        ZeroDivisionError: If current_liabilities is zero.
    """
    if not all(isinstance(arg, float) for arg in [liquid_assets, current_liabilities]):
        raise TypeError("All arguments must be of type float")
    if current_liabilities == 0:
        raise ZeroDivisionError("Current liabilities cannot be zero")
    return (liquid_assets / current_liabilities) * 100


def current_ratio(current_assets: float, current_liabilities: float) -> float:
    """
    Calculate Current Ratio.

    Args:
        current_assets (float): Current assets.
        current_liabilities (float): Current liabilities.

    Returns:
        float: Current ratio value.

    Raises:
        TypeError: If any input is not a float.
        ZeroDivisionError: If current_liabilities is zero.
    """
    if not all(isinstance(arg, float) for arg in [current_assets, current_liabilities]):
        raise TypeError("All arguments must be of type float")
    if current_liabilities == 0:
        raise ZeroDivisionError("Current liabilities cannot be zero")
    return (current_assets / current_liabilities) * 100


def pe(market_cap: float, net_income: float) -> float:
    """
    Calculate Price-to-Earnings (P/E) Ratio.

    Args:
        market_cap (float): Market capitalization of the company.
        net_income (float): Net income of the company.

    Returns:
        float: P/E ratio value.

    Raises:
        TypeError: If any input is not a float.
        ZeroDivisionError: If net_income is zero.
    """
    if not all(isinstance(arg, float) for arg in [market_cap, net_income]):
        raise TypeError("All arguments must be of type float")
    if net_income == 0:
        raise ZeroDivisionError("Net income cannot be zero")
    return market_cap / net_income


def pb(price_per_share: float, book_value_per_share: float) -> float:
    """
    Calculate Price-to-Book (P/B) Ratio.

    Args:
        price_per_share (float): Price per share.
        book_value_per_share (float): Book value per share.

    Returns:
        float: P/B ratio value.

    Raises:
        TypeError: If any input is not a float.
        ZeroDivisionError: If book_value_per_share is zero.
    """
    if not all(isinstance(arg, float) for arg in [price_per_share, book_value_per_share]):
        raise TypeError("All arguments must be of type float")
    if book_value_per_share == 0:
        raise ZeroDivisionError("Book value per share cannot be zero")
    return (price_per_share / book_value_per_share) * 100


def capm(risk_free_rate: float, beta: float, expected_market_return: float) -> float:
    """
    Calculate Capital Asset Pricing Model (CAPM).

    Args:
        risk_free_rate (float): Risk-free rate.
        beta (float): Beta of the stock.
        expected_market_return (float): Expected market return.

    Returns:
        float: CAPM value.

    Raises:
        TypeError: If any input is not a float.
    """
    if not all(isinstance(arg, float) for arg in [risk_free_rate, beta, expected_market_return]):
        raise TypeError("All arguments must be of type float")
    return risk_free_rate + (beta * (expected_market_return - risk_free_rate))


def wacc(equity: float, debt: float, cost_of_equity: float, cost_of_debt: float, tax_rate: float) -> float:
    """
    Calculate Weighted Average Cost of Capital (WACC).

    Args:
        equity (float): Market value of equity.
        debt (float): Market value of debt.
        cost_of_equity (float): Cost of equity (from CAPM or other sources).
        cost_of_debt (float): Cost of debt (interest rate).
        tax_rate (float): Corporate tax rate as a decimal (e.g., 0.30 for 30%).

    Returns:
        float: WACC value.

    Raises:
        TypeError: If any input is not a float.
    """
    if not all(isinstance(arg, float) for arg in [equity, debt, cost_of_equity, cost_of_debt, tax_rate]):
        raise TypeError("All arguments must be of type float")

    total_value = equity + debt
    cost_of_equity_portion = (equity / total_value) * cost_of_equity
    cost_of_debt_portion = (debt / total_value) * cost_of_debt * (1 - tax_rate)

    return cost_of_equity_portion + cost_of_debt_portion


def enterprise_value(market_cap: float, total_debt: float, cash: float) -> float:
    """
    Calculate Enterprise Value of a company.

    Enterprise Value (EV) is calculated as the sum of market capitalization,
    total debt, and cash equivalents. It represents the total value of a company's
    operating assets.

    Args:
        market_cap (float): Market capitalization of the company.
        total_debt (float): Total debt of the company.
        cash (float): Cash and cash equivalents held by the company.

    Returns:
        float: Enterprise Value of the company.

    Raises:
        TypeError: If any input is not a float.
    """
    if not all(isinstance(arg, float) for arg in [market_cap, total_debt, cash]):
        raise TypeError("All arguments must be of type float")

    return market_cap + total_debt - cash



