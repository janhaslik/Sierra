from ml import LinearRegression
from qfin.portfolio_optimization import PortfolioOptimizer
import qfin.time_series_analysis as tsa
from qfin.option_pricing import BlackScholesPricer
from fin.valuation import CompsAnalysis, DCFAnalysis
from fin.amortization_table import calculate_amortization_table
import numpy as np
from matplotlib import pyplot as plt

"""q
Class for testing around the functionality of the sierra features.
"""

print("Sierra Playground:\n")

optimizer = PortfolioOptimizer()
model = tsa.ARIMA()

companies = [
    {'Name': 'Apple Inc.', 'Market Cap': 2295.26, 'Enterprise Value': 2427.26,
     'Revenue': 347.77, 'EBITDA': 96.78, 'Net Income': 94.55},
    {'Name': 'Microsoft Corporation', 'Market Cap': 2211.3, 'Enterprise Value': 2141.3,
     'Revenue': 168.1, 'EBITDA': 71.8, 'Net Income': 61.27},
    {'Name': 'Amazon.com Inc.', 'Market Cap': 1682.71, 'Enterprise Value': 1579.71,
     'Revenue': 469.8, 'EBITDA': 69.87, 'Net Income': 21.33},
    {'Name': 'Alphabet Inc. Class A', 'Market Cap': 1694.41, 'Enterprise Value': 1573.41,
     'Revenue': 182.53, 'EBITDA': 75.35, 'Net Income': 64.68},
    {'Name': 'Meta Platforms Inc.', 'Market Cap': 876.78, 'Enterprise Value': 819.78,
     'Revenue': 117.9, 'EBITDA': 39.4, 'Net Income': 29.27}
]

comps = CompsAnalysis(companies=companies)
multiples = comps.calculate_multiples()
statistics = comps.calculate_statistics()

print("Comps Analysis \n")
print(multiples, '\n')
print("Overview: Statistics")
print(statistics)

print("DCF Analysis \n")
data = {
    "1": {'EBIT': 1500, 'Taxrate': 0.21, 'D&A': 150, 'CapEx': 300, 'Change OWC': -10},
    "2": {'EBIT': 1700, 'Taxrate': 0.21, 'D&A': 160, 'CapEx': 320, 'Change OWC': -15},
    "3": {'EBIT': 1900, 'Taxrate': 0.21, 'D&A': 170, 'CapEx': 340, 'Change OWC': -20},
    "4": {'EBIT': 2100, 'Taxrate': 0.21, 'D&A': 180, 'CapEx': 360, 'Change OWC': -25},
    "5": {'EBIT': 2300, 'Taxrate': 0.21, 'D&A': 190, 'CapEx': 380, 'Change OWC': -30},
    "6": {'EBIT': 2500, 'Taxrate': 0.21, 'D&A': 200, 'CapEx': 400, 'Change OWC': -35},
    "7": {'EBIT': 2700, 'Taxrate': 0.21, 'D&A': 210, 'CapEx': 420, 'Change OWC': -40},
    "8": {'EBIT': 2900, 'Taxrate': 0.21, 'D&A': 220, 'CapEx': 440, 'Change OWC': -45},
    "9": {'EBIT': 3100, 'Taxrate': 0.21, 'D&A': 230, 'CapEx': 460, 'Change OWC': -50},
    "10": {'EBIT': 3200, 'Taxrate': 0.21, 'D&A': 240, 'CapEx': 480, 'Change OWC': -55}
}

assumptions = {
    "risk_free_rate": 0.03,  # Risk-free rate (3%)
    "beta": 1.2,  # Beta coefficient for the company (1.2)
    "expected_market_return": 0.08,  # Expected market return (8%)
    "cost_of_debt": 0.045,  # Cost of debt (4.5%)
    "equity": 1000000.0,  # Value of equity ($1,000,000)
    "debt": 500000.0,  # Value of debt ($500,000)
    "tax_rate": 0.21,  # Corporate tax rate (21%)
    "growth_rate": 0.05  # Long-term growth rate for FCF (5%)
}

dcf_analysis = DCFAnalysis(period=6, assumptions=assumptions)
dcf_analysis.calculate_free_cash_flow(data=data)
tv = dcf_analysis.calculate_terminal_value()
ev = dcf_analysis.calculate_enterprise_value()

print(dcf_analysis.free_cash_flow_table)
print(dcf_analysis.discount_table)
print("\nTerminal Value:", f'${tv:.2f}')
print("Enterprise Value:", f'${ev:.2f}\n')

print("Black Scholes Model\n")

pricer = BlackScholesPricer(spot_price=421, strike_price=350, time_to_maturity=5,
                            risk_free_interest_rate=0.03, volatility=0.15)
call_price = pricer.call_option()
put_price = pricer.put_option()

print("Params:\n")
pricer.print_params()
print("\nCall Option, Price:", call_price)
print("Put Option, Price:", put_price)

"""
print("\nBlack Scholes Model, Effect of Time on Price\n")

time_range = 10
for t in range(1, time_range):
    pricer = BlackScholesPricer(spot_price=300, strike_price=250, time_to_maturity=t,
                                risk_free_interest_rate=0.03, volatility=0.15)
    call_price = pricer.call_option()
    put_price = pricer.put_option()
    print("Call Option, Price:", call_price)
    print("Put Option, Price:", put_price)
"""

print("\nAmortization Table\n")
total_interests, total_principal, total_payment, table, months_taken = calculate_amortization_table(1500, 0.03, 200000, 15)

print(table)
print(f'\nTotal Interests: ${total_interests}')
print(f'Total Principal: ${total_principal}')
print(f'Total Payment: ${total_payment}')
print(f'Months Taken: {months_taken}m or {round(months_taken / 12, 2)}yrs')

print("\nLinear Regression\n")

np.random.seed(0)

X = np.random.randint(1, 11, size=(100, 1))

y = 30000 + 5000 * X + np.random.randn(100, 1) * 2000
y = y.flatten()

split_index = int(len(X)*0.8)
X_train, X_test, y_train, y_test = X[:split_index], X[split_index:], y[:split_index], y[split_index:]

linear = LinearRegression(iterations=2000, learning_rate=0.05)
linear.fit(X_train, y_train)
mse_score, accuracy_score = linear.evaluate(X_test, y_test)

print("MSE:", mse_score)
print("Accuracy:", accuracy_score)

print("Weights:", linear.weights)
print("Bias:", linear.bias)

# Plotting training history
plt.figure(figsize=(12, 6))

# Plot MSE history
plt.subplot(1, 2, 1)
plt.plot(linear.history)
plt.xlabel('Iterations')
plt.ylabel('MSE')
plt.title('Training Loss (MSE) History')

# Plot predictions vs actual data
plt.subplot(1, 2, 2)
plt.scatter(X, y, color='blue', label='Actual data')
plt.plot(X, linear.predict(X), color='red', label='Fitted line')
plt.xlabel('Job Level')
plt.ylabel('Salary')
plt.title('Linear Regression Fit')
plt.legend()

plt.tight_layout()
plt.show()