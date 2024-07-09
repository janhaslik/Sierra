from qfin.portfolio_optimization import PortfolioOptimizer
import qfin.time_series_analysis as tsa
from fin.formulas import capm
from fin.valuation import Comps

"""
Class for testing around the functionality of the sierra features.
"""

optimizer = PortfolioOptimizer()
model = tsa.ARIMA()

capm = capm(risk_free_rate=2.5, beta=0.0, expected_market_return=8.0)
print(f'CAPM: {capm}%')

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

comps = Comps(companies=companies)
multiples = comps.calculate_multiples()
statistics = comps.calculate_statistics()

print("Comps Analysis \n")
print(multiples, '\n')
print("Overview: Statistics")
print(statistics)

