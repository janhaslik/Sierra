from qfin.portfolio_optimization import PortfolioOptimizer
import qfin.time_series_analysis as tsa
from fin.kpis import calculate_pe_ratio

optimizer = PortfolioOptimizer()
model = tsa.ARIMA()

pe_ratio = calculate_pe_ratio(20.0, 4.0)
print("P/E Ratio: ", pe_ratio)

