class PortfolioOptimizer:
    def __init__(self, portfolio=None):
        if portfolio is None:
            portfolio = {}
        self.portfolio = portfolio
