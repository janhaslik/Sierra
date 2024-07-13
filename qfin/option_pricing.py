import numpy as np
from scipy.stats import norm


class BlackScholesPricer:
    """
    A class to calculate Black-Scholes option pricing for call and put options.

    Attributes:
    -----------
    S_t : float
        Current spot price of the underlying asset.
    K : float
        Strike price of the option.
    r : float
        Risk-free interest rate.
    t : float
        Time to maturity of the option (in years).
    sigma : float
        Volatility of the underlying asset.

    Methods:
    --------
    call_option():
        Returns the price of a call option.
    put_option():
        Returns the price of a put option.
    print_params():
        Prints the parameters of the Black-Scholes model.
    """

    def __init__(self, spot_price, strike_price, risk_free_interest_rate, time_to_maturity, volatility):
        """
        Initializes the Black-Scholes Pricer with the given parameters.

        Parameters:
        -----------
        spot_price : float
            Current spot price of the underlying asset.
        strike_price : float
            Strike price of the option.
        risk_free_interest_rate : float
            Risk-free interest rate.
        time_to_maturity : float
            Time to maturity of the option (in years).
        volatility : float
            Volatility of the underlying asset.
        """
        try:
            self.S_t = float(spot_price)
            self.K = float(strike_price)
            self.r = float(risk_free_interest_rate)
            self.t = float(time_to_maturity)
            self.sigma = float(volatility)

            if self.S_t <= 0 or self.K <= 0 or self.t <= 0 or self.sigma <= 0:
                raise ValueError("All input values must be positive and non-zero.")

            self.N = norm.cdf
            self.d1 = (np.log(self.S_t / self.K) + (self.r + np.power(self.sigma, 2) / 2) * self.t) / (
                        self.sigma * np.sqrt(self.t))
            self.d2 = self.d1 - self.sigma * np.sqrt(self.t)

        except ValueError as e:
            raise ValueError(f"Invalid input: {e}")

    def call_option(self) -> float:
        """
        Calculates the price of a call option using the Black-Scholes formula.

        Returns:
        --------
        float
            Price of the call option.
        """
        return self.S_t * self.N(self.d1) - self.K * np.exp(-self.r * self.t) * self.N(self.d2)

    def put_option(self) -> float:
        """
        Calculates the price of a put option using the Black-Scholes formula.

        Returns:
        --------
        float
            Price of the put option.
        """
        return self.K * np.exp(-self.r * self.t) * self.N(-self.d2) - self.S_t * self.N(-self.d1)

    def print_params(self):
        """
        Prints the parameters used in the Black-Scholes model.
        """
        print(f"Spot Price: ${self.S_t}")
        print(f"Strike Price: ${self.K}")
        print(f"Risk-Free Interest Rate: {self.r * 100}%")
        print(f"Time to Maturity: {self.t} years")
        print(f"Volatility: {self.sigma * 100}%")