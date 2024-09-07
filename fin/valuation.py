import pandas as pd
from fin.formulas import pe, wacc, capm


class DCFAnalysis:
    """
    Performs Discounted Cash Flow (DCF) analysis based on given financial data and assumptions.

    Attributes:
    - period (int): Number of periods for the DCF analysis.
    - assumptions (dict): Dictionary containing financial assumptions required for calculations.
    - pv_sum (float): Present value sum of discounted cash flows.
    - enterprise_value (float): Calculated enterprise value.
    - discount_table (pd.DataFrame): Table storing discounted cash flows.
    - free_cash_flow_table (pd.DataFrame): Table storing free cash flows for each period.
    - terminal_value (float): Terminal value of the cash flows.

    Methods:
    - calculate_free_cash_flow(data): Calculates free cash flows for each period based on input data.
    - calculate_terminal_value(): Calculates the terminal value based on discounted cash flows and growth rate.
    - calculate_enterprise_value(): Calculates the enterprise value based on present value sum and terminal value.
    """

    def __init__(self, period: int, assumptions: dict):
        """
        Initializes a DCFAnalysis object with a given period and financial assumptions.

        Args:
        - period (int): Number of periods for the analysis.
        - assumptions (dict): Dictionary of financial assumptions required for calculations.
        """
        self.pv_sum = None
        self.enterprise_value = None
        self.discount_table = None
        self.free_cash_flow_table = None
        self.terminal_value = None
        self.assumptions = assumptions
        self.period = period

    def calculate_free_cash_flow(self, data: dict) -> pd.DataFrame:
        """
        Calculates free cash flows for each period based on input data.

        Args:
        - data (dict): Dictionary containing financial data for each period.

        Returns:
        - pd.DataFrame: Table storing free cash flows for each period and expected keys.
        """
        expected_keys = ['EBIT', 'Tax', 'D&A', 'CapEx', 'Change OWC', 'FCF']

        if self.period > len(data):
            raise ValueError('Period must be equal to or less than the number of data points')

        free_cash_flow_table = pd.DataFrame(columns=range(1, self.period + 1), index=expected_keys)
        for year in range(1, self.period + 1):
            str_year = str(year)

            ebit = data[str_year]['EBIT']
            taxrate = data[str_year]['Taxrate']
            tax = ebit * taxrate
            D_and_A = data[str_year]['D&A']
            CapEx = data[str_year]['CapEx']
            Change_in_OWC = data[str_year]['Change OWC']

            fcf = ebit - tax + D_and_A - CapEx - Change_in_OWC

            free_cash_flow_table.loc['EBIT', int(year)] = ebit
            free_cash_flow_table.loc['Tax', int(year)] = -tax
            free_cash_flow_table.loc['D&A', int(year)] = D_and_A
            free_cash_flow_table.loc['CapEx', int(year)] = -CapEx
            free_cash_flow_table.loc['Change OWC', int(year)] = -Change_in_OWC
            free_cash_flow_table.loc['FCF', int(year)] = fcf

        self.free_cash_flow_table = free_cash_flow_table
        return free_cash_flow_table

    def calculate_terminal_value(self) -> float:
        """
        Calculates the terminal value based on discounted cash flows and growth rate.

        Returns:
        - float: Terminal value of the cash flows.
        """
        if self.free_cash_flow_table is None:
            raise ValueError('Free cash flow table is not calculated. Call calculate_free_cash_flow first.')

        self.discount_table = self.free_cash_flow_table.copy().drop(index=['EBIT', 'Tax', 'D&A', 'CapEx', 'Change OWC'])

        risk_free_rate = self.assumptions['risk_free_rate']
        beta = self.assumptions['beta']
        expected_market_return = self.assumptions['expected_market_return']

        cost_of_equity = capm(risk_free_rate=risk_free_rate, beta=beta, expected_market_return=expected_market_return)
        cost_of_debt = self.assumptions['cost_of_debt']
        equity = self.assumptions['equity']
        debt = self.assumptions['debt']
        tax_rate = self.assumptions['tax_rate']
        growth_rate = self.assumptions['growth_rate']

        discount_rate = wacc(equity=equity, debt=debt, cost_of_equity=cost_of_equity, cost_of_debt=cost_of_debt,
                             tax_rate=tax_rate)

        for year in range(1, self.period + 1):
            discount_factor = 1 / (1 + discount_rate) ** year
            self.discount_table.loc['Discount Factor', year] = discount_factor
            self.discount_table.loc['Present Value', year] = self.discount_table.loc['FCF', year] * discount_factor

        self.pv_sum = self.discount_table.loc['Present Value'].sum()
        fcf_n = self.discount_table.loc['Present Value', self.period]

        self.terminal_value = (fcf_n * (1 + growth_rate)) / (discount_rate - growth_rate)

        return self.terminal_value

    def calculate_enterprise_value(self) -> float:
        """
        Calculates the enterprise value based on present value sum and terminal value.

        Returns:
        - float: Calculated enterprise value.
        """
        if self.pv_sum is None or self.terminal_value is None:
            raise ValueError('PV sum or terminal value not calculated. Call calculate_terminal_value first.')

        self.enterprise_value = self.pv_sum + self.terminal_value
        return self.enterprise_value


class CompsAnalysis:
    """
    Class for performing Comparable Company Analysis (Comps).

    This class initializes with a DataFrame containing company financial data and provides methods
    to calculate financial multiples and statistics for analysis.

    Attributes:
        df (pd.DataFrame): DataFrame containing company financial data with columns:
            'Name', 'Market Cap', 'Enterprise Value', 'Net Income', 'Revenue', 'EBITDA'.
    """

    def __init__(self, companies):
        """
        Initialize the Comps instance with a list of companies.

        Args:
            companies (list): List of dictionaries where each dictionary represents a company's financial data.
                Expected keys: 'Name', 'Market Cap', 'Enterprise Value', 'Net Income', 'Revenue', 'EBITDA'.
                All dictionaries in the list must contain these keys.
        """
        # Validate keys in each dictionary
        expected_keys = {'Name', 'Market Cap', 'Enterprise Value', 'Net Income', 'Revenue', 'EBITDA'}
        for company in companies:
            if not expected_keys.issubset(company.keys()):
                raise ValueError(
                    f"Missing required key(s) in company '{company['Name']}' key(s): {expected_keys - company.keys()}")

        df = pd.DataFrame(companies,
                          columns=['Name', 'Market Cap', 'Enterprise Value', 'Net Income', 'Revenue', 'EBITDA'])
        df.drop_duplicates(subset=['Name'], inplace=True)
        self.df = df

    def calculate_multiples(self):
        """
        Calculate financial multiples (EV/Revenue, EV/EBITDA, P/E) for each company.

        Returns:
            pd.DataFrame: DataFrame containing company names and their corresponding 'EV/Revenue', 'EV/EBITDA', 'P/E' ratios.

        Raises:
            KeyError: If required columns ('Market Cap', 'Net Income') are missing in the DataFrame.
            TypeError: If 'Market Cap' or 'Net Income' columns contain non-numeric values.
            ZeroDivisionError: If 'Net Income' is zero for any company.
        """
        try:
            self.df['EV/Revenue'] = self.df['Enterprise Value'] / self.df['Revenue']
            self.df['EV/EBITDA'] = self.df['Enterprise Value'] / self.df['EBITDA']
            self.df['P/E'] = self.df.apply(lambda row: pe(row['Market Cap'], row['Net Income']), axis=1)
            return self.df[['Name', 'EV/Revenue', 'EV/EBITDA', 'P/E']]
        except KeyError as e:
            raise KeyError(f"Missing column: {e}")
        except (TypeError, ZeroDivisionError) as e:
            raise type(e)(str(e))

    def calculate_statistics(self):
        """
        Calculate statistics (minimum, 25th Percentile, mean, median, 75th Percentile, maximum) for financial multiples across all companies.

        Returns:
            pd.DataFrame: DataFrame containing mean and median values for 'EV/Revenue', 'EV/EBITDA', and 'P/E'.
        """
        statistics_table_columns = ['EV/Revenue', 'EV/EBITDA', 'P/E']
        statistic_table = pd.DataFrame(index=['Min', '25th Percentile', 'Mean', 'Median', '75th Percentile', 'Max'],
                                       columns=statistics_table_columns)

        for multiple in statistics_table_columns:
            statistic_table.loc['Min', multiple] = min(self.df[multiple])
            statistic_table.loc['25th Percentile', multiple] = self.df[multiple].quantile(0.25)
            statistic_table.loc['Mean', multiple] = self.df[multiple].mean()
            statistic_table.loc['Median', multiple] = self.df[multiple].median()
            statistic_table.loc['75th Percentile', multiple] = self.df[multiple].quantile(0.75)
            statistic_table.loc['Max', multiple] = max(self.df[multiple])

        return statistic_table
