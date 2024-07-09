import pandas as pd
from fin.formulas import pe


class DCF:
    def __init__(self):
        return


class Comps:
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
        statistic_table = pd.DataFrame(index=['Min', '25th Percentile', 'Mean', 'Median', '75th Percentile', 'Max'], columns=statistics_table_columns)

        for multiple in statistics_table_columns:
            statistic_table.loc['Min', multiple] = min(self.df[multiple])
            statistic_table.loc['25th Percentile', multiple] = self.df[multiple].quantile(0.25)
            statistic_table.loc['Mean', multiple] = self.df[multiple].mean()
            statistic_table.loc['Median', multiple] = self.df[multiple].median()
            statistic_table.loc['75th Percentile', multiple] = self.df[multiple].quantile(0.75)
            statistic_table.loc['Max', multiple] = max(self.df[multiple])

        return statistic_table

