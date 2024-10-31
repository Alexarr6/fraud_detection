import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import json


def earnings_and_expenses(
    df: pd.DataFrame, client_id: int, start_date: str, end_date: str
) -> pd.DataFrame:
    """
    For the period defined in between start_date and end_date (both included), get the client data available and return
    a pandas DataFrame with the Earnings and Expenses total amount for the period range and user given.The expected
    columns are:
        - Earnings
        - Expenses
    The DataFrame should have the columns in this order ['Earnings','Expenses']. Round the amounts to 2 decimals.

    Create a Bar Plot with the Earnings and Expenses absolute values and save it as
    "reports/figures/earnings_and_expenses.png" .

    Parameters
    ----------
    df : pandas DataFrame
       DataFrame of the data to be used for the agent.
    client_id : int
        Id of the client.
    start_date : str
        Start date for the date period. In the format "YYYY-MM-DD".
    end_date : str
        End date for the date period. In the format "YYYY-MM-DD".


    Returns
    -------
    Pandas Dataframe with the earnings and expenses rounded to 2 decimals.

    73 points task 2
    46 points task 5
    """

    df = df.copy(deep=True)

    df['date'] = pd.to_datetime(df['date'])
    df['amount'] = df['amount'].replace('[\$,]', '', regex=True).astype(float)
    client_df = df[df['client_id'] == client_id]

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    client_period_df = client_df[(client_df['date'] >= start_date) & (client_df['date'] <= end_date)]

    earnings = round(client_period_df[client_period_df['amount'] > 0]['amount'].sum(), 2)
    expenses = round(client_period_df[client_period_df['amount'] < 0]['amount'].sum(), 2)

    os.makedirs('reports/figures', exist_ok=True)
    sns.set(style='whitegrid')
    plt.figure(figsize=(8, 6))
    sns.barplot(x=['Earnings', 'Expenses'], y=[earnings, expenses], palette='pastel', edgecolor='black')
    plt.ylabel('Amount')
    plt.title('Earnings and Expenses')
    plt.tight_layout()
    plt.savefig('reports/figures/earnings_and_expenses.png')
    plt.close()

    return pd.DataFrame({"Earnings": [earnings], "Expenses": [expenses]})


def expenses_summary(
    df: pd.DataFrame, client_id: int, start_date: str, end_date: str
) -> pd.DataFrame:
    """
    For the period defined in between start_date and end_date (both included), get the client data available and return
    a Pandas Data Frame with the Expenses by merchant category. The expected columns are:
        - Expenses Type --> (merchant category names)
        - Total Amount
        - Average
        - Max
        - Min
        - Num. Transactions
    The DataFrame should be sorted alphabeticaly by Expenses Type and values have to be rounded to 2 decimals. Return
    the dataframe with the columns in the given order.
    The merchant category names can be found in data/raw/mcc_codes.json .

    Create a Bar Plot with the data in absolute values and save it as "reports/figures/expenses_summary.png" .

    Parameters
    ----------
    df : pandas DataFrame
       DataFrame  of the data to be used for the agent.
    client_id : int
        Id of the client.
    start_date : str
        Start date for the date period. In the format "YYYY-MM-DD".
    end_date : str
        End date for the date period. In the format "YYYY-MM-DD".


    Returns
    -------
    Pandas Dataframe with the Expenses by merchant category.

    63 points task 2 (F1 + F2 = 136)
    46 points task 5 -> No suma nada! Es porque no se genera el report. Me dan 46 puntos sin generar el report :)

    """

    with open('data/raw/mcc_codes.json', 'r') as f:
        mcc_codes = json.load(f)

    df = df.copy(deep=True)

    df['date'] = pd.to_datetime(df['date'])
    df['amount'] = df['amount'].replace('[\$,]', '', regex=True).astype(float)

    client_df = df[df['client_id'] == client_id]

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    client_period_df = client_df[(client_df['date'] >= start_date) & (client_df['date'] <= end_date)]

    expenses_df = client_period_df[client_period_df['amount'] < 0]

    expenses_df['Expenses Type'] = expenses_df['mcc'].astype(str).map(mcc_codes)

    solution_df = expenses_df.groupby('Expenses Type')['amount'].agg(
        Total_Amount=lambda x: round(-x.sum(), 2),
        Average=lambda x: round(-x.mean(), 2),
        Max=lambda x: round(x.abs().min(), 2),
        Min=lambda x: round(x.abs().max(), 2),
        Num_Transactions='count'
    ).reset_index()

    solution_df = solution_df.rename(columns={
        'Total_Amount': 'Total Amount',
        'Num_Transactions': 'Num. Transactions'
    })
    solution_df = solution_df[['Expenses Type', 'Total Amount', 'Average', 'Max', 'Min', 'Num. Transactions']]
    solution_df = solution_df.sort_values('Expenses Type')

    os.makedirs('reports/figures', exist_ok=True)
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 8))
    sns.barplot(
        data=solution_df,
        x='Expenses Type',
        y='Total Amount',
        palette="pastel",
        edgecolor="black"
    )
    plt.xticks(rotation=90)
    plt.ylabel('Total Amount')
    plt.title('Expenses by Merchant Category')
    plt.tight_layout()
    plt.savefig('reports/figures/expenses_summary.png')
    plt.close()

    return solution_df


def cash_flow_summary(
    df: pd.DataFrame, client_id: int, start_date: str, end_date: str
) -> pd.DataFrame:
    """
    For the period defined by start_date and end_date (both inclusive), retrieve the available client data and return a
    Pandas DataFrame containing cash flow information.

    If the period exceeds 60 days, group the data by month, using the end of each month for the date. If the period is
    60 days or shorter, group the data by week.

        The expected columns are:
            - Date --> the date for the period. YYYY-MM if period larger than 60 days, YYYY-MM-DD otherwise.
            - Inflows --> the sum of the earnings (positive amounts)
            - Outflows --> the sum of the expenses (absolute values of the negative amounts)
            - Net Cash Flow --> Inflows - Outflows
            - % Savings --> Percentage of Net Cash Flow / Inflows

        The DataFrame should be sorted by ascending date and values rounded to 2 decimals. The columns should be in the
        given order.

        Parameters
        ----------
        df : pandas DataFrame
           DataFrame  of the data to be used for the agent.
        client_id : int
            Id of the client.
        start_date : str
            Start date for the date period. In the format "YYYY-MM-DD".
        end_date : str
            End date for the date period. In the format "YYYY-MM-DD".


        Returns
        -------
        Pandas Dataframe with the cash flow summary.

        19 points task 2 (F1 + F2 + F3 = 155)
        138 points task 5 -> Se completa el report
    """

    df = df.copy(deep=True)

    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    df['amount'] = df['amount'].replace('[\$,]', '', regex=True).astype(float)
    client_df = df[df['client_id'] == client_id]

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    client_period_df = client_df[(client_df['date'] >= start_date) & (client_df['date'] <= end_date)]

    period_length = (end_date - start_date).days + 1

    if period_length > 60:
        client_period_df['Period'] = client_period_df['date'].dt.to_period('M').dt.to_timestamp('M')
        date_format = '%Y-%m'

    else:
        client_period_df['Period'] = client_period_df['date'].dt.to_period('W-SUN').apply(lambda r: r.end_time)
        date_format = '%Y-%m-%d'

    solution_df = client_period_df.groupby('Period')['amount'].agg(
        Inflows=lambda x: x[x > 0].sum(),
        Outflows=lambda x: x[x < 0].abs().sum()
    ).reset_index()

    solution_df['Inflows'] = solution_df['Inflows'].fillna(0)
    solution_df['Outflows'] = solution_df['Outflows'].fillna(0)

    solution_df['Net Cash Flow'] = solution_df['Inflows'] - solution_df['Outflows']
    solution_df['% Savings'] = np.where(
        solution_df['Inflows'] == 0,
        0,
        (solution_df['Net Cash Flow'] / solution_df['Inflows']) * 100
    )

    solution_df['Inflows'] = solution_df['Inflows'].round(2)
    solution_df['Outflows'] = solution_df['Outflows'].round(2)
    solution_df['Net Cash Flow'] = solution_df['Net Cash Flow'].round(2)
    solution_df['% Savings'] = solution_df['% Savings'].round(2)

    solution_df['Date'] = solution_df['Period'].dt.strftime(date_format)

    cash_flow_df = solution_df[['Date', 'Inflows', 'Outflows', 'Net Cash Flow', '% Savings']]
    cash_flow_df = cash_flow_df.sort_values('Date').reset_index(drop=True)

    return cash_flow_df


if __name__ == "__main__":

    current_path = os.getcwd()
    root_dir_name = "hackathon-caixabank-data-ai-report"
    root_dir_path = current_path[:current_path.index(root_dir_name) + len(root_dir_name)]
    os.chdir(root_dir_path)

    transactions_data_df = pd.read_csv('data/raw/transactions_data.csv')

    earnings_and_expenses_df = earnings_and_expenses(transactions_data_df, 5, "2018-01-01", "2019-01-01")
    expenses_summary_df = expenses_summary(transactions_data_df, 5, "2018-01-01", "2019-01-01")
    cash_flow_summary_df = cash_flow_summary(transactions_data_df, 5, "2008-01-01", "2020-06-15")
