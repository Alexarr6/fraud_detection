import pandas as pd
import json
import os


def question_1(cards_df: pd.DataFrame) -> dict:
    """
    Q1: - The `card_id` with the latest expiry date and the lowest credit limit amount.
    """

    cards_df['expires'] = pd.to_datetime(cards_df['expires'], format='%m/%Y')
    cards_df['credit_limit'] = cards_df['credit_limit'].replace('[\$,]', '', regex=True).astype(float)

    result = cards_df. \
        sort_values(by=['expires', 'credit_limit'], ascending=[False, True]). \
        iloc[0]['card_id']

    return {'card_id': int(result)}


def question_2(client_df: pd.DataFrame) -> dict:
    """
    Q2: - The `client_id` that will retire within a year that has the lowest credit score and highest debt.
    """

    client_df['total_debt'] = client_df['total_debt'].replace('[\$,]', '', regex=True).astype(float)
    client_df['years_to_retirement'] = client_df['retirement_age'] - client_df['current_age']
    df_retiring_soon = client_df[(client_df['years_to_retirement'] <= 1) & (client_df['years_to_retirement'] >= 0)]

    result = df_retiring_soon. \
        sort_values(by=['credit_score', 'total_debt'], ascending=[True, False]). \
        iloc[0]['client_id']

    return {'client_id': int(result)}


def question_3(transactions_df: pd.DataFrame) -> dict:
    """
    Q3: - The `transaction_id` of an Online purchase on a 31st of December with the highest absolute amount
    (either earnings or expenses).
    """

    transactions_df['amount'] = transactions_df['amount'].replace('[\$,]', '', regex=True).astype(float)
    transactions_df['date'] = pd.to_datetime(transactions_df['date'])

    online_transactions_31_dec_df = transactions_df[
        (transactions_df['date'].dt.month == 12) &
        (transactions_df['date'].dt.day == 31) &
        (transactions_df['use_chip'] == 'Online Transaction')
    ]

    online_transactions_31_dec_df['amount'] = online_transactions_31_dec_df['amount'].abs()
    online_transactions_31_dec_df = online_transactions_31_dec_df.sort_values(by='amount', ascending=False)

    result = online_transactions_31_dec_df.iloc[0]['id']

    return {"transaction_id": int(result)}


def question_4(client_df: pd.DataFrame, cards_df: pd.DataFrame, transactions_df: pd.DataFrame) -> dict:
    """
    Q4: - Which client over the age of 40 made the most transactions with a Visa card in February 2016?
    Please return the `client_id`, the `card_id` involved, and the total number of transactions.
    """

    clients_over_40_df = client_df[client_df['current_age'] > 40]
    visa_cards_df = cards_df[cards_df['card_brand'] == 'Visa']
    clients_visa_df = pd.merge(clients_over_40_df, visa_cards_df, on='client_id', how='inner')

    transactions_df['date'] = pd.to_datetime(transactions_df['date'])
    february_transactions_df = transactions_df[
        (transactions_df['date'].dt.year == 2016) &
        (transactions_df['date'].dt.month == 2)
    ]

    february_transactions_visa_df = pd.merge(
        february_transactions_df,
        clients_visa_df,
        on=['client_id', 'card_id'],
        how='inner'
    )

    transaction_counts_df = february_transactions_visa_df. \
        groupby(['client_id', 'card_id']). \
        size(). \
        reset_index(name='transaction_count'). \
        sort_values(by='transaction_count', ascending=False)

    result = transaction_counts_df.iloc[0].to_list()

    return {"client_id": int(result[0]), "card_id": int(result[1]), "number_transactions": int(result[2])}


if __name__ == "__main__":

    current_path = os.getcwd()
    root_dir_name = "hackathon-caixabank-data-ai-report"
    root_dir_path = current_path[:current_path.index(root_dir_name) + len(root_dir_name)]
    os.chdir(root_dir_path)

    transactions_data_df = pd.read_csv('data/raw/transactions_data.csv')
    cards_data_df = pd.read_csv('data/raw/card_data.csv')
    client_data_df = pd.read_csv('data/raw/client_data.csv')

    result_question_1 = question_1(cards_data_df)
    result_question_2 = question_2(client_data_df)
    result_question_3 = question_3(transactions_data_df)
    result_question_4 = question_4(client_data_df, cards_data_df, transactions_data_df)

    final_result = {
        'target': {
            'query_1': result_question_1,
            'query_2': result_question_2,
            'query_3': result_question_3,
            'query_4': result_question_4
        }
    }

    with open('predictions/predictions_1.json', 'w') as archivo:
        json.dump(final_result, archivo, indent=4)
