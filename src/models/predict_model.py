import json
import os
import pickle

from pandas import DataFrame, Period, read_csv, Categorical


def compute_fraud_predictions() -> None:

    transactions_df = read_csv('data/processed/processed_transactions_data.csv')

    with open('predictions/predictions_3.json', 'r') as f:
        prediction_template = json.load(f)

    with open('models/xgb_fraud_detection_model.pkl', 'rb') as f:
        model = pickle.load(f)

    test_transactions_ids = list(prediction_template['target'].keys())
    test_transactions_ids = [int(transaction_id) for transaction_id in test_transactions_ids]

    test_transactions_df = transactions_df[transactions_df['id'].isin(test_transactions_ids)]
    test_df = test_transactions_df.drop(columns=['id'])

    y_test_pred = model.predict(test_df)
    test_transactions_df['prediction'] = y_test_pred
    test_transactions_df['prediction'] = test_transactions_df['prediction'].map({0: 'No', 1: 'Yes'})

    test_transactions_df['id'] = Categorical(
        test_transactions_df['id'],
        categories=test_transactions_ids,
        ordered=True
    )
    test_transactions_df = test_transactions_df.sort_values('id').reset_index(drop=True)
    test_transactions_df['id'] = test_transactions_df['id'].astype(str)

    solution = test_transactions_df.set_index('id')['prediction'].to_dict()
    final_solution = {'target': solution}

    with open('predictions/predictions_3.json', 'w') as archivo:
        json.dump(final_solution, archivo, indent=4)


def compute_forecast_expenses_predictions() -> None:

    monthly_expenses_df = read_csv('data/processed/monthly_expenses_data.csv')

    with open('predictions/predictions_4.json', 'r') as f:
        prediction_template = json.load(f)

    with open('models/xgb_forecast_expenses_model.pkl', 'rb') as f:
        model = pickle.load(f)

    predictions = {'target': {}}
    last_data = monthly_expenses_df.groupby('client_id').last().reset_index()

    for client_id_str, dates_dict in prediction_template['target'].items():
        client_id = int(client_id_str)

        client_history = last_data[last_data['client_id'] == client_id]
        lags = [
            client_history['amount'].values[0],
            client_history['lag_1'].values[0],
            client_history['lag_2'].values[0]
        ]
        client_predictions = {}

        for date_str in dates_dict.keys():
            next_month = Period(date_str, freq='M')
            month_number = next_month.month
            year = next_month.year
            features = {
                'month_number': month_number,
                'year': year,
                'lag_1': lags[0],
                'lag_2': lags[1],
                'lag_3': lags[2],
                'credit_score': client_history['credit_score'].values[0],
                'total_debt': client_history['total_debt'].values[0],
                'per_capita_income': client_history['per_capita_income'].values[0],
                'yearly_income': client_history['yearly_income'].values[0],
                'gender': client_history['gender'].values[0]
            }
            X_pred = DataFrame([features])
            amount_pred = model.predict(X_pred)[0]
            lags = [amount_pred] + lags[:2]
            client_predictions[date_str] = - round(max(0, amount_pred), 2)

        predictions['target'][client_id_str] = client_predictions

    with open('predictions/predictions_4.json', 'w') as f:
        json.dump(predictions, f, indent=4)


if __name__ == "__main__":

    current_path = os.getcwd()
    root_dir_name = 'hackathon-caixabank-data-ai-report'
    root_dir_path = current_path[:current_path.index(root_dir_name) + len(root_dir_name)]
    os.chdir(root_dir_path)

    compute_fraud_predictions()
    compute_forecast_expenses_predictions()
