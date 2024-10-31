# file for implementing api call functions
import requests
import pandas as pd
import os


def make_api_request_call(client_id: str, url: str) -> dict:

    params = {'client_id': client_id}
    response = requests.get(url, params=params)

    if response.status_code == 200:
        return response.json()['values']
    else:
        return {'error': f'Failed to retrieve data. Status code: {response.status_code}'}


if __name__ == "__main__":

    current_path = os.getcwd()
    root_dir_name = 'hackathon-caixabank-data-ai-report'
    root_dir_path = current_path[:current_path.index(root_dir_name) + len(root_dir_name)]
    os.chdir(root_dir_path)

    url_clients = 'https://faas-lon1-917a94a7.doserverless.co/api/v1/web/' \
                  'fn-a1f52b59-3551-477f-b8f3-de612fbf2769/default/clients-data'

    url_cards = 'https://faas-lon1-917a94a7.doserverless.co/api/v1/web/' \
                'fn-a1f52b59-3551-477f-b8f3-de612fbf2769/default/cards-data'

    transactions_data_df = pd.read_csv('data/raw/transactions_data.csv')

    clients_ids = list(set(transactions_data_df.client_id.to_list()))

    client_data_list = []
    card_data_list = []
    for client_id in clients_ids:

        client_data = make_api_request_call(client_id, url_clients)
        card_data = make_api_request_call(client_id, url_cards)

        if 'error' not in client_data:
            client_data['client_id'] = client_id
            client_data_list.append(client_data)

        if 'error' not in card_data.keys():
            for card_id, card_info in card_data.items():
                card_info['client_id'] = client_id
                card_info['card_id'] = card_id
                card_data_list.append(card_info)

    client_data_df = pd.DataFrame(client_data_list)
    card_data_df = pd.DataFrame(card_data_list)

    client_data_df.to_csv('data/raw/client_data.csv', index=False)
    card_data_df.to_csv('data/raw/card_data.csv', index=False)
