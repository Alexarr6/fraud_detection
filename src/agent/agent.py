import os

import pandas as pd
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from src.agent.tools import is_client, is_date_format, create_pdf_report
from src.data.data_functions import earnings_and_expenses, expenses_summary, cash_flow_summary


def run_agent(df: pd.DataFrame, client_id: int, input: str) -> dict:

    """
    Create a simple AI Agent that generates PDF reports using the three functions from Task 2 (src/data/data_functions.py).
    The agent should generate a PDF report only if valid data is available for the specified client_id and date range.
    Using the data and visualizations from Task 2, the report should be informative and detailed.

    The agent should return a dictionary containing the start and end dates, the client_id, and whether the report was successfully created.

    Parameters
    ----------
    df : pandas DataFrame
        DataFrame  of the data to be used for the agent.
    client_id : int
        Id of the client making the request.
    input : str
        String with the client input for creating the report.


    Returns
    -------
    variables_dict : dict
        Dictionary of the variables of the query.
            {
                "start_date": "YYYY-MM-DD",
                "end_date" : "YYYY-MM-DD",
                "client_id": int,
                "create_report" : bool
            }

    """

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                Your task is to extract the `start_date` and `end_date` from the user input in `YYYY-MM-DD` format.
                Respond only in JSON format, showing just the dates, with no extra text. This is the most important
                part.
                
                Reasoning:
                For easy problems-
                    Make a simple plan and use COT
                    
                For moderate to hard problems-
                    1. Devise a step-by-step plan to solve the problem. (don't actually start solving yet, just make a plan)
                    2. Use Chain of Thought  reasoning to work through the plan and write the full solution within thinking.
                    
                Try to get the days, months and years and select the start_date and end_date carefully.
                
                Description:
                    1. Extract start_date and end_date from user input in YYYY-MM-DD format
                    
                Important:
                    - Be precise in date extraction
                    - Just reply the JSON
                    
                Reply in JSON format just the dates, without other sentences:
                
                {{
                    "start_date": "YYYY-MM-DD",
                    "end_date": "YYYY-MM-DD"
                }}
                
                Example 1: 
                
                    Human: 
                        Create a pdf report for the 6 month of 2018
                    AI: 
                        {{
                            "start_date": "2018-06-01",
                            "end_date": "2018-06-30"
                        }}

                Example 2: 
       
                    Human: 
                        Create a pdf report from 2017-01-01 to 2017-03-31
                    AI: 
                        {{
                            "start_date": "2017-01-01",
                            "end_date": "2017-03-31"
                        }}
                        
                Example 3: 
       
                    Human: 
                        Report for Q1 2019.
                    AI: 
                        {{
                            "start_date": "2019-01-01",
                            "end_date": "2019-03-31"
                        }}
                
                 """,
            ),
            ("human", "{input}"),
        ]
    )

    prompt_2 = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                Your task is to extract the `start_date` and `end_date` from the user input in `YYYY-MM-DD` format.
                Respond only in JSON format, showing just the dates, with no extra text. This is the most important
                part.

                Reasoning:
                For easy problems-
                    Make a simple plan and use COT

                For moderate to hard problems-
                    1. Devise a step-by-step plan to solve the problem. (don't actually start solving yet, just make a plan)
                    2. Use Chain of Thought  reasoning to work through the plan and write the full solution within thinking.

                Try to get the days, months and years and select the start_date and end_date carefully.

                Description:
                    1. Extract start_date and end_date from user input in YYYY-MM-DD format

                Important:
                    - Be precise in date extraction
                    - Just reply the JSON

                Reply in JSON format just the dates, without other sentences:

                {{
                    "start_date": "YYYY-MM-DD",
                    "end_date": "YYYY-MM-DD"
                }}

                Example 1: 
       
                    Human: 
                        Report for Q1 2019.
                    AI: 
                        {{
                            "start_date": "2019-01-01",
                            "end_date": "2019-03-31"
                        }}
                     
                Example 2: 
       
                    Human: 
                        Generate a report from January to March 2021.
                    AI: 
                        {{
                            "start_date": "2021-01-01",
                            "end_date": "2021-03-31"
                        }}
                        
                Example 3: 
                
                    Human: 
                        Create a pdf report for the fifth  month of 2019
                    AI: 
                        {{
                            "start_date": "2019-05-01",
                            "end_date": "2019-05-31"
                        }}
                 """,
            ),
            ("human", "{input}"),
        ]
    )

    model = ChatOllama(model="llama3.2:1b", temperature=0)
    model_2 = ChatOllama(model="llama3.2:1b", temperature=0)

    try:
        chain = prompt | model | JsonOutputParser()
        dates = chain.invoke({"input": input})
    except:
        try:
            chain = prompt_2 | model_2 | JsonOutputParser()
            dates = chain.invoke({"input": input})
        except:
            dates = {"start_date": 'YYYY-MM-DD', "end_date": 'YYYY-MM-DD'}

    if not is_date_format(dates['start_date']) or not dates['end_date']:
        variables_dict = {
            "start_date": 'YYYY-MM-DD',
            "end_date": 'YYYY-MM-DD',
            "client_id": client_id,
            "create_report": False
        }
        return variables_dict

    variables_dict = {
        "start_date": dates['start_date'],
        "end_date": dates['end_date'],
        "client_id": client_id,
        "create_report": False
    }

    if not is_client(df, client_id):
        return variables_dict

    try:
        earnings_df = earnings_and_expenses(df, client_id, dates['start_date'], dates['end_date'])
        expenses_df = expenses_summary(df, client_id, dates['start_date'], dates['end_date'])
        cash_flow_df = cash_flow_summary(df, client_id, dates['start_date'], dates['end_date'])

        create_pdf_report(client_id, dates['start_date'], dates['end_date'], earnings_df, expenses_df, cash_flow_df)
        variables_dict["create_report"] = True

    except Exception as e:
        print(f"Error generating report: {e}")
        variables_dict["create_report"] = False

    return variables_dict


if __name__ == "__main__":

    current_path = os.getcwd()
    root_dir_name = 'hackathon-caixabank-data-ai-report'
    root_dir_path = current_path[:current_path.index(root_dir_name) + len(root_dir_name)]
    os.chdir(root_dir_path)

    transactions_df = pd.read_csv('data/raw/transactions_data.csv')
    # variables_dict = run_agent(transactions_df, 1352, "Create a pdf report from 2018-01-01 to 2018-05-31")
    # variables_dict = run_agent(transactions_df, 122, "Create a pdf report for the fourth month of 2017")
    variables_dict = run_agent(transactions_df, 122, "Hola, que tal?")
