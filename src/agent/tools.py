import os
from pandas import DataFrame
from fpdf import FPDF
from datetime import datetime


def create_pdf_report(
        client_id: int,
        start_date: str,
        end_date: str,
        earnings_df: DataFrame,
        expenses_df: DataFrame,
        cash_flow_df: DataFrame
) -> None:
    """
        Generates a comprehensive PDF financial report for a specific client over a defined period.

        The report includes summaries of earnings and expenses, visualizations, and a cash flow overview.

        The report structure includes:
            - Title and client information.
            - Earnings and Expenses Summary table.
            - Earnings and Expenses visualization image.
            - Expenses by Merchant Category table.
            - Expenses Summary visualization image.
            - Cash Flow Summary table.

        The generated PDF is saved in the "reports/" directory with a filename indicating the client ID.

        Parameters
        ----------
        client_id : int
            Unique identifier for the client for whom the report is being generated.
        start_date : str
            Start date of the reporting period in the format "YYYY-MM-DD".
        end_date : str
            End date of the reporting period in the format "YYYY-MM-DD".
        earnings_df : pandas.DataFrame
            DataFrame containing the client's earnings data within the specified period.
        expenses_df : pandas.DataFrame
            DataFrame containing the client's expenses data within the specified period.
        cash_flow_df : pandas.DataFrame
            DataFrame summarizing the client's cash flow over the specified period.

        Returns
        -------
        None
            The function generates and saves a PDF report but does not return any value.
        """

    pdf_output_folder = "reports/"
    os.makedirs(pdf_output_folder, exist_ok=True)
    pdf_filename = f"{pdf_output_folder}financial_report_client_{client_id}.pdf"

    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Helvetica", 'B', 16)
    pdf.cell(0, 10, f"Financial Report for Client {client_id}", align='C')
    pdf.ln(10)

    pdf.set_font('Helvetica', '', 12)
    pdf.cell(0, 10, f'Period: {start_date} to {end_date}')
    pdf.ln(10)

    pdf.set_font("Helvetica", 'B', 12)
    pdf.cell(0, 10, "Earnings and Expenses Summary")
    pdf.ln(10)
    pdf.set_font("Helvetica", size=10)
    pdf.multi_cell(0, 10, earnings_df.to_string(index=False))
    pdf.ln(10)

    pdf.image('reports/figures/earnings_and_expenses.png', x=10, w=190)
    pdf.ln(10)

    pdf.set_font("Helvetica", 'B', 12)
    pdf.cell(0, 10, "Expenses by Merchant Category")
    pdf.ln(10)
    pdf.set_font("Helvetica", size=10)
    pdf.multi_cell(0, 10, expenses_df.to_string(index=False))
    pdf.ln(10)

    pdf.image('reports/figures/expenses_summary.png', x=10, w=190)
    pdf.ln(10)

    pdf.set_font("Helvetica", 'B', 12)
    pdf.cell(0, 10, "Cash Flow Summary")
    pdf.ln(10)
    pdf.set_font("Helvetica", size=10)
    pdf.multi_cell(0, 10, cash_flow_df.to_string(index=False))
    pdf.ln(10)

    pdf.output(pdf_filename)

    print(f"Report generated and saved as {pdf_filename}")


def is_date_format(date: str) -> bool:
    """
    Validates whether a given string adheres to the "YYYY-MM-DD" date format.

    This function attempts to parse the input string using the specified date format. If parsing is successful without raising a `ValueError`, the function returns `True`, indicating that the string is in the correct format. Otherwise, it returns `False`.

    Parameters
    ----------
    date : str
        The date string to validate. Expected format is "YYYY-MM-DD".

    Returns
    -------
    bool
        `True` if the input string matches the "YYYY-MM-DD" format, `False` otherwise.
    """
    try:
        datetime.strptime(date, '%Y-%m-%d')
        return True

    except ValueError:
        return False


def is_client(df: DataFrame, client_id: str) -> bool:
    """
    Determines whether a given client ID exists within the provided DataFrame.

    This function checks if the specified `client_id` is present in the 'client_id' column of the DataFrame. It handles data type casting to ensure accurate comparison.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing client data with a 'client_id' column.
    client_id : str
        The client ID to verify. Will be cast to the DataFrame's 'client_id' dtype if possible.

    Returns
    -------
    bool
        `True` if the `client_id` exists in the DataFrame, `False` otherwise.
    """

    clients_ids = set(df.client_id.unique())

    if client_id in clients_ids:
        return True

    else:
        return False


if __name__ == "__main__":
    ...
