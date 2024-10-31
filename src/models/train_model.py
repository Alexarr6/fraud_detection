import json
import os
import pickle
from pandas import DataFrame, merge, get_dummies, read_csv, to_datetime, concat
import numpy as np
import matplotlib.pyplot as plt
from sdv.single_table import CTGANSynthesizer
from imblearn.over_sampling import SMOTE
from sdv.metadata import Metadata
from sdv.evaluation.single_table import run_diagnostic
from sdv.evaluation.single_table import evaluate_quality
from sklearn.metrics import f1_score, precision_score, recall_score, balanced_accuracy_score, \
    mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
import lightgbm as lgb
import shap

# plt.switch_backend('Agg')
plt.switch_backend('TkAgg')


class FraudDetectionModel:

    @staticmethod
    def load_data() -> tuple:
        transactions_df = read_csv('data/raw/transactions_data.csv')
        cards_df = read_csv('data/raw/card_data.csv')
        clients_df = read_csv('data/raw/client_data.csv')

        return transactions_df, cards_df, clients_df

    @staticmethod
    def load_labels() -> tuple:
        with open('data/raw/train_fraud_labels.json', 'r') as f:
            train_fraud_labels = json.load(f)

        return train_fraud_labels

    @staticmethod
    def merge_data(transactions_df: DataFrame, cards_df: DataFrame, clients_df: DataFrame) -> DataFrame:
        """
        Merges transaction data with card and client data.

        Parameters
        ----------
        transactions_df : pandas DataFrame
            DataFrame containing transaction data.
        cards_df : pandas DataFrame
            DataFrame containing card data.
        clients_df : pandas DataFrame
            DataFrame containing client data.

        Returns
        -------
        pandas DataFrame
            Merged DataFrame containing transactions enriched with card and client information.
        """

        transactions_df = merge(
            transactions_df,
            cards_df[
                ['client_id', 'card_id', 'card_brand', 'card_on_dark_web', 'card_type', 'credit_limit', 'has_chip',
                 'num_cards_issued', 'acct_open_date', 'expires', 'year_pin_last_changed']
            ],
            on=['client_id', 'card_id'],
            how='inner'
        )

        transactions_df = merge(
            transactions_df,
            clients_df[
                ['client_id', 'credit_score', 'current_age', 'gender', 'num_credit_cards', 'per_capita_income',
                 'total_debt', 'yearly_income']],
            on='client_id',
            how='inner'
        )

        return transactions_df

    @staticmethod
    def _transform_salary_to_number(transactions_df: DataFrame, feature: str) -> DataFrame:
        """
        Converts currency-formatted strings to numeric values in the specified feature column.

        Parameters
        ----------
        transactions_df : pandas DataFrame
            DataFrame containing the data.
        feature : str
            Name of the feature/column to transform.

        Returns
        -------
        pandas DataFrame
            DataFrame with the specified feature converted to numeric values.
        """

        transactions_df[feature] = transactions_df[feature].replace('[\$,]', '', regex=True).astype(float)
        return transactions_df

    def transform_data(self, transactions_df: DataFrame) -> DataFrame:
        """
        Transforms and engineers features in the transactions DataFrame.

        The method performs the following operations:
        - Converts currency columns to numeric.
        - Extracts time-based features from date columns.
        - Calculates account age, card expiry, and PIN change durations.
        - Processes the MCC (Merchant Category Code) column.
        - Extracts error-related features.
        - Calculates utilization rate.
        - Drops unnecessary columns.
        - Encodes categorical variables using one-hot encoding.

        Parameters
        ----------
        transactions_df : pandas DataFrame
            DataFrame containing merged transaction, card, and client data.

        Returns
        -------
        pandas DataFrame
            Transformed DataFrame ready for model training.
        """

        transactions_df = self._transform_salary_to_number(transactions_df, 'amount')
        transactions_df = self._transform_salary_to_number(transactions_df, 'total_debt')
        transactions_df = self._transform_salary_to_number(transactions_df, 'yearly_income')
        transactions_df = self._transform_salary_to_number(transactions_df, 'per_capita_income')
        transactions_df = self._transform_salary_to_number(transactions_df, 'credit_limit')

        transactions_df['date'] = to_datetime(transactions_df['date'])

        transactions_df['transaction_hour'] = transactions_df['date'].dt.hour
        transactions_df['transaction_dayofweek'] = transactions_df['date'].dt.dayofweek

        transactions_df['acct_open_date'] = to_datetime(transactions_df['acct_open_date'],
                                                        format='%m/%Y', errors='coerce')
        transactions_df['account_age_days'] = (transactions_df['date'] - transactions_df['acct_open_date']).dt.days

        transactions_df['expires'] = to_datetime(transactions_df['expires'], format='%m/%Y', errors='coerce')
        transactions_df['card_expires_in_days'] = (transactions_df['expires'] - transactions_df['date']).dt.days

        transactions_df['year_pin_last_changed'] = to_datetime(transactions_df['year_pin_last_changed'],
                                                               format='%Y', errors='coerce')
        transactions_df['pin_last_changed_days'] = (transactions_df['date'] - transactions_df['year_pin_last_changed'])\
            .dt.days

        transactions_df['mcc'] = transactions_df['mcc'].astype(str)
        '''
        relevant_mcc_codes = ['4784', '5499', '5541', '5300', '4900']
        for mcc_code in relevant_mcc_codes:
            transactions_df[f'mcc_{mcc_code}'] = (transactions_df['mcc'] == mcc_code).astype(int)
        '''

        transactions_df['error_count'] = transactions_df['errors'].apply(lambda x: len(str(x).split(',')) if x else 0)
        transactions_df['has_errors'] = transactions_df['error_count'].apply(lambda x: 1 if x > 0 else 0)

        transactions_df['utilization_rate'] = transactions_df['amount'] / (transactions_df['credit_limit'] + 1e-5)

        transactions_df = transactions_df.drop(columns=[
            'date', 'merchant_city', 'merchant_state', 'merchant_id', 'acct_open_date', 'expires',
            'year_pin_last_changed', 'errors'
        ])

        transactions_df = get_dummies(
            transactions_df,
            columns=['use_chip', 'gender', 'card_brand', 'card_on_dark_web', 'card_type', 'has_chip', 'mcc'],
            drop_first=True,
            dtype=int
        )

        return transactions_df

    @staticmethod
    def add_labels(transactions_df: DataFrame, train_fraud_labels: dict) -> DataFrame:
        """
        Adds fraud labels to the transactions DataFrame by merging with the labels data.

        Parameters
        ----------
        transactions_df : pandas DataFrame
            DataFrame containing transaction data.
        train_fraud_labels : dict
            Dictionary containing fraud labels with transaction IDs as keys.

        Returns
        -------
        pandas DataFrame
            DataFrame with the 'is_fraud' label added.
        """

        train_fraud_df = DataFrame(list(train_fraud_labels['target'].items()), columns=['id', 'is_fraud'])
        train_fraud_df['id'] = train_fraud_df['id'].astype(int)
        train_fraud_df['is_fraud'] = train_fraud_df['is_fraud'].map({'Yes': 1, 'No': 0})

        train_transactions_df = transactions_df.merge(train_fraud_df, on='id', how='inner')

        return train_transactions_df

    @staticmethod
    def create_smote_synthetic_data(X_train_df: DataFrame, y_train: DataFrame) -> tuple:
        """
        Generates synthetic samples using SMOTE to address class imbalance.

        Parameters
        ----------
        X_train_df : pandas DataFrame
            Training feature data.
        y_train : pandas DataFrame or Series
            Training labels.

        Returns
        -------
        tuple
            Tuple containing resampled feature data and labels (X_train_resampled, y_train_resampled).
        """

        imputer = SimpleImputer(strategy='mean')
        x = imputer.fit_transform(X_train_df)

        smote = SMOTE(random_state=1)
        X_train_resampled, y_train_resampled = smote.fit_resample(x, y_train)

        return X_train_resampled, y_train_resampled

    @staticmethod
    def create_ctgan_synthetic_data(X_train_df: DataFrame, y_train: DataFrame) -> tuple:
        """
        Generates synthetic fraud samples using CTGAN to augment the minority class.

        Parameters
        ----------
        X_train_df : pandas DataFrame
            Training feature data.
        y_train : pandas DataFrame or Series
            Training labels.

        Returns
        -------
        tuple
            Tuple containing augmented training feature data and labels (X_train, y_train).
        """

        train_transactions_df = X_train_df.copy()
        train_transactions_df['is_fraud'] = y_train

        fraud_data_df = train_transactions_df[train_transactions_df['is_fraud'] == 1].reset_index(drop=True)

        metadata = Metadata.detect_from_dataframe(
            data=fraud_data_df,
            table_name='fraud_data'
        )

        synthesizer = CTGANSynthesizer(metadata=metadata)
        synthesizer.fit(fraud_data_df)

        print('Model fitted')
        print()

        synthetic_data = synthesizer.sample(num_rows=len(fraud_data_df) * 50)

        print('Data created')
        print(synthetic_data.head())

        diagnostic = run_diagnostic(
            real_data=fraud_data_df,
            synthetic_data=synthetic_data,
            metadata=metadata
        )
        quality_report = evaluate_quality(
            fraud_data_df,
            synthetic_data,
            metadata
        )

        print(diagnostic)
        print(quality_report)

        ctgan_train_transactions_df = concat([train_transactions_df, synthetic_data], ignore_index=True)

        X_train = ctgan_train_transactions_df.drop(columns=['is_fraud'])
        y_train = ctgan_train_transactions_df['is_fraud']

        print('Previous length:', len(fraud_data_df))
        print('Current length:', len(synthetic_data) + len(fraud_data_df))

        return X_train, y_train

    @staticmethod
    def create_train_test_sets(train_transactions_df: DataFrame) -> tuple:
        """
        Splits the data into training and testing sets.

        Parameters
        ----------
        train_transactions_df : pandas DataFrame
            DataFrame containing transaction data with labels.

        Returns
        -------
        tuple
            Tuple containing training features, testing features, training labels, and testing labels.
            (X_train_df, X_test_df, y_train, y_test)
        """

        X = train_transactions_df.drop(columns=['id', 'is_fraud'])
        y = train_transactions_df['is_fraud']
        X_train_df, X_test_df, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)

        return X_train_df, X_test_df, y_train, y_test

    @staticmethod
    def train_model(X_train_df: DataFrame, y_train: DataFrame) -> XGBClassifier:
        """
        Trains an XGBoost classifier on the training data.

        Parameters
        ----------
        X_train_df : pandas DataFrame
            Training feature data.
        y_train : pandas DataFrame or Series
            Training labels.

        Returns
        -------
        XGBClassifier
            Trained XGBoost classifier model.
        """

        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        model.fit(X_train_df, y_train)

        return model

    @staticmethod
    def _select_threshold(model: XGBClassifier, X_test_df: DataFrame, y_test: DataFrame) -> float:
        """
        Selects the optimal classification threshold based on balanced accuracy.

        Parameters
        ----------
        model : XGBClassifier
            Trained classifier model.
        X_test_df : pandas DataFrame
            Testing feature data.
        y_test : pandas DataFrame or Series
            Testing labels.

        Returns
        -------
        float
            Optimal threshold value for classification.
        """

        y_test_probs = model.predict_proba(X_test_df)[:, 1]

        alphas = np.linspace(0, 1, 100)

        best_alpha = 0
        best_balanced_accuracy = 0

        for alpha in alphas:
            y_test_pred_threshold = (y_test_probs >= alpha).astype(int)

            balanced_accuracy = balanced_accuracy_score(y_test, y_test_pred_threshold)
            print("Best Balanced Accuracy Score:", balanced_accuracy)
            print('Alpha:', alpha)
            print()

            if balanced_accuracy > best_balanced_accuracy:
                best_balanced_accuracy = balanced_accuracy
                best_alpha = alpha

        print("Best Alpha:", best_alpha)
        print("Best Balanced Accuracy Score:", best_balanced_accuracy)

        return best_alpha

    def compute_metrics(
            self,
            model: XGBClassifier,
            X_train_df: DataFrame,
            X_test_df: DataFrame,
            y_train: DataFrame,
            y_test: DataFrame
    ) -> None:
        """
        Computes and prints performance metrics for the model.

        Parameters
        ----------
        model : XGBClassifier
            Trained classifier model.
        X_train_df : pandas DataFrame
            Training feature data.
        X_test_df : pandas DataFrame
            Testing feature data.
        y_train : pandas DataFrame or Series
            Training labels.
        y_test : pandas DataFrame or Series
            Testing labels.

        Returns
        -------
        None
        """

        # best_alpha = self._select_threshold(model, X_test_df, y_test)

        best_alpha = 0.5

        y_train_probs = model.predict_proba(X_train_df)[:, 1]
        y_test_probs = model.predict_proba(X_test_df)[:, 1]

        y_train_pred = (y_train_probs >= best_alpha).astype(int)
        y_test_pred = (y_test_probs >= best_alpha).astype(int)

        train_f1 = f1_score(y_train, y_train_pred)
        test_f1 = f1_score(y_test, y_test_pred)
        train_precision = precision_score(y_train, y_train_pred)
        test_precision = precision_score(y_test, y_test_pred)
        train_recall = recall_score(y_train, y_train_pred)
        test_recall = recall_score(y_test, y_test_pred)
        train_balanced_accuracy_score = balanced_accuracy_score(y_train, y_train_pred)
        test_balanced_accuracy_score = balanced_accuracy_score(y_test, y_test_pred)

        print("F1-Score (Train):", train_f1)
        print("F1-Score (Test):", test_f1)
        print("Precision (Train):", train_precision)
        print("Precision (Test):", test_precision)
        print("Recall (Train):", train_recall)
        print("Recall (Test):", test_recall)
        print("Balance accuracy score (Train):", train_balanced_accuracy_score)
        print("Balance accuracy score (Test):", test_balanced_accuracy_score)

    @staticmethod
    def compute_shap_values(model: XGBClassifier, X_sample: DataFrame):
        """
        Computes and plots SHAP values for feature importance analysis.

        Parameters
        ----------
        model : XGBClassifier
            Trained classifier model.
        X_sample : pandas DataFrame
            Sample of feature data to compute SHAP values.

        Returns
        -------
        None
        """

        if len(X_sample) > 1000:
            X_sample = X_sample.sample(n=1000, random_state=42)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

        shap.summary_plot(shap_values, X_sample, plot_type="bar")

        shap.summary_plot(shap_values, X_sample)
        plt.show()

    @staticmethod
    def save_model(model: XGBClassifier) -> None:
        """
        Saves the trained model to a file using pickle.

        Parameters
        ----------
        model : XGBClassifier
            Trained classifier model.

        Returns
        -------
        None
        """

        with open('models/xgb_fraud_detection_model.pkl', 'wb') as f:
            pickle.dump(model, f)

    def main(self, save: bool, smote: bool, ctgan: bool) -> None:
        """
        Main method to execute the fraud detection model pipeline.

        Parameters
        ----------
        save : bool
            If True, processes raw data and saves the transformed data. If False, loads preprocessed data.
        smote : bool
            If True, uses SMOTE for synthetic data generation.
        ctgan : bool
            If True, uses CTGAN for synthetic data generation.

        Returns
        -------
        None
        """

        train_fraud_labels = self.load_labels()

        if save:
            transactions_df, cards_df, clients_df = self.load_data()
            transactions_df = self.merge_data(transactions_df, cards_df, clients_df)
            transactions_df = self.transform_data(transactions_df)
            transactions_df.to_csv('data/processed/processed_transactions_data.csv', index=False)

        else:
            transactions_df = read_csv('data/processed/processed_transactions_data.csv')

        train_transactions_df = self.add_labels(transactions_df, train_fraud_labels)
        X_train_df, X_test_df, y_train, y_test = self.create_train_test_sets(train_transactions_df)

        if smote:
            smote_X_train, smote_y_train = self.create_smote_synthetic_data(X_train_df, y_train)
            model = self.train_model(smote_X_train, smote_y_train)

        elif ctgan:
            ctgan_X_train, ctgan_y_train = self.create_ctgan_synthetic_data(X_train_df, y_train)
            model = self.train_model(ctgan_X_train, ctgan_y_train)

        else:
            model = self.train_model(X_train_df, y_train)

        self.compute_metrics(model, X_train_df, X_test_df, y_train, y_test)
        self.save_model(model)
        self.compute_shap_values(model, X_train_df)


class ForecastPredictionModel:

    @staticmethod
    def load_data() -> tuple:
        transactions_df = read_csv('data/raw/transactions_data.csv')
        clients_df = read_csv('data/raw/client_data.csv')

        return transactions_df, clients_df

    def merge_data(self, monthly_expenses_df: DataFrame, clients_df: DataFrame) -> DataFrame:
        """
        Merges the monthly expenses data with client data and processes it.

        Parameters
        ----------
        monthly_expenses_df : pandas DataFrame
            DataFrame containing monthly expenses per client.
        clients_df : pandas DataFrame
            DataFrame containing client data.

        Returns
        -------
        pandas DataFrame
            Merged DataFrame with monthly expenses enriched with client information.
        """

        monthly_expenses_df = merge(
            monthly_expenses_df,
            clients_df[
                ['client_id', 'credit_score', 'total_debt', 'per_capita_income', 'yearly_income', 'gender']],
            on='client_id',
            how='left'
        )

        monthly_expenses_df = self._transform_salary_to_number(monthly_expenses_df, 'total_debt')
        monthly_expenses_df = self._transform_salary_to_number(monthly_expenses_df, 'per_capita_income')
        monthly_expenses_df = self._transform_salary_to_number(monthly_expenses_df, 'yearly_income')

        label_encoder = LabelEncoder()
        monthly_expenses_df['gender'] = label_encoder.fit_transform(monthly_expenses_df['gender'].astype(str))

        return monthly_expenses_df

    @staticmethod
    def _transform_salary_to_number(df: DataFrame, feature: str) -> DataFrame:
        """
        Converts currency-formatted strings to numeric values in the specified feature column.

        Parameters
        ----------
        df : pandas DataFrame
            DataFrame containing the data.
        feature : str
            Name of the feature/column to transform.

        Returns
        -------
        pandas DataFrame
            DataFrame with the specified feature converted to numeric values.
        """

        df[feature] = df[feature].replace('[\$,]', '', regex=True).astype(float)
        return df

    def transform_data(self, transactions_df: DataFrame) -> DataFrame:
        """
        Transforms the transactions data to prepare for modeling.

        The method performs the following operations:
        - Converts 'amount' column to numeric.
        - Filters expenses (negative amounts) and processes dates.
        - Aggregates expenses on a monthly basis per client.
        - Generates lag features for previous months' expenses.

        Parameters
        ----------
        transactions_df : pandas DataFrame
            DataFrame containing transaction data.

        Returns
        -------
        pandas DataFrame
            DataFrame containing monthly expenses with lag features.
        """

        transactions_df = self._transform_salary_to_number(transactions_df, 'amount')
        transactions_df['date'] = to_datetime(transactions_df['date'])

        transactions_df['amount'] = transactions_df['amount'].astype(float)
        expenses_df = transactions_df[transactions_df['amount'] < 0].copy()

        expenses_df['month'] = expenses_df['date'].dt.to_period('M')
        monthly_expenses_df = expenses_df.groupby(['client_id', 'month'])['amount'].sum().reset_index()
        monthly_expenses_df['amount'] = -monthly_expenses_df['amount']

        monthly_expenses_df['month_number'] = monthly_expenses_df['month'].dt.month
        monthly_expenses_df['year'] = monthly_expenses_df['month'].dt.year

        monthly_expenses_df = monthly_expenses_df.sort_values(['client_id', 'month'])
        for lag in range(1, 4):
            monthly_expenses_df[f'lag_{lag}'] = monthly_expenses_df.groupby('client_id')['amount'].shift(lag)
        monthly_expenses_df.fillna(0, inplace=True)

        return monthly_expenses_df

    def train_model(self, monthly_expenses_df: DataFrame) -> lgb.basic.Booster:
        """
        Trains a LightGBM regression model to predict future expenses.

        Parameters
        ----------
        monthly_expenses_df : pandas DataFrame
            DataFrame containing monthly expenses and features.

        Returns
        -------
        lgb.basic.Booster
            Trained LightGBM model.
        """

        features = ['month_number', 'year', 'lag_1', 'lag_2', 'lag_3', 'credit_score',
                    'total_debt', 'per_capita_income', 'yearly_income', 'gender']

        train_df = monthly_expenses_df[monthly_expenses_df['month'] < monthly_expenses_df['month'].max() - 1]
        val_df = monthly_expenses_df[monthly_expenses_df['month'] >= monthly_expenses_df['month'].max() - 1]

        X_train = train_df[features]
        y_train = train_df['amount']
        X_val = val_df[features]
        y_val = val_df['amount']

        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_eval = lgb.Dataset(X_val, y_val, reference=lgb_train)

        model = lgb.train(
            {'objective': 'regression', 'metric': 'r2', 'verbose': -1},
            lgb_train,
            valid_sets=[lgb_train, lgb_eval],
            num_boost_round=1000
        )

        self._evaluate_model(model, X_val, y_val)

        return model

    @staticmethod
    def _evaluate_model(model, X_val: DataFrame, y_val: DataFrame) -> None:
        """
        Evaluates the model on validation data and prints performance metrics.

        Parameters
        ----------
        model : lgb.basic.Booster
            Trained LightGBM model.
        X_val : pandas DataFrame
            Validation feature data.
        y_val : pandas DataFrame or Series
            Validation target data.

        Returns
        -------
        None
        """

        y_pred = model.predict(X_val)
        r2 = r2_score(y_val, y_pred)
        mae = mean_absolute_error(y_val, y_pred)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        print(f"R2: {r2}, MAE: {mae}, RMSE: {rmse}")

    @staticmethod
    def save_model(model: lgb.basic.Booster) -> None:
        """
        Saves the trained model to a file using pickle.

        Parameters
        ----------
        model : lgb.basic.Booster
            Trained LightGBM model.

        Returns
        -------
        None
        """

        with open('models/xgb_forecast_expenses_model.pkl', 'wb') as f:
            pickle.dump(model, f)

    def main(self, save: bool) -> None:
        """
        Main method to execute the expense forecasting pipeline.

        Parameters
        ----------
        save : bool
            If True, processes raw data and saves the transformed data. If False, loads preprocessed data.

        Returns
        -------
        None
        """

        if save:
            transactions_df, clients_df = self.load_data()
            monthly_expenses_df = self.transform_data(transactions_df)
            monthly_expenses_df = self.merge_data(monthly_expenses_df, clients_df)

            monthly_expenses_df.to_csv('data/processed/monthly_expenses_data.csv', index=False)

        else:
            monthly_expenses_df = read_csv('data/processed/monthly_expenses_data.csv')
            monthly_expenses_df['month'] = to_datetime(monthly_expenses_df['month']).dt.to_period('M')

        model = self.train_model(monthly_expenses_df)
        self.save_model(model)


if __name__ == "__main__":
    current_path = os.getcwd()
    root_dir_name = 'hackathon-caixabank-data-ai-report'
    root_dir_path = current_path[:current_path.index(root_dir_name) + len(root_dir_name)]
    os.chdir(root_dir_path)

    save = True
    smote = True
    ctgan = False

    fraud_detection_model = FraudDetectionModel()
    fraud_detection_model.main(save, smote, ctgan)

    forecast_prediction_model = ForecastPredictionModel()
    forecast_prediction_model.main(save)
