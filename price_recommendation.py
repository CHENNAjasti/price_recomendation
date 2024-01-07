import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

class DataProcessor:
    def __init__(self, file_path):
        self.dataframe = pd.read_csv(file_path, encoding='unicode_escape')

    def clean_data(self):
        self.dataframe.dropna(subset=['CustomerID'], axis=0, inplace=True)
        self.dataframe['Date'] = pd.to_datetime(self.dataframe['InvoiceDate'])
        self.dataframe['Month-Year'] = self.dataframe['Date'].dt.strftime('%b-%Y')
        self.dataframe.drop(['InvoiceDate', 'Date'], axis=1, inplace=True)
        self.dataframe.reset_index(drop=True, inplace=True)

        # Create and drop 'Total Price' column
        self.dataframe['Total Price'] = self.dataframe['UnitPrice'] * self.dataframe['Quantity']
        self.dataframe.drop(['Total Price'], axis=1, inplace=True)

        self.dataframe['Description'].value_counts()

    def visualize_top_selling_items(self):
        columns = ["Description", "Quantity", "UnitPrice"]
        dataframe_quan = self.dataframe.groupby('Description')['Quantity'].sum().reset_index()
        dataframe_quan.columns = ['Description', 'Total Quantity']
        dataframe_quan_top15 = dataframe_quan.nlargest(15, 'Total Quantity')

        plt.figure(figsize=(15, 10))
        sns.barplot(data=dataframe_quan_top15, x="Total Quantity", y="Description", capsize=3, palette="magma")
        plt.title("Top 10 Items Sold by Total Quantity")
        plt.xlabel("Total Quantity")
        plt.ylabel("Description")
        plt.show()

    def encode_features(self):
        label_en = LabelEncoder()
        self.dataframe['StockCode_Encode'] = label_en.fit_transform(self.dataframe['StockCode'])
        self.dataframe['Invoice_Encode'] = label_en.fit_transform(self.dataframe['InvoiceNo'])

    def get_features_and_target(self):
        X = self.dataframe.drop(columns=['InvoiceNo', 'StockCode', 'Month-Year', 'Description', 'CustomerID', 'Country'])
        y = self.dataframe['Total Price']
        return X, y

class ModelTrainer:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def scale_features(self):
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)

    def train_and_evaluate_models(self, models):
        for m in models:
            m.fit(self.X_train, self.y_train)
            y_pred = m.predict(self.X_test)
            r2_value = r2_score(self.y_test, y_pred)
            print(f"{m.__class__.__name__} R2 Score: {r2_value}")

    def lasso_alpha_variation(self):
        r2_lasso = []
        for x in range(10, 500, 10):
            model_l = Lasso(alpha=x)
            model_l.fit(self.X_train, self.y_train)
            y_pred = model_l.predict(self.X_test)
            r2_value = r2_score(self.y_test, y_pred)
            r2_lasso.append(r2_value)

        # Plotting Lasso Regression R2 Score variation
        sns.set_style("whitegrid")
        plt.figure(figsize=(10, 6))
        sns.lineplot(x=list(range(10, 400, 10)), y=r2_lasso[:39], marker='o', color='blue', linestyle='dashed')
        plt.title('R2 Score variation for Lasso Regression Model', fontsize=14)
        plt.xlabel('Alpha value', fontsize=14)
        plt.ylabel('R2 Score', fontsize=14)
        plt.show()


def main():
    # Load and clean data
    data_processor = DataProcessor(r"C:\Users\chenn\OneDrive\Documents\HTML docs\intern\data.csv")
    data_processor.clean_data()

    # Visualize top selling items
    data_processor.visualize_top_selling_items()

    # Encode features
    data_processor.encode_features()

    # Train-test split
    X, y = data_processor.get_features_and_target()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model training and evaluation
    model_trainer = ModelTrainer(X_train, X_test, y_train, y_test)
    model_trainer.scale_features()

    models = [
        LinearRegression(),
        RandomForestRegressor(n_estimators=100, random_state=42),
        Lasso(alpha=10),
        Ridge(alpha=10)
    ]

    model_trainer.train_and_evaluate_models(models)
    model_trainer.lasso_alpha_variation()


if __name__ == "__main__":
    main()
