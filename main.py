import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.preprocessing import data_preprocessing
from src.models import mlp_with_bmi, mlp_without_bmi, logistic_regression_with_bmi, logistic_regression_without_bmi

def read_csv(filepath):
    try:
        return pd.read_csv(filepath, index_col='id', encoding="ISO-8859-1")
    except FileNotFoundError:
        print(f"Error: File not found at '{filepath}'.")
        exit()
    except Exception as e:
        print(f"Error: File not found at '{filepath}'")
        exit()
        
def main():
    # Read the data
    df = read_csv('dataset/cardio_train.csv')

    # Feature engineering
    bmi = df['weight'] / (df['height']/100) ** 2
    df.insert(4, 'BMI', bmi)

    # Make gender one hot encoding instead
    gender_dummies = pd.get_dummies(df['gender'], prefix='gender')
    position_to_insert = 2
    df = pd.concat([df.iloc[:, :position_to_insert], gender_dummies, df.iloc[:, position_to_insert+1:]], axis=1)
    df = df.drop('gender', axis=1)

    # Show correlation heatmap
    df_no_bmi = df.drop(columns=['BMI'])
    fig, ax = plt.subplots(figsize=(10, 8))  # Adjust the width and height as needed
    sns.heatmap(df_no_bmi.corr(), annot=True, ax=ax)
    plt.show()

    X_train_scaled, X_test_scaled, y_train, y_test, X_train_scaled_BMI, X_test_scaled_BMI = data_preprocessing(df)

    # Model Selection and Evaluation
    mlp_with_bmi(X_train_scaled_BMI, X_test_scaled_BMI, y_train, y_test)
    mlp_without_bmi(X_train_scaled, X_test_scaled, y_train, y_test)
    
    logistic_regression_with_bmi(X_train_scaled_BMI, X_test_scaled_BMI, y_train, y_test)
    logistic_regression_without_bmi(X_train_scaled, X_test_scaled, y_train, y_test)
    
if __name__ == '__main__':
    main()
