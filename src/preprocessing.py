from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def data_preprocessing(df):
    COLUMNS = df.columns

    # Separate features and target variable
    X = df[COLUMNS[:-1]]
    y = df[COLUMNS[-1]]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Remove BMI
    X_train_without_BMI = X_train.drop(columns=['BMI'])
    X_test_without_BMI = X_test.drop(columns=['BMI'])

    # Standardize the features using StandardScaler
    scaler = StandardScaler()

    # Without BMI
    X_train_scaled = scaler.fit_transform(X_train_without_BMI)
    X_test_scaled = scaler.transform(X_test_without_BMI)

    # With BMI
    X_train_scaled_BMI = scaler.fit_transform(X_train)
    X_test_scaled_BMI = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, X_train_scaled_BMI, X_test_scaled_BMI