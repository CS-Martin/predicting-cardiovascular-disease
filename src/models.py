from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

def mlp_with_bmi(X_train_scaled_BMI, X_test_scaled_BMI, y_train, y_test):
    mlp_clf_bmi = MLPClassifier(hidden_layer_sizes=(26, 52, 26), alpha=0.001, max_iter=100, tol=0.001)

    mlp_clf_bmi.fit(X_train_scaled_BMI, y_train)

    y_pred = mlp_clf_bmi.predict(X_test_scaled_BMI)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    # 5-fold cross validation
    cv_results_bmi = cross_validate(mlp_clf_bmi, X_train_scaled_BMI, y_train, cv = 5, scoring = ["accuracy"], return_train_score=True)

    print('\n\nMLP Classifier with BMI\n')
    print(cv_results_bmi)
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(f"Confusion Matrix:\n{conf_matrix}")
    print(f"Classification Report:\n{class_report}")
    
def mlp_without_bmi(X_train_scaled, X_test_scaled, y_train, y_test):
    mlp_clf = MLPClassifier(hidden_layer_sizes=(26, 52, 26), alpha=0.001, max_iter=100, tol=0.001)

    mlp_clf.fit(X_train_scaled, y_train)

    y_pred = mlp_clf.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    # 5-fold cross validation
    cv_results = cross_validate(mlp_clf, X_train_scaled, y_train, cv = 5, scoring = ["accuracy"], return_train_score=True)

    print('\n\nMLP Classifier without BMI\n')
    print(cv_results)
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(f"Confusion Matrix:\n{conf_matrix}")
    print(f"Classification Report:\n{class_report}")
    
def logistic_regression_with_bmi(X_train_scaled_BMI, X_test_scaled_BMI, y_train, y_test):
    log_clf = LogisticRegression()

    log_clf.fit(X_train_scaled_BMI, y_train)

    y_pred = log_clf.predict(X_test_scaled_BMI)

    # Assuming y_true and y_pred are your true and predicted labels
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    print('\n\nLogistic Regression Classifier with BMI\n')
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(f"Confusion Matrix:\n{conf_matrix}")
    print(f"Classification Report:\n{class_report}")
    
def logistic_regression_without_bmi(X_train_scaled, X_test_scaled, y_train, y_test):
    log_clf = LogisticRegression()

    log_clf.fit(X_train_scaled, y_train)

    y_pred = log_clf.predict(X_test_scaled)

    # Assuming y_true and y_pred are your true and predicted labels
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    print('\n\nLogistic Regression Classifier without BMI\n')
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(f"Confusion Matrix:\n{conf_matrix}")
    print(f"Classification Report:\n{class_report}")
    
# Artificial Neural Network Classifier with 3 hidden layers with BMI
def ann_with_bmi(X_train_scaled_BMI, X_test_scaled_BMI, y_train, y_test):
    ann = tf.keras.models.Sequential([
    tf.keras.layers.Dense(12, activation=tf.nn.relu),
    tf.keras.layers.Dense(24, activation=tf.nn.relu),
    tf.keras.layers.Dense(48, activation=tf.nn.relu),
    tf.keras.layers.Dense(24, activation=tf.nn.relu),
    tf.keras.layers.Dense(2, activation=tf.nn.softmax)
    ])

    ann.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    ann.fit(X_train_scaled_BMI, y_train, epochs=10)

    # Use predict_classes to convert probabilities to class labels
    y_pred_prob = ann.predict(X_test_scaled_BMI)
    y_pred = tf.argmax(y_pred_prob, axis=1)

    # Assuming y_test is one-dimensional
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    print('\n\nArtificial Neural Network Classifier with BMI\n')
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(f"Confusion Matrix:\n{conf_matrix}")
    print(f"Classification Report:\n{class_report}")
    
# Artificial Neural Network Classifier with 3 hidden layers without BMI
def ann_without_bmi(X_train_scaled, X_test_scaled, y_train, y_test):
    ann = tf.keras.models.Sequential([
    tf.keras.layers.Dense(12, activation=tf.nn.relu),
    tf.keras.layers.Dense(24, activation=tf.nn.relu),
    tf.keras.layers.Dense(48, activation=tf.nn.relu),
    tf.keras.layers.Dense(24, activation=tf.nn.relu),
    tf.keras.layers.Dense(2, activation=tf.nn.softmax)
])

    ann.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    ann.fit(X_train_scaled, y_train, epochs=10)

    # Use predict_classes to convert probabilities to class labels
    y_pred_prob = ann.predict(X_test_scaled)
    y_pred = tf.argmax(y_pred_prob, axis=1)

    # Assuming y_test is one-dimensional
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(f"Confusion Matrix:\n{conf_matrix}")
    print(f"Classification Report:\n{class_report}")