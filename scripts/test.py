import ember
import os
import lightgbm as lgb
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

# Load the EMBER feature set (already vectorized)
data_dir = "data"
#print("Loading the EMBER feature set")
#ember.create_vectorized_features(data_dir) # creates X_train.dat and y_train.dat, X_test.dat and y_test.dat

# Load the vectorized features
print("Loading vectorized features")
print("Loading training data")
X_train, y_train = ember.read_vectorized_features(data_dir, subset="train")
print("Loading test data")
X_test, y_test = ember.read_vectorized_features(data_dir, subset="test")


print("Training data shape: {}".format(X_train.shape))
print("Test data shape: {}".format(X_test.shape))

# Reduce the size of the dataset
print("Reducing the size of the dataset")
X_train, y_train = X_train[:400000], y_train[:400000]
X_test, y_test = X_test[:100000], y_test[:100000]

print(f"Training Data Distribution: {sum(y_train)} positive, {len(y_train) - sum(y_train)} negative")



# Optimise params
#print("Optimising model parameters")
"""
params = {
    "boosting": "gbdt",
    "objective": "binary",
    "num_iterations": 100,
    "learning_rate": 0.1,
    "num_leaves": 31,  # Default and sufficient for most cases
    "max_depth": -1,   # Unlimited depth to let num_leaves decide complexity
    "min_data_in_leaf": 20,  # Minimum number of samples in a leaf
    "feature_fraction": 0.8,  # Use 80% of features per iteration
    "bagging_fraction": 0.8,  # Use 80% of data per iteration
    "bagging_freq": 5         # Perform bagging every 5 iterations
}
"""

params = ember.optimize_model(X_train, y_train)


# Train the model
print("Training LightGBM model")

# Filter unlabeled data
train_rows = (y_train != -1)

# Train
lgbm_dataset = lgb.Dataset(X_train[train_rows], y_train[train_rows])
lgbm_model = lgb.train(params, lgbm_dataset)

lgbm_model.save_model(os.path.join(data_dir, "model_mine.txt"))


# Evaluate the model
print("Evaluating the model")
y_pred = lgbm_model.predict(X_test)
print(f"Test Data Distribution: {sum(y_test)} positive, {len(y_test) - sum(y_test)} negative")

y_pred_binary = (y_pred > 0.5).astype(int)
print(f"Accuracy: {accuracy_score(y_test, y_pred_binary)}")
print(f"AUC: {roc_auc_score(y_test, y_pred)}")
print("Classification Report:")
print(classification_report(y_test, y_pred_binary))