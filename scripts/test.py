import ember
import os

# Load the EMBER feature set (already vectorized)
data_dir = "data"
ember.create_vectorized_features(data_dir) # creates X_train.dat and y_train.dat, X_test.dat and y_test.dat

# Load the vectorized features
print("Loading vectorized features")
X_train, y_train = ember.read_vectorized_features(data_dir, subset="train")

# Optimise params
#print("Optimising model parameters")
params = {
            "boosting": "gbdt",
            "objective": "binary",
            "num_iterations": 1000,
            "learning_rate": 0.05,
            "num_leaves": 2048,
            "max_depth": 15,
            "min_data_in_leaf": 50,
            "feature_fraction": 0.5
        }

# Train the model
print("Training LightGBM model")
lgbm_model = ember.train_model(data_dir, params)
lgbm_model.save_model(os.path.join(data_dir, "model_mine.txt"))

# Evaluate the model
print("Evaluating the model")
X_test, y_test = ember.read_vectorized_features(data_dir, subset="test")
predictions = lgbm_model.predict(X_test)

ember.predict_sample(data_dir, lgbm_model)

