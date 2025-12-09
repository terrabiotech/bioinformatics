# Resources used for building this project:
# https://www.kaggle.com/code/prashant111/random-forest-classifier-tutorial
# https://www.kaggle.com/code/elikplim/predicting-the-localization-sites-of-proteins
# https://www.perplexity.ai/search/protein-localization-sites-in-6rSqCJnyRfG60X3JXsrJHQ

##Before running the import, use computer terminal to installs packages:
###way 1: pip install -U ucimlrepo scikit-learn
###way 2: conda install scikit-learn
###       pip install -U ucimlrepo


from ucimlrepo import fetch_ucirepo                                 # Fetch datasets from UCI's ML repo.
from sklearn.ensemble import RandomForestClassifier                 # Machine learning model
from sklearn.metrics import classification_report, accuracy_score   # Model evaluation metrics
from sklearn.model_selection import train_test_split                # Split dataset into training and testing sets.
from sklearn.preprocessing import StandardScaler                    # Normalize all features to a comparative scale.

# 1. Prepare the dataset
ecoli = fetch_ucirepo(id = 39)  # Fetch the Ecoli dataset: https://archive.ics.uci.edu/dataset/39/ecoli
X     = ecoli.data.features     # Features (Protein attributes)
y     = ecoli.data.targets      # Targets (Protein localization sites)

# 2. Split dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Standardize the features
scaler         = StandardScaler()               # Create a scaler, which standardizes features.
X_train_scaled = scaler.fit_transform(X_train)  # Compute the mean and standard deviation of the training data.
X_test_scaled  = scaler.transform(X_test)       # Transform the test data, ensuring consistent scaling.

# 4. Initialize and train the RandomForest classifier
model = RandomForestClassifier(random_state = 42)   # Create many small decision trees to make reliable guesses.
model.fit(X_train_scaled, y_train.values.ravel())   # Train the model by showing it labeled examples.

# 5. Feed test data to the model, causing it to make predictions based on what it learned during training.
y_pred = model.predict(X_test_scaled)  # Predict where on the cell the protein is located.

# 6. Evaluate model accuracy.
accuracy = accuracy_score(y_test, y_pred)  # Calculate how often the model's predicted answers match the actual answers in the test data.
print(f"Accuracy: {accuracy:.4f}\n")         # Display the accuracy value as a decimal number.
print("Classification Report:\n" +
      classification_report(y_true        = y_test,
                            y_pred        = y_pred,
                            zero_division = 0))
print("This model would be much more accurate if there was a larger training dataset because it would have more examples to learn from.")
