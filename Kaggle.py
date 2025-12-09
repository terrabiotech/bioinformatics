import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC

from pandas.plotting import scatter_matrix

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical
from keras.constraints import max_norm

warnings.filterwarnings("ignore")

# ========= 0. Config & sanity =========
seed = 7
np.random.seed(seed)

print("TensorFlow version:", tf.__version__)

# Folder where Kaggle.py and data live
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print("Script folder =", BASE_DIR)

# List folder contents to find the data file
contents = os.listdir(BASE_DIR)
print("Folder contents:", contents)

# Pick the first non-.py file starting with 'ecoli.'
data_file = None
for fname in contents:
    lower = fname.lower()
    if lower.startswith("ecoli.") and not lower.endswith(".py"):
        data_file = fname
        break

if data_file is None:
    raise FileNotFoundError(
        f"No data file starting with 'ecoli.' found in {BASE_DIR}. "
        f"Current contents: {contents}"
    )

csv_path = os.path.join(BASE_DIR, data_file)
print("Using data file:", data_file)
print("Full path      :", csv_path)
print("Path exists?   :", os.path.exists(csv_path))

if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Data file not found at: {csv_path}")

# ========= 1. Load dataset (UCI whitespace format) =========
# Your file is the original UCI ecoli.data (whitespace, no header), just renamed to .csv.csv. [web:73][web:152]
dataframe = pd.read_csv(csv_path, header=None, delim_whitespace=True)
dataframe.columns = ['seq_name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'site']

# Drop sequence name
dataframe = dataframe.drop("seq_name", axis=1)

print("Columns:", dataframe.columns)
print("Unique site labels:", dataframe["site"].unique())
print("\nHead:\n", dataframe.head())
print("\nShape:", dataframe.shape)

# Encode class labels if still strings
if dataframe["site"].dtype == "object":
    dataframe["site"] = dataframe["site"].replace(
        ("cp", "im", "pp", "imU", "om", "omL", "imL", "imS"),
        (1, 2, 3, 4, 5, 6, 7, 8),
    )

dataset = dataframe.values
X = dataset[:, 0:7]
Y = dataset[:, 7]

# ========= 2. Feature selection (RFE) =========
fs_model = LogisticRegression(max_iter=1000)
rfe = RFE(estimator=fs_model, n_features_to_select=3)
fit = rfe.fit(X, Y)

print("\nNumber of Features:", fit.n_features_)
print("Selected Features:", fit.support_)
print("Feature Ranking:", fit.ranking_)

# ========= 3. Basic plots =========
plt.figure()
plt.hist(dataframe["site"], bins=8)
plt.title("Class distribution (site)")
plt.show()

dataframe.plot(kind="density", subplots=True, layout=(3, 3),
               sharex=False, sharey=False, figsize=(10, 8))
plt.tight_layout()
plt.show()

dataframe.plot(kind="box", subplots=True, layout=(3, 3),
               sharex=False, sharey=False, figsize=(10, 8))
plt.tight_layout()
plt.show()

scatter_matrix(dataframe, figsize=(10, 10))
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(dataframe.corr(), vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0, dataframe.shape[1], 1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(dataframe.columns, rotation=90)
ax.set_yticklabels(dataframe.columns)
plt.tight_layout()
plt.show()

# ========= 4. Classical ML models with CV =========
models = [
    ("LR", LogisticRegression(max_iter=1000)),
    ("LDA", LinearDiscriminantAnalysis()),
    ("KNN", KNeighborsClassifier()),
    ("CART", DecisionTreeClassifier(random_state=seed)),
    ("NB", GaussianNB()),
    ("SVM", SVC()),
    ("L_SVM", LinearSVC()),
    ("ETC", ExtraTreesClassifier(random_state=seed)),
    ("RFC", RandomForestClassifier(random_state=seed)),
]

results = []
names = []
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)

for name, model in models:
    cv_results = cross_val_score(model, X, Y, cv=kfold, scoring="accuracy")
    results.append(cv_results)
    names.append(name)
    print(f"{name}: {cv_results.mean():.4f} ({cv_results.std():.4f})")

fig = plt.figure()
fig.suptitle("Algorithm Comparison")
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

# ========= 5. Neural net with Keras + StratifiedKFold =========
Y_int = Y.astype(int)
Y_cat = to_categorical(Y_int - 1)  # 1..8 → 0..7

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
cvscores = []

for train_idx, test_idx in skf.split(X, Y_int):
    model = Sequential()
    model.add(Dense(20, input_dim=7, kernel_initializer="uniform", activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(10, kernel_initializer="uniform", activation="relu",
                    kernel_constraint=max_norm(3)))
    model.add(Dropout(0.2))
    model.add(Dense(5, kernel_initializer="uniform", activation="relu"))
    model.add(Dense(8, kernel_initializer="uniform", activation="softmax"))

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    model.fit(X[train_idx], Y_cat[train_idx], epochs=50, batch_size=10, verbose=0)

    scores = model.evaluate(X[test_idx], Y_cat[test_idx], verbose=0)
    print(f"Fold accuracy: {scores[1] * 100:.2f}%")
    cvscores.append(scores[1] * 100)

print("Mean NN accuracy: %.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))





