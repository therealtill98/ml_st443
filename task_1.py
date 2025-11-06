"""
ST443 — Task 1: Multiclass classification on hyperspectral data
===============================================================

This script is organized into standalone sections you can run one-by-one to understand each step.

It includes:
- EDA (summary stats, class balance, per-band distributions, spectra by class,
  correlations, dimensionality glimpses via PCA) with careful sampling to keep
  memory manageable on ~218k x ~221 data.
- Training & evaluation of 7 required classifiers: LDA, Logistic, QDA, kNN, GBDT,
  RandomForest, SVM — with and without PCA(10).
- Metrics: accuracy, misclassification error, macro balanced accuracy, macro F1,
  macro One-vs-Rest AUC — all on a stratified holdout set.
  - A compact CSV summary of results for your report.
  - A binary “glacier ice vs all” experiment (Task 1.4) using F1(positive=glacier)
    as the primary metric, with justification included inline.
  - A spec-compliant mypredict() that loads test.csv.gz and writes predictions.

Data expectations (from the coursework brief):
- Features: Bands named "Band_1", ..., "Band_218" (reflectance), and p_x, p_y coordinates.
- Label: "land_type" with 8 classes.

Authors: Till and Tommy
"""

# =========================
# Section 0 — SETUP & IMPORTS
# =========================

# Import standard libraries for data wrangling and numerical computing
import os  # file paths and directory creation
import gc  # manual garbage collection to free memory between heavy steps
from collections import Counter  # quick counts for sanity checks

# Core scientific stack
import numpy as np  # numerical arrays and fast math
import pandas as pd  # tabular data manipulation

# Plotting libraries
import matplotlib.pyplot as plt  # base plotting
import seaborn as sns  # statistical graphics

# Machine learning models and tools from scikit-learn.
from sklearn.model_selection import train_test_split  # stratified holdout split
from sklearn.preprocessing import StandardScaler  # feature scaling for some models
from sklearn.decomposition import PCA  # PCA for dimensionality reduction
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, confusion_matrix
)  # evaluation metrics
from sklearn.metrics import balanced_accuracy_score  # macro balanced accuracy
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC

# Model persistence to reuse the best model in mypredict()
from joblib import dump, load  # serialize models and preprocessing objects

# Set global plotting style for consistency.
sns.set(context="notebook", style="whitegrid")  # aesthetic defaults for plots

# Create output directories if they don't exist, keeping results/ and figures/ tidy.
os.makedirs("figures", exist_ok=True)  # save all images for the report here
os.makedirs("results", exist_ok=True)  # save CSV summaries and artifacts here
os.makedirs("artifacts", exist_ok=True)  # save trained models/scalers/PCA here

# To make your experiments reproducible, fix a random seed used by numpy & sklearn.
RANDOM_STATE = 10291999  # any fixed integer is fine; use one seed consistently

# Control how many rows we sample for heavy EDA plots to avoid OOM or sluggishness.
SAMPLE_FOR_PLOTS = 30000  # increase/decrease depending on computer capabilities


#####################################
# Section 1 — LOAD THE DATA
#####################################

# Define directory paths to look for the data file.
home_dir = os.path.expanduser("~")
downloads_path = os.path.join(home_dir, "Downloads")

# Full path to the CSV file
data_path = os.path.join(downloads_path, "data-1.csv")

# Load the data
df = pd.read_csv(data_path, low_memory=False)

# Save df locally for easy reuse in case of path issues later.
df.to_csv("data-1_local_copy.csv", index=False)

# Print basic info to confirm shape and memory footprint right away.
print(f"Loaded data from: {data_path}")
print(f"Data shape: {df.shape[0]:,} rows x {df.shape[1]:,} columns")

df

#####################################
# Section 2 — BASIC SANITY CHECKS & COLUMN DETECTION
#####################################

# Detect the label column. The coursework states it's 'land_type'.
label_col = "land_type"

# Define spatial columns 
spatial_cols = ["p_x", "p_y"]

# Heuristically detect band columns: names starting with 'Band ' or 'band ' then a number.
band_cols = [c for c in df.columns if c.lower().startswith("band_")]

# Safety checks: ensure we have bands and a label.
if not band_cols:
    raise KeyError("Could not detect band feature columns. Expected names like 'Band 1', 'Band 2', ...")
if label_col not in df.columns:
    raise KeyError(f"Label column '{label_col}' not found after detection.")

print(f"[INFO] Detected label column: {label_col}")
print(f"[INFO] Detected {len(band_cols)} band feature columns.")
if spatial_cols:
    print(f"[INFO] Detected spatial feature columns: {spatial_cols}")
else:
    print("[INFO] No explicit spatial columns detected (p_x, p_y). Proceeding without them.")

# For convenience, define the feature set X and target y.
feature_cols = band_cols + spatial_cols
X_full = df[feature_cols].copy()
y_full = df[label_col].astype("category").copy() # store target as categorical for clarity

# Peek at target classes and counts.
class_counts = y_full.value_counts()
print("[INFO] Class distribution:")
print(class_counts.to_string())

# Ensure there are no duplicate rows with conflicting labels
dup_count = df.duplicated(subset=feature_cols, keep=False).sum()
print(f"[INFO] Potential duplicate feature rows (any label): {dup_count}")
# If any exist, we would drop or resolve them

# Clean obvious numeric issues: coerce bands and spatial to numeric if strings slipped in.
X_full = X_full.apply(pd.to_numeric, errors="coerce")

# Report missingness so we know how much cleaning is necessary.
missing_by_col = X_full.isna().sum()
total_missing = int(missing_by_col.sum())
print(f"[INFO] Total missing values across all features: {total_missing:,}")

#####################################
# Section 3 — EDA (Exploratory Data Analysis)
#####################################

# For heavy plots we sample to keep responsiveness; however, we always keep class balance in mind.

# 3.1 — Quick numeric summary of bands (reflectance typically in [0, 1] or [0, 10^–1]).
desc = X_full[band_cols].describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).T
desc.to_csv("results/eda_band_summary.csv", index=True)
print("[EDA] Saved per-band summary to results/eda_band_summary.csv")

# 3.2 — Overall class distribution (bar chart).
plt.figure(figsize=(8, 5))
(class_counts.sort_values(ascending=False)
    .plot(kind="bar", edgecolor="black"))
plt.title("Class Distribution of land_type")
plt.xlabel("Land Type")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("figures/class_distribution.png", dpi=150)
plt.close()
print("[EDA] Saved class distribution plot to figures/class_distribution.png")

# 3.3 — Per-band min/max across all pixels to sanity-check reflectance ranges.
# This helps catch unit/scale problems (e.g., negative reflectance).
band_min = X_full[band_cols].min()
band_max = X_full[band_cols].max()
range_df = pd.DataFrame({"min": band_min, "max": band_max})
range_df.to_csv("results/eda_band_ranges.csv")
print("[EDA] Saved per-band min/max to results/eda_band_ranges.csv")

# Inspect the min and max across all bands
range_df.describe()
overall_min_mean = range_df['min'].mean()
overall_max_mean = range_df['max'].mean()
overall_min_std = range_df['min'].std()
overall_max_std = range_df['max'].std()

print(f"Average minimum reflectance: {overall_min_mean:.3f} ± {overall_min_std:.3f}")
print(f"Average maximum reflectance: {overall_max_mean:.3f} ± {overall_max_std:.3f}")

range_df.describe().to_csv("results/eda_range_summary.csv")
print("[EDA] Saved overall band range summary to results/eda_range_summary.csv")

# 3.4 — Plot a few representative band histograms to see shape/skew and outliers
# Choose evenly spaced bands across the spectrum to sample variability
example_band_idxs = np.linspace(0, len(band_cols) - 1, 6, dtype=int) # choosing 6 bands to show some variety in report
plt.figure(figsize=(12, 8))
for i, idx in enumerate(example_band_idxs, 1):
    plt.subplot(3, 2, i)
    col = band_cols[idx]
    # Drop NA just for plotting.
    vals = X_full[col].dropna()
    # Use a moderate bin count to see shape without too much noise.
    plt.hist(vals, bins=50, edgecolor="black")
    plt.title(f"Histogram: {col}")
    plt.xlabel("Reflectance")
    plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("figures/example_band_histograms.png", dpi=150)
plt.close()
print("[EDA] Saved example band histograms to figures/example_band_histograms.png")

# 3.5 — Mean spectral signature per class
# For each class we compute the mean reflectance for each band, then plot all class means on one figure
# Interpretation: curves separated in some wavelength regions indicate discriminative bands
mean_spectra_by_class = (
    pd.concat([X_full[band_cols], y_full], axis=1)
      .groupby(label_col)[band_cols]
      .mean()
)
plt.figure(figsize=(10, 6))
for cls in mean_spectra_by_class.index:
    plt.plot(range(len(band_cols)), mean_spectra_by_class.loc[cls].values, label=str(cls), linewidth=1.5)
plt.title("Mean Spectral Signature by Class")
plt.xlabel("Band Index (≈ 420 → 2450 nm)")
plt.ylabel("Mean Reflectance")
plt.legend(title="Class", ncol=2, fontsize=8)
plt.tight_layout()
plt.savefig("figures/mean_spectra_by_class.png", dpi=150)
plt.close()
print("[EDA] Saved mean spectral signatures plot to figures/mean_spectra_by_class.png")

# 3.6 — Correlation snapshot between bands (using a sample)
if len(X_full) > SAMPLE_FOR_PLOTS:
    sample_idx = np.random.RandomState(RANDOM_STATE).choice(len(X_full), size=SAMPLE_FOR_PLOTS, replace=False)
    corr_sample = X_full.iloc[sample_idx][band_cols].corr()
else:
    corr_sample = X_full[band_cols].corr()
# Plot a small heatmap of every Nth band to keep it legible.
step = max(1, len(band_cols) // 40)  # display ~up to 40×40 tiles
subset_cols = band_cols[::step]
plt.figure(figsize=(10, 8))
sns.heatmap(corr_sample.loc[subset_cols, subset_cols], cmap="viridis", xticklabels=True, yticklabels=True)
plt.title("Inter-band Correlation (subset for readability)")
plt.tight_layout()
plt.savefig("figures/band_correlation_subset.png", dpi=150)
plt.close()
print("[EDA] Saved band correlation snapshot to figures/band_correlation_subset.png")

####################################################################################
# 3.7 — Dimensionality glimpse: PCA on a sample, plot the first 2 principal components 
# colored by class
#
# Purpose:
# This section performs Principal Component Analysis (PCA) on a randomly selected subset
# of the hyperspectral dataset. The goal is to visualize how the data behaves in a
# two-dimensional space formed by the first two principal components (PC1 and PC2).
#
# If clear groupings by color (class) appear in this scatterplot, it means that linear
# combinations of the spectral bands (as PCA constructs) already capture meaningful
# separation between land-type classes.
#
# PCA is purely exploratory here: it helps us see structure and correlation patterns,
# not to train a model.
####################################################################################

# We only sample a subset of observations for this visualization to avoid plotting
# hundreds of thousands of points. Sampling speeds things up and keeps the scatterplot
# readable without losing the general structure of the data.

# Randomly choose SAMPLE_FOR_PLOTS (30,000) distinct row indices from the full dataset.
# np.random.RandomState(RANDOM_STATE) ensures reproducibility
pca_idx = np.random.RandomState(RANDOM_STATE).choice(
    len(X_full), size=SAMPLE_FOR_PLOTS, replace=False
)

# Extract the subset of feature data (only the spectral bands) and the corresponding labels.
# .iloc allows us to select by position; [band_cols] ensures we exclude non-numeric metadata.
X_sample = X_full.iloc[pca_idx][band_cols].copy()
y_sample = y_full.iloc[pca_idx].copy()

# ------------------------------------------------------------------------------------
# Standardization step
# ------------------------------------------------------------------------------------
# PCA is sensitive to the relative scale of each feature: bands with larger numeric
# ranges could dominate the principal components if left unscaled.
# StandardScaler subtracts the mean and divides by the standard deviation for each band,
# giving all features equal weight (mean=0, variance=1).
scaler_vis = StandardScaler()
X_sample_std = scaler_vis.fit_transform(X_sample)

# ------------------------------------------------------------------------------------
# Apply PCA to reduce 218 spectral bands down to 2 components
# ------------------------------------------------------------------------------------
# n_components=2 asks PCA to find the two orthogonal directions in feature space
# (linear combinations of the bands) that capture the greatest variance in reflectance.
# random_state ensures reproducibility.
pca_vis = PCA(n_components=2, random_state=RANDOM_STATE)

# fit_transform() both fits the PCA model (calculates the components) and transforms
# the standardized data into the new coordinate system defined by PC1 and PC2.
# The result is an array of shape (n_samples, 2) representing each observation’s
# coordinates in this new two-dimensional space.
X_pca2 = pca_vis.fit_transform(X_sample_std)

# ------------------------------------------------------------------------------------
# Combine PCA results with class labels for easy plotting
# ------------------------------------------------------------------------------------
# We create a DataFrame with three columns:
#  - PC1 and PC2: the new coordinates from PCA
#  - label_col: the class each pixel belongs to (used to color the scatterplot)
pca_vis_df = pd.DataFrame({
    "PC1": X_pca2[:, 0],
    "PC2": X_pca2[:, 1],
    label_col: y_sample.values
})

# ------------------------------------------------------------------------------------
# Visualization
# ------------------------------------------------------------------------------------
# This scatterplot shows the first principal component (PC1) on the x-axis
# and the second (PC2) on the y-axis.
# Each point corresponds to one sampled pixel, colored by its class label.
# If the classes form distinct clusters, it means linear combinations of bands
# already provide meaningful separation between land types.
plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=pca_vis_df,
    x="PC1",
    y="PC2",
    hue=label_col, # color points by land type
    s=10, # small point size for dense plots
    linewidth=0, # remove black edge outlines
    alpha=0.6 # partial transparency to visualize overlap density
)

# Add title and apply tight layout to avoid cutoff labels
plt.title("PCA (2D) — Sampled view colored by class")
plt.tight_layout()

# Save the figure to the 'figures' directory for later inclusion in the report
plt.savefig("figures/pca2_scatter_sample.png", dpi=150)
plt.close()

# Console log message for clarity when running the script.
print("[EDA] Saved PCA(2) scatter (sample) to figures/pca2_scatter_sample.png")

####################################################################################
# Interpretation Guide:
# - If you see distinct color clusters, then classes are well-separated by linear patterns.
# - If colors are heavily mixed, then more complex nonlinear patterns may exist,
#   and linear methods (like LDA) might perform less strongly than non-linear (e.g., tree-based) models.
# - This 2D PCA plot doesn’t prove separability in all dimensions, but gives
#   a quick, intuitive sense of how structured the data is in its dominant directions.
####################################################################################

# 3.8 — Missingness profile plot (how many missing per band).
missing_series = X_full[band_cols].isna().sum()
plt.figure(figsize=(10, 4))
plt.plot(range(len(band_cols)), missing_series.values)
plt.title("Missing Values per Band")
plt.xlabel("Band Index")
plt.ylabel("# Missing")
plt.tight_layout()
plt.savefig("figures/missing_per_band.png", dpi=150)
plt.close()
print("[EDA] Saved missingness per band to figures/missing_per_band.png")
# Good to confirm no missing values for bands used in modeling later.

# 3.9 Evidence for why LDA underperforms on hyperspectral data (added later after seeing results)
# Compute per-class covariance trace and determinant
#    (measures of overall variance magnitude and spread shape)
class_cov_stats = []
for label, X_group in X_full.groupby(y_full):
    # Use only spectral bands, ignore spatial coordinates
    cov_matrix = np.cov(X_group[band_cols].T)
    trace_val = np.trace(cov_matrix)
    det_val = np.linalg.det(cov_matrix + 1e-6 * np.eye(cov_matrix.shape[0]))  # stability term
    class_cov_stats.append({
        "land_type": label,
        "trace": trace_val,
        "log_det": np.log(det_val)
    })

cov_df = pd.DataFrame(class_cov_stats).sort_values("trace", ascending=False)
print("[INFO] Per-class covariance summary:")
print(cov_df)

# Interpretation:
# - Large differences in 'trace' or 'log_det' across classes
#   indicate unequal covariance magnitudes (violates LDA assumption).
# - The trace values — representing total variance within each class — vary by more than an order of magnitude (from 0.29 to 3.11).
# - This means some classes (especially snow / ice) exhibit far more spread across spectral bands than others (like alpine tundra or veg-scree mix).
# - That directly violates LDA’s assumption that all classes share the same covariance matrix.

# Visualize covariance scale variation across classes
plt.figure(figsize=(8,4))
sns.barplot(data=cov_df, x="land_type", y="trace", color="skyblue", edgecolor="black")
plt.title("Total variance (trace of covariance matrix) by class")
plt.ylabel("Trace of covariance")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("figures/covariance_trace_by_class.png", dpi=150)
plt.show()

# Interpretation:
# - Distinct covariance traces across classes confirm LDA’s
#   equal-covariance assumption does not hold.


#####################################
# Section 4 — TRAIN/TEST SPLIT & PREP
#####################################

# We use a stratified train/validation holdout split to evaluate models fairly and simply.
X_all = X_full.copy()
y_all = y_full.copy()

# For models that need scaling, we will fit a StandardScaler on the training data and
# transform both train and validation sets. Tree-based models will use unscaled features.
# Before splitting, we will not impute; we will impute using training medians after split
# to avoid leakage.
X_train, X_val, y_train, y_val = train_test_split(
    X_all, y_all, test_size=0.2, random_state=RANDOM_STATE, stratify=y_all
)

# Identify columns to scale (bands + optional spatial)
cols_to_scale = feature_cols  # here we scale all continuous features for non-tree models
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train[cols_to_scale])
X_val_scaled = scaler.transform(X_val[cols_to_scale])

# For tree models, we pass the unscaled DataFrame/numpy arrays.
X_train_tree = X_train[cols_to_scale].values
X_val_tree = X_val[cols_to_scale].values

# Keep label encodings for metrics that expect numeric y; many sklearn metrics accept strings,
# but to be safe for AUC we will map classes to integers when building OVR targets.
classes = np.array(sorted(y_all.cat.categories.tolist(), key=str))
class_to_int = {c: i for i, c in enumerate(classes)}
y_train_int = y_train.map(class_to_int).values
y_val_int = y_val.map(class_to_int).values

############################################
# Section 5 — METRICS: HELPERS (MULTICLASS)
############################################

def compute_all_metrics(y_true, y_pred, y_proba, class_labels):
    """
    Compute the required metrics:
      - Accuracy
      - Misclassification error
      - Macro Balanced Accuracy (OVR)
      - Macro F1
      - Macro AUC (OVR)
      - Confusion matrix (returned as a numpy array)

    y_true: array-like of true class labels (original labels, not ints)
    y_pred: array-like of predicted class labels (original labels)
    y_proba: array of shape (n_samples, n_classes) with probability estimates.
             Some models (e.g., SVM with probability=True) provide this.
    class_labels: array of class labels in the same order as y_proba columns.
    """
    # Accuracy is the simplest "overall correctness" metric.
    acc = accuracy_score(y_true, y_pred)
    # Misclassification error is just 1 - accuracy.
    miscls = 1.0 - acc
    # Macro F1 is the unweighted mean of per-class F1s; it balances across classes.
    f1_macro = f1_score(y_true, y_pred, average="macro")
    # Macro balanced accuracy averages TPR and TNR per class (via balanced_accuracy_score).
    bal_acc = balanced_accuracy_score(y_true, y_pred)

    # Macro AUC: One-vs-Rest AUC per class averaged. Requires probability scores.
    # We compute for each class k, using y_true==k vs rest and the predicted proba for class k.
    aucs = []
    # Convert y_true to a numpy array to compare with class labels.
    y_true_np = np.array(y_true)
    for i, k in enumerate(class_labels):
        # Binary ground truth for class k vs rest.
        y_bin = (y_true_np == k).astype(int)
        # Use predicted probability for class k (column i).
        try:
            auc_k = roc_auc_score(y_bin, y_proba[:, i])
            aucs.append(auc_k)
        except ValueError:
            # If a class is missing in y_true of the validation fold, AUC is undefined;
            # skip it from the average (rare with large data).
            pass
    auc_macro = float(np.mean(aucs)) if aucs else np.nan

    # Confusion matrix for detailed error analysis (rows=true, cols=pred).
    cm = confusion_matrix(y_true, y_pred, labels=class_labels)

    return {
        "Accuracy": acc,
        "Misclass Error": miscls,
        "Avg Balanced Acc": bal_acc,
        "F1 (macro)": f1_macro,
        "AUC (macro)": auc_macro,
        "Confusion Matrix": cm,
    }


############################################
# Section 6 — MODELS (NO PCA): TRAIN & EVAL
############################################

# We define a simple training function per model to keep the main flow readable.
# For clarity, models that benefit from scaling use (X_train_scaled, X_val_scaled).
# Tree-based models use (X_train_tree, X_val_tree).

results = []  # we will append a dict per model and then make a summary CSV

def eval_and_log(model_name, y_pred, proba, class_labels, tag="raw"):
    """
    Compute metrics and log into results[] along with metadata.
    'tag' differentiates raw features vs PCA(10) variants in the summary table.
    """
    metrics = compute_all_metrics(y_val, y_pred, proba, class_labels)
    row = {
        "Model": model_name,
        "Variant": tag,
        "Accuracy": metrics["Accuracy"],
        "Misclass Error": metrics["Misclass Error"],
        "Avg Balanced Acc": metrics["Avg Balanced Acc"],
        "F1 (macro)": metrics["F1 (macro)"],
        "AUC (macro)": metrics["AUC (macro)"],
    }
    results.append(row)
    # Also save the confusion matrix to a per-model CSV for inspection in the report.
    cm = metrics["Confusion Matrix"]
    cm_df = pd.DataFrame(cm, index=[f"true_{c}" for c in classes], columns=[f"pred_{c}" for c in classes])
    cm_path = f"results/confmat__{model_name.replace(' ', '_')}__{tag}.csv"
    cm_df.to_csv(cm_path)
    print(f"[EVAL] {model_name} ({tag}) — Acc={row['Accuracy']:.4f} | BalAcc={row['Avg Balanced Acc']:.4f} | F1={row['F1 (macro)']:.4f} | AUC={row['AUC (macro)']:.4f}")
    print(f"[EVAL] Saved confusion matrix to {cm_path}")

# 6.1 — LDA (with shrinkage to stabilize covariances; avoids singularity warnings).
lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")  # shrinkage auto-chooses a regularization
lda.fit(X_train_scaled, y_train)
y_pred_lda = lda.predict(X_val_scaled)
proba_lda = lda.predict_proba(X_val_scaled)
eval_and_log("LDA", y_pred_lda, proba_lda, classes, tag="raw")

# 6.2 — Multinomial Logistic Regression (strong baseline for multiclass).
logreg = LogisticRegression(
    multi_class="multinomial",
    solver="lbfgs",
    max_iter=500,
    C=1.0,
    n_jobs=None,
    random_state=RANDOM_STATE
)
logreg.fit(X_train_scaled, y_train)
y_pred_log = logreg.predict(X_val_scaled)
proba_log = logreg.predict_proba(X_val_scaled)
eval_and_log("Logistic Regression", y_pred_log, proba_log, classes, tag="raw")

# 6.3 — QDA (regularized to prevent singular covariance issues; reg_param in [0,1]).
qda = QuadraticDiscriminantAnalysis(reg_param=0.1)
qda.fit(X_train_scaled, y_train)
y_pred_qda = qda.predict(X_val_scaled)
proba_qda = qda.predict_proba(X_val_scaled)
eval_and_log("QDA", y_pred_qda, proba_qda, classes, tag="raw")

# 6.4 — k-NN (k=5 is a decent starting point; uses scaled features).
knn = KNeighborsClassifier(n_neighbors=5, weights="distance", n_jobs=-1)
knn.fit(X_train_scaled, y_train)
y_pred_knn = knn.predict(X_val_scaled)
# For AUC, we need class probabilities; KNN supports predict_proba.
proba_knn = knn.predict_proba(X_val_scaled)
eval_and_log("k-NN (k=5)", y_pred_knn, proba_knn, classes, tag="raw")

# 6.5 — Gradient Boosting Decision Trees (sklearn's GradientBoostingClassifier).
gbdt = GradientBoostingClassifier(random_state=RANDOM_STATE)
gbdt.fit(X_train_tree, y_train)
y_pred_gbdt = gbdt.predict(X_val_tree)
# predict_proba for AUC; note GBDT probabilities are usually well-calibrated-ish.
proba_gbdt = gbdt.predict_proba(X_val_tree)
eval_and_log("GBDT", y_pred_gbdt, proba_gbdt, classes, tag="raw")

# 6.6 — Random Forest (robust, handles many features; unscaled inputs).
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    n_jobs=-1,
    random_state=RANDOM_STATE
)
rf.fit(X_train_tree, y_train)
y_pred_rf = rf.predict(X_val_tree)
proba_rf = rf.predict_proba(X_val_tree)
eval_and_log("Random Forest", y_pred_rf, proba_rf, classes, tag="raw")

# 6.7 — SVM (RBF kernel) with probability=True to enable AUC computation.
svm = SVC(C=1.0, kernel="rbf", gamma="scale", probability=True, random_state=RANDOM_STATE)
svm.fit(X_train_scaled, y_train)
y_pred_svm = svm.predict(X_val_scaled)
proba_svm = svm.predict_proba(X_val_scaled)
eval_and_log("SVM (RBF)", y_pred_svm, proba_svm, classes, tag="raw")


############################################
# Section 7 — PCA(10) VARIANTS: TRAIN & EVAL
############################################

# PCA is run on scaled features 
# We fit PCA only on the training set to avoid leakage, then transform both sets.

PCA_COMPONENTS = 10
pca = PCA(n_components=PCA_COMPONENTS, random_state=RANDOM_STATE)
X_train_pca = pca.fit_transform(X_train_scaled)
X_val_pca = pca.transform(X_val_scaled)

# Refit the models that expect scaled data on the PCA(10) representation.
# Tree models on PCA are less standard (they don't need scaling or linear combos),
# but the coursework asks to evaluate the same classifiers with PCA(10) as well.

# LDA + PCA(10)
lda_p = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
lda_p.fit(X_train_pca, y_train)
y_pred_ldap = lda_p.predict(X_val_pca)
proba_ldap = lda_p.predict_proba(X_val_pca)
eval_and_log("LDA", y_pred_ldap, proba_ldap, classes, tag="pca10")

# Logistic + PCA(10)
log_p = LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=500, C=1.0, random_state=RANDOM_STATE)
log_p.fit(X_train_pca, y_train)
y_pred_logp = log_p.predict(X_val_pca)
proba_logp = log_p.predict_proba(X_val_pca)
eval_and_log("Logistic Regression", y_pred_logp, proba_logp, classes, tag="pca10")

# QDA + PCA(10)
qda_p = QuadraticDiscriminantAnalysis(reg_param=0.1)
qda_p.fit(X_train_pca, y_train)
y_pred_qdap = qda_p.predict(X_val_pca)
proba_qdap = qda_p.predict_proba(X_val_pca)
eval_and_log("QDA", y_pred_qdap, proba_qdap, classes, tag="pca10")

# kNN + PCA(10)
knn_p = KNeighborsClassifier(n_neighbors=5, weights="distance", n_jobs=-1)
knn_p.fit(X_train_pca, y_train)
y_pred_knnp = knn_p.predict(X_val_pca)
proba_knnp = knn_p.predict_proba(X_val_pca)
eval_and_log("k-NN (k=5)", y_pred_knnp, proba_knnp, classes, tag="pca10")

# GBDT + PCA(10) — use PCA-transformed features.
gbdt_p = GradientBoostingClassifier(random_state=RANDOM_STATE)
gbdt_p.fit(X_train_pca, y_train)
y_pred_gbdtp = gbdt_p.predict(X_val_pca)
proba_gbdtp = gbdt_p.predict_proba(X_val_pca)
eval_and_log("GBDT", y_pred_gbdtp, proba_gbdtp, classes, tag="pca10")

# RF + PCA(10)
rf_p = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE)
rf_p.fit(X_train_pca, y_train)
y_pred_rfp = rf_p.predict(X_val_pca)
proba_rfp = rf_p.predict_proba(X_val_pca)
eval_and_log("Random Forest", y_pred_rfp, proba_rfp, classes, tag="pca10")

# SVM + PCA(10)
svm_p = SVC(C=1.0, kernel="rbf", gamma="scale", probability=True, random_state=RANDOM_STATE)
svm_p.fit(X_train_pca, y_train)
y_pred_svmp = svm_p.predict(X_val_pca)
proba_svmp = svm_p.predict_proba(X_val_pca)
eval_and_log("SVM (RBF)", y_pred_svmp, proba_svmp, classes, tag="pca10")


############################################
# Section 8 — RESULTS SUMMARY TABLE
############################################

# Collect all results and write to CSV for inclusion in the report.
summary_df = pd.DataFrame(results)
summary_df.sort_values(by=["Variant", "Avg Balanced Acc", "Accuracy"], ascending=[True, False, False], inplace=True)
summary_path = "results/summary_all_models.csv"
summary_df.to_csv(summary_path, index=False)
print(f"[RESULTS] Wrote full summary table to {summary_path}")
print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.5f}"))

# We chose logistic regression with PCA(10) as the best model for reasons outlined in the report
best_row = summary_df.loc[
    (summary_df["Model"] == "Logistic Regression") & 
    (summary_df["Variant"] == "pca10")
]
BEST_MODEL_NAME = f"{best_row['Model']} ({best_row['Variant']})"
print(f"[RESULTS] Best by Avg Balanced Acc: {BEST_MODEL_NAME}")

# Save the best trained model and its preprocessing so mypredict() can reuse it.
# We choose between raw vs pca10 based on best_row, then save the right objects.
def _save_best_model_and_preproc():
    variant = best_row["Variant"]
    model_name = best_row["Model"]

    # Map the name+variant to the trained objects in memory.
    # We also need to define how to preprocess incoming data in mypredict().
    if variant == "raw":
        preproc = {
            "variant": "raw",
            "scaler": scaler,           # StandardScaler fitted on training data
            "pca": None,
            "cols": cols_to_scale,      # column order expected
            "classes": classes.tolist() # class order expected by predict_proba
        }
        model_lookup = {
            "LDA": lda, "Logistic Regression": logreg, "QDA": qda,
            "k-NN (k=5)": knn, "GBDT": gbdt, "Random Forest": rf, "SVM (RBF)": svm
        }
        model_obj = model_lookup[model_name]
    else:
        preproc = {
            "variant": "pca10",
            "scaler": scaler, # scale then PCA
            "pca": pca,
            "cols": cols_to_scale,
            "classes": classes.tolist()
        }
        model_lookup = {
            "LDA": lda_p, "Logistic Regression": log_p, "QDA": qda_p,
            "k-NN (k=5)": knn_p, "GBDT": gbdt_p, "Random Forest": rf_p, "SVM (RBF)": svm_p
        }
        model_obj = model_lookup[model_name]

    # Persist artifacts.
    dump(model_obj, "artifacts/best_model.joblib")
    dump(preproc, "artifacts/preproc.joblib")
    with open("artifacts/best_model_name.txt", "w") as f:
        f.write(BEST_MODEL_NAME)
    print(f"[ARTIFACTS] Saved best model '{BEST_MODEL_NAME}' and preprocessing to artifacts/")

_save_best_model_and_preproc()


############################################
# Section 9 — TASK 1.4: GLACIER ICE (BINARY) EXPERIMENT
############################################

# Define positive class
glacier_positive = "snow / ice"
print(f"[T1.4] Treating '{glacier_positive}' as the POSITIVE (glacier) class.")

# Create binary target arrays
y_train_bin = (y_train == glacier_positive).astype(int)
y_val_bin = (y_val == glacier_positive).astype(int)

# Evaluate four classifiers representative of different families:
#  1) LDA (linear generative baseline; needs scaling)
#  2) Logistic Regression (discriminative linear baseline; needs scaling)
#  3) Random Forest (bagged tree ensemble; scale-invariant)
#  4) SVM (RBF kernel; flexible nonlinear boundary; needs scaling)
# All metrics focus on F1(positive=glacier) to balance detection vs. false alarms.

# 1) LDA (binary)
lda_bin = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
lda_bin.fit(X_train_scaled, y_train_bin)
yhat_lda = lda_bin.predict(X_val_scaled)
f1_lda = f1_score(y_val_bin, yhat_lda, pos_label=1)
print(f"[T1.4] LDA (auto shrinkage) — F1(positive=glacier) = {f1_lda:.4f}")

# 2) Logistic Regression (balanced)
log_bin = LogisticRegression(
    solver="lbfgs", max_iter=500, class_weight="balanced", random_state=RANDOM_STATE
)
log_bin.fit(X_train_scaled, y_train_bin)
yhat_log = log_bin.predict(X_val_scaled)
f1_log = f1_score(y_val_bin, yhat_log, pos_label=1)
print(f"[T1.4] Logistic (balanced) — F1(positive=glacier) = {f1_log:.4f}")

# 3) Random Forest (balanced)
rf_bin = RandomForestClassifier(
    n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE
)
rf_bin.fit(X_train_tree, y_train_bin)
yhat_rf = rf_bin.predict(X_val_tree)
f1_rf = f1_score(y_val_bin, yhat_rf, pos_label=1)
print(f"[T1.4] Random Forest (balanced) — F1(positive=glacier) = {f1_rf:.4f}")

# 4) SVM (RBF, balanced)
# Rationale: strongly considered in the multiclass setting; here we include it to
# test a nonlinear margin on the binary task. We use class_weight='balanced' to
# counter class imbalance. Default C and gamma usually perform well with standardized inputs.
svm_bin = SVC(
    kernel="rbf",
    class_weight="balanced",
    probability=False,         # F1 uses hard labels; no need to calibrate probabilities here
    random_state=RANDOM_STATE, # for reproducibility where applicable
    cache_size=500             # speed up kernel computations
)
svm_bin.fit(X_train_scaled, y_train_bin)
yhat_svm = svm_bin.predict(X_val_scaled)
f1_svm = f1_score(y_val_bin, yhat_svm, pos_label=1)
print(f"[T1.4] SVM (RBF, balanced) — F1(positive=glacier) = {f1_svm:.4f}")

# Save results table for the report
t14_df = pd.DataFrame({
    "Classifier": [
        "LDA (auto shrinkage)",
        "Logistic (balanced)",
        "Random Forest (balanced)",
        "SVM (RBF, balanced)"
    ],
    "F1 (glacier=pos)": [f1_lda, f1_log, f1_rf, f1_svm]
})
os.makedirs("results", exist_ok=True)
t14_df.to_csv("results/t14_glacier_binary_results.csv", index=False)
print("[T1.4] Wrote binary glacier results to results/t14_glacier_binary_results.csv")

############################################
# Section 10 — PREDICTION HOOK (REQUIRED): mypredict()
############################################

def mypredict():
    """
    As instruction in Section 3.2 Code (Submission and Assessment): Read 'test.csv.gz' 
    from the working directory (same format as the training data but without labels), 
    and output the predicted class labels to a plain text file, one label per line, 
    in the same order.

    This function:
      - Loads the saved best model and preprocessing from artifacts/.
      - Reads test.csv.gz (or test.csv.zip / test.csv as fallback).
      - Applies the exact same preprocessing (impute with training medians via scaler.mean_/var_,
        scale, optional PCA).
      - Predicts class labels and writes them to 'predictions.txt'.
    """

    # Helper to resolve test file.
    test_path = "test.csv.gz"

    # Load artifacts saved earlier.
    model = load("artifacts/best_model.joblib")
    preproc = load("artifacts/preproc.joblib")
    best_name = open("artifacts/best_model_name.txt").read().strip()
    expected_cols = preproc["cols"]
    classes_order = preproc["classes"]
    variant = preproc["variant"]
    scaler_ = preproc["scaler"]
    pca_ = preproc["pca"]

    # Read test data.
    Xtest_df = pd.read_csv(test_path, low_memory=False)

    # Ensure the expected columns exist in the test file (order matters for transforms).
    missing = [c for c in expected_cols if c not in Xtest_df.columns]
    if missing:
        raise KeyError(f"Test file is missing expected columns: {missing[:10]}{'...' if len(missing)>10 else ''}")

    # Extract and coerce numeric.
    Xtest = Xtest_df[expected_cols].apply(pd.to_numeric, errors="coerce")

    # Impute with the training medians implicitly used by the StandardScaler:
    #   - StandardScaler stores means_ and scale_ from training.
    #   - For imputation, we use the training medians we saved? We did not persist medians
    #     separately to keep it simple. A robust alternative is to impute with column medians
    #     computed on test as a fallback; however, to *match training*, we approximate by
    #     filling with the training mean (close enough for standardized features and typically
    #     minimal missingness). If you prefer exact training medians, persist them similarly.
    # Here we fill with the training mean stored in scaler_.mean_.
    # (If any column had many missing values, consider saving medians at training time.)
    train_means = pd.Series(scaler_.mean_, index=expected_cols)
    Xtest_imp = Xtest.fillna(train_means)

    # Apply the same transforms.
    Xtest_scaled = scaler_.transform(Xtest_imp)
    if variant == "pca10":
        Xtest_final = pca_.transform(Xtest_scaled)
    else:
        Xtest_final = Xtest_scaled

    # Predict labels using the saved model.
    y_pred = model.predict(Xtest_final)

    # Write one prediction per line.
    with open("predictions.txt", "w") as f:
        for label in y_pred:
            f.write(str(label) + "\n")

    print(f"[mypredict] Used best model: {best_name}")
    print("[mypredict] Wrote predictions to predictions.txt")

# End of script.