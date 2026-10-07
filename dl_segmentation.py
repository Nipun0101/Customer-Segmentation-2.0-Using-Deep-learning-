# %% [markdown]
# # Customer Segmentation using Autoencoder-based Deep Clustering (DSE3120)
# **Part A:** Autoencoder (neural network) compresses each customer to 2 numbers -> K-Means groups them.
# **Part B:** Feedforward neural network predicts whether a customer accepts the last campaign.

# %% Imports and setup
import json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers, regularizers
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (silhouette_score, davies_bouldin_score, adjusted_rand_score, accuracy_score,
                             precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix)

SEED = 42
keras.utils.set_random_seed(SEED)
OUT = "dl_results"
os.makedirs(OUT, exist_ok=True)
results = {}

# %% Load and clean (same cleaning as the dashboard)
MNT = ["MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts", "MntSweetProducts", "MntGoldProds"]
FEATURES = ["Age", "Income", "Total_Spending", "NumWebPurchases", "NumStorePurchases", "NumWebVisitsMonth", "Recency"]

df = pd.read_csv("customer_segmentation.csv").dropna(subset=["Income"])
df["Age"] = 2025 - df["Year_Birth"]
df = df[(df["Age"] < 100) & (df["Income"] < 200000)].copy()
df["Total_Spending"] = df[MNT].sum(axis=1)
df["Total_Children"] = df["Kidhome"] + df["Teenhome"]
df = df.reset_index(drop=True)
print("Customers after cleaning:", len(df))
results["customers"] = len(df)

# %% PART A - scale the 7 features
sc_ae = StandardScaler().fit(df[FEATURES])
X = sc_ae.transform(df[FEATURES])

# %% Baseline: K-Means directly, and PCA(2) compression
K = 6
km_base = KMeans(K, n_init=10, random_state=SEED).fit(X)
lab_base = km_base.labels_
pca = PCA(2, random_state=SEED).fit(X)
Z_pca = pca.transform(X)
pca_mse = float(np.mean((X - pca.inverse_transform(Z_pca)) ** 2))
lab_pca = KMeans(K, n_init=10, random_state=SEED).fit_predict(Z_pca)

# %% Autoencoder: 7 -> 16 -> 8 -> 2 (bottleneck) -> 8 -> 16 -> 7
inp = keras.Input(shape=(7,))
h = layers.Dense(16, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(inp)
h = layers.Dense(8, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(h)
z = layers.Dense(2, name="bottleneck")(h)
d = layers.Dense(8, activation="relu")(z)
d = layers.Dense(16, activation="relu")(d)
out = layers.Dense(7)(d)
ae = keras.Model(inp, out)
encoder = keras.Model(inp, z)
ae.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
es = keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True, monitor="val_loss")
hist_ae = ae.fit(X, X, epochs=300, batch_size=32, validation_split=0.2, callbacks=[es], verbose=0)
ae_mse = float(np.mean((X - ae.predict(X, verbose=0)) ** 2))
Z_ae = encoder.predict(X, verbose=0)
km_ae = KMeans(K, n_init=10, random_state=SEED).fit(Z_ae)
lab_ae = km_ae.labels_
print(f"Epochs trained: {len(hist_ae.history['loss'])}")

# %% Compare all three clusterings in the SAME space (the 7 scaled features)
def metrics(lab):
    return {"silhouette": round(float(silhouette_score(X, lab)), 4),
            "davies_bouldin": round(float(davies_bouldin_score(X, lab)), 4)}

results["partA"] = {
    "epochs_trained": len(hist_ae.history["loss"]),
    "reconstruction_mse": {"pca_2d": round(pca_mse, 4), "autoencoder_2d": round(ae_mse, 4)},
    "kmeans_on_7_features": metrics(lab_base),
    "kmeans_on_pca_2d": metrics(lab_pca),
    "kmeans_on_autoencoder_2d": metrics(lab_ae),
    "agreement_ARI_autoencoder_vs_baseline": round(float(adjusted_rand_score(lab_base, lab_ae)), 4),
    "cluster_sizes_autoencoder": np.bincount(lab_ae).tolist(),
}
print(json.dumps(results["partA"], indent=2))

# %% Plots for Part A
fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
ax[0].plot(hist_ae.history["loss"], label="train"); ax[0].plot(hist_ae.history["val_loss"], label="validation")
ax[0].set_title("Autoencoder loss (MSE)"); ax[0].set_xlabel("epoch"); ax[0].legend()
ax[1].scatter(Z_pca[:, 0], Z_pca[:, 1], c=lab_pca, cmap="tab10", s=8); ax[1].set_title("PCA 2D + K-Means")
ax[2].scatter(Z_ae[:, 0], Z_ae[:, 1], c=lab_ae, cmap="tab10", s=8); ax[2].set_title("Autoencoder 2D + K-Means")
plt.tight_layout(); plt.savefig(f"{OUT}/partA_autoencoder.png", dpi=130); plt.close()

prof = df.assign(Segment=lab_ae).groupby("Segment")[FEATURES].mean().round(1)
prof["Customers"] = np.bincount(lab_ae)
tier = lambda v: "High" if v >= 1.5 * df.Total_Spending.mean() else "Mid" if v >= 0.75 * df.Total_Spending.mean() else "Low"
prof["Label"] = [f"{tier(r.Total_Spending)} spend · {'Active' if r.Recency < prof.Recency.median() else 'Lapsing'}" for _, r in prof.iterrows()]
prof.to_csv(f"{OUT}/autoencoder_segment_profiles.csv")
print(prof)

# %% PART B - feedforward network predicting campaign Response
FB = FEATURES + ["Total_Children", "NumCatalogPurchases", "NumDealsPurchases"]
y = df["Response"].values
Xtr, Xte, ytr, yte = train_test_split(df[FB], y, test_size=0.2, stratify=y, random_state=SEED)
sc = StandardScaler().fit(Xtr)
Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
pos_rate = float(y.mean())
cw = {0: 1.0, 1: float((1 - pos_rate) / pos_rate)}
print(f"Share of customers who accepted: {pos_rate:.1%}")

def build(hidden, lr, drop=0.3):
    m = keras.Sequential([keras.Input(shape=(len(FB),))])
    for u in hidden:
        m.add(layers.Dense(u, activation="relu", kernel_regularizer=regularizers.l2(1e-3)))
        m.add(layers.Dropout(drop))
    m.add(layers.Dense(1, activation="sigmoid"))
    m.compile(optimizer=keras.optimizers.Adam(lr), loss="binary_crossentropy")
    return m

# %% Hyperparameter tuning (small grid) - choose by validation AUC
grid, best = [], None
Xa, Xv, ya, yv = train_test_split(Xtr_s, ytr, test_size=0.2, stratify=ytr, random_state=SEED)
for hidden in [(16,), (32, 16), (64, 32)]:
    for lr in [1e-3, 1e-2]:
        keras.utils.set_random_seed(SEED)
        m = build(hidden, lr)
        m.fit(Xa, ya, epochs=150, batch_size=32, validation_data=(Xv, yv), class_weight=cw, verbose=0,
              callbacks=[keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)])
        auc = float(roc_auc_score(yv, m.predict(Xv, verbose=0).ravel()))
        grid.append({"hidden": list(hidden), "learning_rate": lr, "val_auc": round(auc, 4)})
        if best is None or auc > best[0]:
            best = (auc, hidden, lr)
print(pd.DataFrame(grid))
results["partB_tuning"] = grid
results["partB_best_config"] = {"hidden": list(best[1]), "learning_rate": best[2]}

# %% Train the best network and evaluate on the untouched test set
keras.utils.set_random_seed(SEED)
fnn = build(best[1], best[2])
hist = fnn.fit(Xtr_s, ytr, epochs=200, batch_size=32, validation_split=0.2, class_weight=cw, verbose=0,
               callbacks=[keras.callbacks.EarlyStopping(patience=12, restore_best_weights=True)])
p = fnn.predict(Xte_s, verbose=0).ravel()
pred = (p >= 0.5).astype(int)

def report(yt, yp, prob):
    return {"accuracy": round(float(accuracy_score(yt, yp)), 4), "precision": round(float(precision_score(yt, yp, zero_division=0)), 4),
            "recall": round(float(recall_score(yt, yp)), 4), "f1": round(float(f1_score(yt, yp)), 4),
            "roc_auc": round(float(roc_auc_score(yt, prob)), 4)}

lr_model = LogisticRegression(max_iter=1000, class_weight="balanced").fit(Xtr_s, ytr)
lr_prob = lr_model.predict_proba(Xte_s)[:, 1]
always_no = {"accuracy": round(float(1 - yte.mean()), 4), "precision": 0.0, "recall": 0.0, "f1": 0.0}
results["partB"] = {
    "positive_rate": round(pos_rate, 4),
    "test_size": int(len(yte)),
    "always_predict_no": always_no,
    "logistic_regression": report(yte, (lr_prob >= 0.5).astype(int), lr_prob),
    "neural_network": report(yte, pred, p),
    "confusion_matrix_nn": confusion_matrix(yte, pred).tolist(),
    "epochs_trained": len(hist.history["loss"]),
}
print(json.dumps(results["partB"], indent=2))

# %% Plots for Part B
fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
ax[0].plot(hist.history["loss"], label="train"); ax[0].plot(hist.history["val_loss"], label="validation")
ax[0].set_title("Neural network loss (training vs validation)"); ax[0].set_xlabel("epoch"); ax[0].legend()
cm = confusion_matrix(yte, pred)
ax[1].imshow(cm, cmap="Blues")
for (i, j), v in np.ndenumerate(cm):
    ax[1].text(j, i, v, ha="center", va="center", fontsize=14, color="white" if v > cm.max() / 2 else "black")
ax[1].set_xticks([0, 1], ["No", "Yes"]); ax[1].set_yticks([0, 1], ["No", "Yes"])
ax[1].set_xlabel("Predicted"); ax[1].set_ylabel("Actual"); ax[1].set_title("Confusion matrix (test set)")
plt.tight_layout(); plt.savefig(f"{OUT}/partB_neural_network.png", dpi=130); plt.close()

json.dump(results, open(f"{OUT}/results.json", "w"), indent=2)
print("Saved results to", OUT)

# %% Export everything the Streamlit app needs (no TensorFlow needed to run the app)
enc_w = encoder.get_weights()
fnn_w = fnn.get_weights()
arrs = {"ae_mean": sc_ae.mean_, "ae_scale": sc_ae.scale_, "ae_centers": km_ae.cluster_centers_,
        "fnn_mean": sc.mean_, "fnn_scale": sc.scale_, "fnn_n": np.array(len(fnn_w) // 2)}
for i in range(3):
    arrs[f"ae_w{i}"], arrs[f"ae_b{i}"] = enc_w[2 * i], enc_w[2 * i + 1]
for i in range(len(fnn_w) // 2):
    arrs[f"fnn_w{i}"], arrs[f"fnn_b{i}"] = fnn_w[2 * i], fnn_w[2 * i + 1]
np.savez(f"{OUT}/model_weights.npz", **arrs)
emb = df[FEATURES].copy()
emb["z1"], emb["z2"], emb["ae_cluster"] = Z_ae[:, 0], Z_ae[:, 1], lab_ae
emb["pca1"], emb["pca2"], emb["pca_cluster"] = Z_pca[:, 0], Z_pca[:, 1], lab_pca
emb.to_csv(f"{OUT}/embeddings.csv", index=False)

# sanity check: pure-numpy forward pass must match Keras
x = X[:5]
for i in range(3):
    x = x @ arrs[f"ae_w{i}"] + arrs[f"ae_b{i}"]
    x = np.maximum(x, 0) if i < 2 else x
assert np.allclose(x, encoder.predict(X[:5], verbose=0), atol=1e-4), "numpy encoder mismatch"
print("Exported model weights, embeddings; numpy forward pass matches Keras")
