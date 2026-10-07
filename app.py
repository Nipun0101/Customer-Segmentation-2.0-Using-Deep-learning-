import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Deep Customer Segmentation", page_icon="🧠", layout="wide")
D = "dl_results/"
R = json.load(open(D + "results.json"))
W = np.load(D + "model_weights.npz")
E = pd.read_csv(D + "embeddings.csv")
P = pd.read_csv(D + "autoencoder_segment_profiles.csv")
FEATS = ["Age", "Income", "Total_Spending", "NumWebPurchases", "NumStorePurchases", "NumWebVisitsMonth", "Recency"]
A, B = R["partA"], R["partB"]
names = {int(r.Segment): f"{int(r.Segment)}: {r.Label}" for r in P.itertuples()}
E["Segment"] = E["ae_cluster"].map(names)
E["PCA group"] = E["pca_cluster"].astype(str)
order = [names[k] for k in sorted(names)]


def encode(x):  # the trained encoder, run with plain numpy
    x = (np.asarray(x, float) - W["ae_mean"]) / W["ae_scale"]
    for i in range(3):
        x = x @ W[f"ae_w{i}"] + W[f"ae_b{i}"]
        if i < 2:
            x = np.maximum(x, 0)
    return x


def respond_prob(x):  # the trained feedforward network, run with plain numpy
    x = (np.asarray(x, float) - W["fnn_mean"]) / W["fnn_scale"]
    n = int(W["fnn_n"])
    for i in range(n):
        x = x @ W[f"fnn_w{i}"] + W[f"fnn_b{i}"]
        x = np.maximum(x, 0) if i < n - 1 else 1 / (1 + np.exp(-x))
    return float(x[0])


st.title("🧠 Customer Segmentation with Deep Learning")
st.caption("DSE3120 Deep Learning · Autoencoder-based clustering + a neural network for campaign response")
t1, t2, t3, t4, t5 = st.tabs(["Overview", "Part A · Autoencoder", "Part B · Neural network", "Try it", "Syllabus map"])

with t1:
    st.markdown(f"""
**Problem:** a store has {R['customers']:,} customers. Which types exist, and who will respond to a marketing campaign?

**Pipeline:** raw CSV → clean + standardise 7 features → **autoencoder** (7 → 16 → 8 → **2** → 8 → 16 → 7) →
K-Means (6 groups) on the 2 compressed numbers → compare with PCA. Separately, a **feedforward network** predicts campaign response.

**Training details:** ReLU activations · Adam optimizer · MSE loss (autoencoder) · binary cross-entropy (network) ·
L2 penalty, dropout and early stopping against overfitting · 6 hyperparameter settings tried.

**Honest summary:** the autoencoder compresses better than PCA, but the clustering gain is mixed; the network ranks responders slightly better than logistic regression but not on F1.
""")

with t2:
    st.subheader("Does the autoencoder beat the baselines?")
    tbl = pd.DataFrame({
        "Method": ["K-Means on 7 features", "K-Means on PCA (2D)", "K-Means on autoencoder (2D)"],
        "Silhouette ↑": [A["kmeans_on_7_features"]["silhouette"], A["kmeans_on_pca_2d"]["silhouette"], A["kmeans_on_autoencoder_2d"]["silhouette"]],
        "Davies-Bouldin ↓": [A["kmeans_on_7_features"]["davies_bouldin"], A["kmeans_on_pca_2d"]["davies_bouldin"], A["kmeans_on_autoencoder_2d"]["davies_bouldin"]],
        "Reconstruction MSE ↓": [None, A["reconstruction_mse"]["pca_2d"], A["reconstruction_mse"]["autoencoder_2d"]]})
    st.dataframe(tbl, hide_index=True, width="stretch")
    st.info("**Silhouette** (higher is better) and **Davies-Bouldin** (lower is better) both measure how well separated the groups are, "
            "all measured in the same 7-feature space. **Reconstruction MSE** shows how well each method compresses 7 numbers into 2.")
    c1, c2 = st.columns(2)
    c1.plotly_chart(px.scatter(E, x="z1", y="z2", color="Segment", category_orders={"Segment": order}, opacity=0.7,
                               title="Autoencoder 2D space (each dot = a customer)"), width="stretch")
    c2.plotly_chart(px.scatter(E, x="pca1", y="pca2", color="PCA group", opacity=0.7, title="PCA 2D space"), width="stretch")
    st.image(D + "partA_autoencoder.png", caption="Left: training vs validation loss. Training stopped early when validation loss stopped improving.")
    st.subheader("Segments found")
    st.dataframe(P.drop(columns=[]).round(1), hide_index=True, width="stretch")

with t3:
    st.subheader("Predicting who accepts the last campaign")
    N, L = B["neural_network"], B["logistic_regression"]
    rows = [["Always say 'no'", B["always_predict_no"]["accuracy"], 0, 0, 0, None],
            ["Logistic regression", L["accuracy"], L["precision"], L["recall"], L["f1"], L["roc_auc"]],
            ["Neural network", N["accuracy"], N["precision"], N["recall"], N["f1"], N["roc_auc"]]]
    st.dataframe(pd.DataFrame(rows, columns=["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]), hide_index=True, width="stretch")
    st.warning(f"Only {B['positive_rate']:.0%} of customers accept a campaign, so a model that always says 'no' gets "
               f"{B['always_predict_no']['accuracy']:.1%} accuracy and finds nobody. That is why precision, recall and F1 matter.")
    st.markdown("**Hyperparameter tuning (validation AUC):**")
    g = pd.DataFrame(R["partB_tuning"]); g["hidden"] = g["hidden"].astype(str)
    st.dataframe(g, hide_index=True, width="stretch")
    st.image(D + "partB_neural_network.png", caption="Left: loss curves (training vs validation). Right: confusion matrix on the test set.")

with t4:
    st.subheader("Try the trained networks on a new customer")
    c = st.columns(5)
    age = c[0].number_input("Age", 18, 100, 45); inc = c[1].number_input("Income", 0, 200000, 60000, step=1000)
    spend = c[2].number_input("Total spending", 0, 3000, 800); rec = c[3].number_input("Recency (days)", 0, 99, 30)
    kids = c[4].number_input("Children at home", 0, 5, 1)
    c = st.columns(5)
    wp = c[0].number_input("Web purchases", 0, 30, 4); sp = c[1].number_input("Store purchases", 0, 30, 6)
    wv = c[2].number_input("Web visits / month", 0, 20, 5); cp = c[3].number_input("Catalog purchases", 0, 30, 2)
    dp = c[4].number_input("Deal purchases", 0, 30, 2)
    if st.button("Run both networks", type="primary"):
        seg7 = [age, inc, spend, wp, sp, wv, rec]
        z = encode(seg7)
        k = int(np.argmin(((W["ae_centers"] - z) ** 2).sum(axis=1)))
        pr = respond_prob(seg7 + [kids, cp, dp])
        a, b = st.columns(2)
        a.success(f"Segment (autoencoder + K-Means): **{names[k]}**")
        b.info(f"Probability of accepting a campaign: **{pr:.0%}**")
        fig = px.scatter(E, x="z1", y="z2", color="Segment", category_orders={"Segment": order}, opacity=0.3)
        fig.add_trace(go.Scatter(x=[z[0]], y=[z[1]], mode="markers", name="This customer",
                                 marker=dict(size=18, color="black", symbol="star")))
        st.plotly_chart(fig, width="stretch")
        st.caption("The 7 numbers were squeezed to 2 by the trained encoder; the star shows where this customer lands.")

with t5:
    st.markdown("""
| Syllabus topic (lecture) | Where it appears |
|---|---|
| PCA (5) | Baseline compression compared with the autoencoder |
| Perceptron and neurons (8) | Every Dense layer is a layer of artificial neurons |
| Activation functions (9) | ReLU in hidden layers, sigmoid on the response network's output |
| Feedforward networks (10) | Part B |
| Backpropagation and gradient descent (11, 12) | How both networks were trained |
| Adam optimizer (13) | Used for both |
| Loss functions and metrics (14) | MSE, binary cross-entropy, precision, recall, F1, AUC |
| Overfitting and regularization (15) | L2, dropout, early stopping, train vs validation curves |
| Hyperparameter tuning (16) | 6-setting grid in Part B |
| Autoencoders (28) | Part A |
""")
