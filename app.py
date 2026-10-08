import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Deep Learning: Customer Segmentation", page_icon="🧠", layout="wide")
D = "dl_results/"
R = json.load(open(D + "results.json"))
W = np.load(D + "model_weights.npz")
E = pd.read_csv(D + "embeddings.csv")
P = pd.read_csv(D + "autoencoder_segment_profiles.csv")
try:
    SN = json.load(open(D + "segment_names.json"))
except Exception:
    SN = {}
A, B = R["partA"], R["partB"]


def nm(k):
    return SN.get(str(k), {}).get("name", f"Segment {k}")


def act(k):
    return SN.get(str(k), {}).get("action", "-")


E["Segment"] = E["ae_cluster"].map(nm)
order = [nm(k) for k in P.sort_values("Total_Spending")["Segment"]]


def encode(x):  # trained encoder, run with plain numpy
    x = (np.asarray(x, float) - W["ae_mean"]) / W["ae_scale"]
    for i in range(3):
        x = x @ W[f"ae_w{i}"] + W[f"ae_b{i}"]
        if i < 2:
            x = np.maximum(x, 0)
    return x


def respond_prob(x):  # trained response network, run with plain numpy
    x = (np.asarray(x, float) - W["fnn_mean"]) / W["fnn_scale"]
    n = int(W["fnn_n"])
    for i in range(n):
        x = x @ W[f"fnn_w{i}"] + W[f"fnn_b{i}"]
        x = np.maximum(x, 0) if i < n - 1 else 1 / (1 + np.exp(-x))
    return float(x[0])


st.title("🧠 Can a neural network find types of customers?")
st.caption("DSE3120 Deep Learning · Nipun Bansal · Autoencoder-based customer segmentation + campaign-response network")
t1, t2, t3, t4, t5, t6 = st.tabs(["1 · The problem", "2 · How it works", "3 · Part A: customer types",
                                  "4 · Part B: who responds", "5 · Try it", "6 · Syllabus & glossary"])

with t1:
    a, b, c = st.columns(3)
    a.metric("Customers", f"{R['customers']:,}")
    b.metric("Facts per customer", 7)
    c.metric("Customers who accept a campaign", f"{B['positive_rate']:.0%}")
    st.markdown("A shop can't treat all customers the same. **Part A** asks: what types of customers exist? **Part B** asks: who will say yes to a marketing offer?")
    st.markdown("**The average customer** (the 7 facts we use):")
    avg = E[["Age", "Income", "Total_Spending", "NumWebPurchases", "NumStorePurchases", "NumWebVisitsMonth", "Recency"]].mean().round(0)
    avg.index = ["Age", "Income", "Total spending", "Web purchases", "Store purchases", "Web visits / month", "Days since last purchase"]
    st.dataframe(avg.rename("Average").to_frame().T, hide_index=True, width="stretch")
    st.info("Data was cleaned (missing incomes and impossible values removed) and **scaled**, so big numbers like income don't overpower small ones like days.")

with t2:
    st.subheader("Part A: the autoencoder, a copying game")
    st.graphviz_chart("""digraph { rankdir=LR; node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=12, color="#1F2A44", fontcolor=white];
      c [label="Customer\\n7 numbers", fillcolor="#1F2A44"]; e [label="Encoder\\n7 → 16 → 8 → 2", fillcolor="#0891B2"];
      s [label="Summary\\n2 numbers", fillcolor="#E8A33D", fontcolor="#1F2A44"]; d [label="Decoder\\n2 → 8 → 16 → 7", fillcolor="#0891B2"];
      r [label="Rebuilt\\n7 numbers", fillcolor="#1F2A44"]; m [label="Mistake = rebuilt vs original", fillcolor="#E11D48"];
      c -> e -> s -> d -> r; r -> m [style=dashed]; m -> e [style=dashed, label=" fix weights"]; }""")
    st.markdown("If the network can rebuild a customer from just **2 numbers**, those numbers must capture what matters. "
                "**K-Means** then groups customers using those 2 numbers (6 groups, chosen from the elbow curve).")
    st.subheader("Part B: the response network")
    st.graphviz_chart("""digraph { rankdir=LR; node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=12, fontcolor=white, color="#1F2A44"];
      i [label="10 facts", fillcolor="#1F2A44"]; h [label="16 neurons\\nReLU", fillcolor="#0891B2"]; dr [label="Dropout 0.3", fillcolor="#64748B"];
      o [label="1 neuron\\nsigmoid = chance of yes", fillcolor="#E8A33D", fontcolor="#1F2A44"]; i -> h -> dr -> o; }""")
    c1, c2, c3 = st.columns(3)
    c1.success("**Learns by trying:** measure the mistake, adjust the weights, repeat (backpropagation + Adam).")
    c2.success("**Avoids memorising:** early stopping, L2 penalty, dropout.")
    c3.success("**Handles rare 'yes':** class weights make the 15% who respond count more.")

with t3:
    st.subheader("Does the autoencoder beat the classic method (PCA)?")
    km, pc, ae = A["kmeans_on_7_features"], A["kmeans_on_pca_2d"], A["kmeans_on_autoencoder_2d"]
    a, b, c = st.columns(3)
    a.metric("Rebuild error (lower is better)", f"{A['reconstruction_mse']['autoencoder_2d']:.3f}",
             f"{A['reconstruction_mse']['autoencoder_2d'] - A['reconstruction_mse']['pca_2d']:+.3f} vs PCA", delta_color="inverse")
    b.metric("Group quality: silhouette (higher is better)", f"{ae['silhouette']:.3f}", f"{ae['silhouette'] - km['silhouette']:+.3f} vs K-Means")
    c.metric("Group overlap: Davies-Bouldin (lower is better)", f"{ae['davies_bouldin']:.2f}",
             f"{ae['davies_bouldin'] - km['davies_bouldin']:+.2f} vs K-Means", delta_color="inverse")
    st.caption("Honest verdict: it compresses better, but the grouping is not clearly better (one score up, one worse).")
    c1, c2 = st.columns([3, 2])
    c1.plotly_chart(px.scatter(E, x="z1", y="z2", color="Segment", category_orders={"Segment": order}, opacity=0.7,
                               labels={"z1": "Summary number 1", "z2": "Summary number 2"}, title="Each dot is a customer"), width="stretch")
    seg = P.assign(Name=P["Segment"].map(nm), Action=P["Segment"].map(act)).sort_values("Total_Spending")
    seg = seg[["Name", "Customers", "Age", "Income", "Total_Spending", "Recency", "Action"]].round(0)
    seg.columns = ["Segment", "People", "Avg age", "Avg income", "Avg spending", "Days since last buy", "Store should…"]
    c2.dataframe(seg, hide_index=True, width="stretch")
    with st.expander("Training details (loss curves, PCA vs autoencoder pictures)"):
        st.image(D + "partA_autoencoder.png")

with t4:
    st.subheader("Why accuracy lies")
    N, L, Z = B["neural_network"], B["logistic_regression"], B["always_predict_no"]
    models = ["Always say 'no'", "Simple model", "Neural network"]
    fig = go.Figure([go.Bar(name="Accuracy %", x=models, y=[Z["accuracy"] * 100, L["accuracy"] * 100, N["accuracy"] * 100], marker_color="#94A3B8", text=[f"{v*100:.0f}" for v in (Z["accuracy"], L["accuracy"], N["accuracy"])]),
                     go.Bar(name="Responders found %", x=models, y=[0, L["recall"] * 100, N["recall"] * 100], marker_color="#0891B2", text=[f"{v*100:.0f}" for v in (0, L["recall"], N["recall"])])])
    fig.update_layout(barmode="group", title="Saying 'no' to everyone looks accurate but finds nobody")
    c1, c2 = st.columns([3, 2])
    c1.plotly_chart(fig, width="stretch")
    (tn, fp), (fn, tp) = B["confusion_matrix_nn"]
    c2.metric("Customers who really would respond", tp + fn)
    c2.metric("Found by the network", tp)
    c2.metric("Missed", fn)
    c2.metric("False alarms", fp)
    st.info(f"Ranking quality (ROC-AUC): neural network {N['roc_auc']:.3f} vs simple model {L['roc_auc']:.3f}. "
            f"F1: network {N['f1']:.2f} vs simple model {L['f1']:.2f}. Deep learning helped a little, not a lot.")
    with st.expander("Tuning (6 settings) and training pictures"):
        g = pd.DataFrame(R["partB_tuning"]); g["hidden"] = g["hidden"].astype(str)
        st.dataframe(g, hide_index=True, width="stretch")
        st.image(D + "partB_neural_network.png")

with t5:
    st.subheader("Try the trained networks on a new customer")
    c = st.columns(5)
    age = c[0].number_input("Age", 18, 100, 45); inc = c[1].number_input("Income", 0, 200000, 60000, step=1000)
    spend = c[2].number_input("Total spending", 0, 3000, 800); rec = c[3].number_input("Days since last purchase", 0, 99, 30)
    kids = c[4].number_input("Children at home", 0, 5, 1)
    c = st.columns(5)
    wp = c[0].number_input("Web purchases", 0, 30, 4); sp = c[1].number_input("Store purchases", 0, 30, 6)
    wv = c[2].number_input("Web visits / month", 0, 20, 5); cp = c[3].number_input("Catalog purchases", 0, 30, 2)
    dp = c[4].number_input("Deal purchases", 0, 30, 2)
    if st.button("Run both networks", type="primary"):
        x7 = [age, inc, spend, wp, sp, wv, rec]
        z = encode(x7)
        k = int(np.argmin(((W["ae_centers"] - z) ** 2).sum(axis=1)))
        pr = respond_prob(x7 + [kids, cp, dp])
        a, b = st.columns(2)
        a.success(f"Customer type: **{nm(k)}**. Store should: **{act(k)}**")
        b.info(f"Chance of accepting a campaign: **{pr:.0%}**")
        fig = px.scatter(E, x="z1", y="z2", color="Segment", category_orders={"Segment": order}, opacity=0.3)
        fig.add_trace(go.Scatter(x=[z[0]], y=[z[1]], mode="markers", name="This customer", marker=dict(size=18, color="black", symbol="star")))
        st.plotly_chart(fig, width="stretch")

GLOSS = [("Neural network", "A chain of simple calculators (neurons) in layers. Each multiplies its inputs by weights, adds them up and passes the result on."),
         ("Training", "Guess, measure the mistake, nudge the weights to be less wrong, repeat thousands of times."),
         ("Backpropagation", "How the network finds which weights caused the mistake, working backwards from the output."),
         ("Gradient descent", "Taking small steps downhill on the 'mistake landscape' until the mistake is small. Step size = learning rate."),
         ("Adam", "A smarter gradient descent that adjusts the step size for every weight automatically."),
         ("ReLU", "Output 0 if the input is negative, otherwise pass it through. Lets networks learn curved patterns."),
         ("Sigmoid", "Squashes any number into 0-1, so it can be read as a probability (used at the end of Part B)."),
         ("Autoencoder", "A network that squeezes data into a small summary and tries to rebuild the original from it."),
         ("MSE (loss)", "Average squared difference between the rebuilt and the original numbers. Smaller is better."),
         ("Binary cross-entropy (loss)", "The standard mistake score for yes/no predictions."),
         ("Overfitting", "Doing great on training data but badly on new data, like memorising answers instead of understanding."),
         ("L2 penalty / Dropout / Early stopping", "Three ways to avoid overfitting: keep weights small, switch neurons off at random, stop training in time."),
         ("PCA", "A classic way to squeeze many numbers into fewer by keeping the directions with the most variation (straight-line patterns only)."),
         ("K-Means", "Picks 6 centres and puts each customer in the group with the nearest centre."),
         ("Precision / Recall / F1", "Precision: of those we flagged, how many were right. Recall: of the real ones, how many we found. F1 mixes both."),
         ("ROC-AUC", "How well the model ranks real responders above non-responders (0.5 = guessing, 1 = perfect).")]
with t6:
    st.markdown("""
| Syllabus topic (lecture) | Where it appears in this project |
|---|---|
| PCA (5) | Classic baseline compared with the autoencoder |
| Perceptron and neurons (8) | Every Dense layer is a layer of neurons |
| Activation functions (9) | ReLU in hidden layers, sigmoid at the end of Part B |
| Feedforward networks (10) | Part B |
| Backpropagation and gradient descent (11, 12) | How both networks were trained |
| Adam optimizer (13) | Used for both |
| Loss functions and metrics (14) | MSE, cross-entropy, precision, recall, F1, AUC |
| Overfitting and regularization (15) | L2, dropout, early stopping, training vs validation curves |
| Hyperparameter tuning (16) | 6-setting grid in Part B |
| Autoencoders (28) | Part A |
""")
    st.subheader("Key words in plain English")
    for t, d in GLOSS:
        with st.expander(t):
            st.write(d)
