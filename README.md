# 🧠 Customer Segmentation with Deep Learning (DSE3120)

An autoencoder compresses each customer to 2 numbers and K-Means groups them (Part A). A feedforward neural network predicts who accepts a marketing campaign (Part B). A Streamlit app presents the results and runs the trained networks on a new customer.
Try link: https://fz6evdpigphakicakxhte2.streamlit.app

## Files
| File | Purpose |
|---|---|
| `dl_segmentation.py` | Trains both networks (Keras/TensorFlow), saves results and weights |
| `app.py` | Streamlit app (runs the saved weights with numpy, so TensorFlow is NOT needed to host it) |
| `dl_results/` | Saved charts, metrics (`results.json`), segment profiles, embeddings, model weights, `segment_names.json` (friendly names and store actions for the 6 segments) |
| `customer_segmentation.csv` | Dataset (about 2,200 supermarket customers) |
| `requirements.txt` | Packages for the Streamlit app |
| `requirements_train.txt` | Extra packages to retrain the models |

## Run the app
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Retrain the models (optional, takes a few minutes)
```bash
pip install -r requirements_train.txt
python dl_segmentation.py
```

## Method
- Cleaning: drop missing income, impossible ages, income outliers; 7 features standardised.
- Autoencoder: 7 → 16 → 8 → 2 → 8 → 16 → 7, ReLU, Adam, MSE, L2, early stopping.
- Baselines: K-Means on the 7 features, and K-Means on PCA (2D).
- Response network: 10 inputs → Dense ReLU → Dropout → sigmoid, binary cross-entropy, class weights, 6 hyperparameter settings.
- Evaluation: silhouette, Davies-Bouldin and reconstruction error (Part A); accuracy, precision, recall, F1, ROC-AUC against logistic regression (Part B).

## Honest results
The autoencoder reconstructs customers better than PCA, but its clustering gain is mixed. The neural network has a better ROC-AUC than logistic regression but not a better F1.

Author: Nipun
