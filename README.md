**作品集 README**

## 背景與靈感

本作品集靈感來源於 Marcos López de Prado 所著之《Advances in Financial Machine Learning》，聚焦於**AI 與機器學習在金融市場上的實際應用**。透過書中所介紹的進階方法，如 Meta-labeling、Purged K-Fold、Sequential Bootstrap，以及特徵重要性計算（MDI、MDA），構建一條從資料處理、特徵工程到交易策略回測的完整流程。

## 作品集概述

本作品集包含三個相互串聯的子專案：

1. **商品特徵分群 (PCA + KMeans)**
2. **特徵重要性計算 (MDI/MDA/SFI)**
3. **交易模型建置 (Meta-labeling + 模型訓練)** 

每個步驟皆採用《Advances in Financial ML》中的方法論，將其作為下一步驟的輸入，最終利用機器學習產生交易模型。

---

## 專案流程

### 1. 商品特徵分群  (PCA + KMeans) - Project1_Clustering.ipynb

Motivation: 在訓練模型時用同性質商品而非單一商品資料進行訓練，能增加模型的泛化能力，降低訓練時overfitting的機率，此Project的目的為將性質類似之商品分類，來建立適合共同訓練之資料集

* **輸入**：券商資料集
* **步驟**：
    1. 讀取29種商品資料，包含貨幣兌、黃金、原油、指數等商品
    2. 將資料轉換為dollar bars(以固定金額為基準的bar，性質較time bars穩定)
    3. 計算指標，並對非平穩指標進行Fractional Difference，讓數據平穩化的同時保持記憶性
    4. 使用PCA(主成分分析)將資料正交化
    5. 使用Kmeans將PCA的結果分群，並將相似的商品歸類在同一群組
* **輸出**：`clusters.csv`

### 2. 特徵重要性計算 (MDI/MDA/SFI)- Project2_Feature_Importance.ipynb

Motivation：在金融領域中，特徵選擇對模型性能與穩健性至關重要。透過隨機森林的 MDI (Mean Decrease Impurity) 來量化每項特徵對模型的貢獻，同時以 MDA (Mean Decrease Accuracy) 和 SFI(Single Factor Importance)來評估特徵在不同資料抽樣下的穩定性，並利用PCA降低特徵間的substitution effect。此流程能協助識別關鍵且可靠的特徵，降低過度擬合風險並提升策略泛化能力。

* **輸入**：`clusters.csv` 和券商資料集
* **步驟**：
      1. 篩選含黃金的群集資料
      1. 將資料轉換為dollar bars
      1. 進行metalabling產生標籤
      1. 計算指標，並對非平穩指標進行Fractional Difference
      1. 使用PCA排除共線性，降低特徵間的substitution effect
      1. 用Purged K-Fold CrossValidation和RandomForest，計算 MDI/MDA/SFI
* **輸出**：`mdi.csv`, `mda.csv`, `sfi.csv`

### 3. 交易模型建置 (Meta-labeling + 模型訓練) - Project3_MetaLabeling_Trading.ipynb

本步驟運用第 6 章之 Meta-labeling 流程，並結合第 4 章 Purged K-Fold 及 Sequential Bootstrap 進行回測驗證。

1. **Meta-labeling**：設定 PT/SL 門檻與垂直障礙，生成二次標籤
2. **特徵挑選**：取用 MDI/MDA 前 N 大特徵
3. **AI 模型訓練**：使用 LightGBM/XGBoost，並以 Purged K-Fold 驗證防止資訊洩漏
4. **回測評估**：計算 Sharpe Ratio、最大回撤等績效指標

* **輸入**：`data/raw/price_data.csv`、`data/processed/feature_importance.csv`
* **輸出**：`models/trade_model.pkl`、`reports/backtest_report.html`

---

## 專案目錄結構

```
portfolio/
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── reports/
├── src/
│   ├── pca_kmeans.py
│   ├── feature_importance.py
│   └── trade_model.py
├── notebooks/
├── requirements.txt
└── README.md
```

---

## 參考

* López de Prado, M. (2018). *Advances in Financial Machine Learning*.
