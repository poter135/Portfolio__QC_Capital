# 作品集總覽 

## 目錄
- [背景與靈感](#背景與靈感)
- [作品集結構](#作品集結構)
- [Project1：商品特徵分群 (Clustering: PCA + KMeans)](#project1商品特徵分群-clustering-pca--kmeans)
- [Project2：特徵重要性計算 (Feature Importance: MDI / MDA / SFI)](#project2特徵重要性計算-feature-importance-mdi--mda--sfi)
- [Project3：交易模型建置 (Meta-Labeling + 模型訓練)](#project3交易模型建置-meta-labeling--模型訓練)
- [專案結構](#專案結構)
- [參考文獻](#參考文獻)


## 背景與靈感

本作品集靈感來源於 Marcos López de Prado 所著之《Advances in Financial Machine Learning》，聚焦於**機器學習在金融市場上的實際應用**。  
透過書中所介紹的進階方法，如 Meta-Labeling、Purged K-Fold、Sequential Bootstrap，以及特徵重要性計算（MDI、MDA、SFI），構建一條從資料處理、特徵工程到建構交易模型的完整流程。

---

## 作品集結構

本作品集包含三個彼此獨立卻相互串聯的子專案，每一步驟皆採用《Advances in Financial Machine Learning》中的方法論，最終利用機器學習產生交易模型：

1. **Project1：商品特徵分群 (Clustering: PCA + KMeans)**  
2. **Project2：特徵重要性計算 (Feature Importance: MDI / MDA / SFI)**  
3. **Project3：Meta-Labeling應用示範**  

下列章節將依序介紹每個子專案的動機、流程及輸入輸出。

> **Warning**：本作品集重點在於展示，並非要跑出極致的策略績效。若要在實際交易環境應用，仍需進一步加入嚴格的回測流程（如 Purged K-Fold CV、PBO、CPCV）與不同的模型選擇。
---

## 1. Project1：商品特徵分群 (Clustering: PCA + KMeans)

### 動機
- 單一商品資料訓練往往難以泛化，容易產生過度擬合。  
- 將性質相似的多種商品（貨幣對、貴金屬、原油、指數等）分群，可讓模型在「同質性更高」的資料群上訓練，降低噪訊影響並提升泛化能力。

### 流程
1. **資料讀取與清理**  
   - 載入 27 種商品歷史價格 CSV，包含：貨幣對 (EURUSD、USDJPY…)、貴金屬 (XAUUSD、XAGUSD)、能源 (原油)、指數 (SPX、DAX…) 等。  
   - 統一欄位名稱與時間索引格式。

2. **Dollar Bars 生成**  
   - 使用累積成交金額閾值 (Dollar Volume) 切割 Bar，相較 Time Bars 更能反映市場流動。  
   - 輸出每個 Dollar Bar 的 OHLCV 資訊，存於 `data/dollar_bars/`。

3. **技術指標計算與 Fractional Differentiation (FFD)**  
   - 在 Dollar Bars 資料基礎上，計算 RSI、ATR、SMA、EMA、布林通道、成交量指標等。  
   - 針對非平穩序列應用 FFD 技術，平穩化同時保留長期記憶性。

4. **PCA 主成分分析**  
   - 對標準化後的特徵矩陣做 PCA，保留累積解釋變異量 ≥ 95% 的主成分，達到降維與去除共線性。  
   - 輸出每個商品在主成分空間的坐標 (Principal Component Scores)。

5. **KMeans 分群**  
   - 以主成分坐標為輸入，使用 KMeans 演算法分群。  
   - 根據Silhouette Score選擇最佳群數 \(K\)。  
   - 最終輸出 `clusters.csv`，記錄「商品名稱 → 群組編號」。

### 輸入
- 券商資料(檔案過大沒有放在作品集裡)：27 種商品原始價格資料 (CSV)。

### 輸出
- `results/clusters.csv`：每個商品對應群組編號。

---

## 2. Project2：特徵重要性計算 (Feature Importance: MDI / MDA / SFI)

### 動機
- 金融資料中常含大量技術指標與衍生特徵，若直接投入模型，容易造成過度擬合且難以辨識關鍵指標。  
- 透過三種重要性方法 (MDI、MDA、SFI) 計算特徵貢獻度，再加上 PCA 移除共線性，能識別對後續策略最有價值的特徵，降低過擬合風險並提升策略穩健性。  
- 本專案以「黃金所在之群組」為示範，示範整個特徵選取流程。

### 流程
1. **資料讀取與群組篩選**  
   - 讀取 `clusters.csv`，篩選出「含黃金(XAUUSD)」的群組清單。  
   - 讀取該群組內各商品的原始價格資料及 Dollar Bars。

2. **Meta-Labeling 產生標籤**  
   - 使用 CUSUM Filter + Triple Barrier Method，對每個事件 (t_event) 進行標籤。  
   - 計算事件唯一性 (Uniqueness)，作為後續訓練之抽樣權重。

3. **技術指標計算與 Fractional Differentiation (FFD)**  
   - 對每段事件存續期間計算 RSI、ATR、SMA、EMA…等指標，並對非平穩序列做 FFD。  
   - 合併不同商品指標並標準化，組成最終特徵矩陣。

4. **PCA 移除共線性**  
   - 對標準化後的特徵矩陣進行 PCA，保留足以解釋變異量的主成分，降低特徵間 substitution effect。

5. **Purged K-Fold Cross-Validation + 隨機森林訓練**  
   - 在每個 Fold 中，先 Purging (剔除與驗證集重疊之事件) 並應用 Embargo。  
   - 使用唯一性 (Uniqueness) 作為樣本權重，並透過 Sequential Bootstrap 重建訓練集。  
   - 計算三種重要性指標：  
     - **MDI (Mean Decrease in Impurity)**：隨機森林節點分裂所減少的 Entropy 累計值。  
     - **MDA (Mean Decrease in Accuracy)**：Permutation Importance，衡量模型準確度下降程度。  
     - **SFI (Single Feature Importance)**：單特徵訓練模型，評估其預測能力。  

6. **結果匯出與可視化**  
   - 輸出 `mdi.csv`、`mda.csv`、`sfi.csv`，記錄各特徵之重要性排序。  
   - 輸出 `pca pipline` 用於後續資料處裡
   - 繪製特徵重要性長條圖，用以比較不同方法下特徵分佈。

### 輸入
- `results/clusters.csv`：Project1 輸出之商品群組編號。  
- 券商資料(檔案過大沒有放在作品集裡)：27 種商品原始價格資料 (CSV)。

### 輸出
- `results/mdi.csv`：MDI 排序結果。  
- `results/mda.csv`：MDA 排序結果。  
- `results/sfi.csv`：SFI 排序結果。 
- `models/pipeline_scaler_pca.jolib` : pca pipeline。 

---

## 3. Meta-Labeling應用示範

### 動機
- 直接在原始市場資料上套用機器學習，容易因市場結構差異而失敗。  
- Meta-Labeling 核心在於：先用「Primary Model」區分是市場結構後，能讓演算法學到性質相似之資料。搭配資料的uniqueness作為權重，可減少資訊洩漏並提高策略穩健度。

### 流程
1. **Meta-Labeling 標籤生成**  
   - 使用 CUSUM Filter 偵測事件 (t_event)，並套用 Triple Barrier Method 設定停利 (PT)、停損 (SL) 與垂直障礙 (Vertical Barrier)，生成最終標籤 (多頭/空頭/中性)。  
   - 計算事件唯一性 (Uniqueness)，作為樣本權重。

2. **特徵篩選**  
   - 從 Project2 輸出之重要性文件 (`mdi.csv`、`mda.csv`、`sfi.csv`) 中，選取有效特徵作為模型輸入。

3. **模型訓練**  
   - 為了克服金融資料非 i.i.d. 的特性，本專案實作一個 Custom Random Forest，修改其 Bootstrap 抽樣機制，使每棵決策樹以「樣本權重」作為抽樣依據。
   - 將第一步所標記的事件（含 Label 與權重）與第二步選出的特徵一起，輸入自訂 Random Forest 進行訓練。
   - 最終比較 Train/Test 資料集上的模型效能，以驗證 Meta-Labeling 策略在不同商品上的適用性與泛化能力。

### 輸入
- 券商資料(檔案過大沒有放在作品集裡)：27 種商品原始價格資料 (CSV)。
- `restuls/clusters`: Project1 輸出之商品群組編號。  
- `results/combined_feature_importance.csv`：Project2 綜合排序後的特徵清單。

### 輸出
- `models/baggin_RF.joblib`：最終訓練好的Random Forest。  
- `results/classification_report(Train).csv` `classification_report(Test).csv`：模型的訓練成果。

---

## 專案目錄結構
```
Portfolio__QC_Capital/
├── .gitignore
├── intermediate_results/
├── models/
│   ├── bagging_RF.joblib
│   └── pipeline_scaler_pca.joblib
├── results/
│   ├── classification_report(Test).csv
│   ├── classification_report(Train).csv
│   ├── clusters.csv
│   ├── mda.csv
│   ├── mdi.csv
│   └── sfi.csv
├── Project1_Clustering.ipynb
├── Project2_Features_Importance.ipynb
├── Project3_MetaLabeling.ipynb
└── README.md
```

---
## 參考文獻
López de Prado, M. (2018). Advances in Financial Machine Learning. Wiley.
