# 📰 Fake News Detection in English Text

## 📌 Introduction
This project builds a system to **detect and classify fake news** using **Natural Language Processing (NLP)**.  
It combines **traditional machine learning models** and **modern Transformer architectures** (BERT, XLNet, RoBERTa) to compare performance. 

---

## 🎯 Objectives
- Analyze the linguistic characteristics of fake news.
- Build text classification models:
  - **Logistic Regression**, **SVM** (Machine Learning)
  - **BERT**, **XLNet**, **RoBERTa** (Transformers)
- Evaluate performance using **Accuracy**, **Precision**, **Recall**, and **F1 Score**.
- Analyze limitations and propose improvements.

---

## 📂 Dataset
- **MisinfoSuperset_TRUE.csv**: Real news (34,975 records) from reputable sources such as Reuters, The New York Times, The Washington Post...
- **MisinfoSuperset_FAKE.csv**: Fake news (43,642 records) from extremist sources and academic datasets.
- Each record contains:
  - `text`: News content
  - `target`: Label (`1` = real, `0` = fake)

---

## 🔍 Data Exploration & Preprocessing
- **Data check**:
  - 29 null values
  - 10,118 duplicate records
- **Analysis**:
  - Real news length: ~1,000 – 14,000 words
  - Fake news length: ~500 – 25,000 words
  - Frequent words:
    - Real news: `said`, `trump`, `mr`
    - Fake news: `trump`, `like`, ...
- **Preprocessing**:
  - Remove special characters, URLs, line breaks
  - Remove nulls and duplicates
  - Remaining: **68,499 records**
- **Train/Test split**:
  - Train: 80% (54,799 records)
  - Test: 20% (13,700 records)

---

## ⚙️ Methods & Models

### 1. Traditional Machine Learning Models
- **Vectorization**: `TfidfVectorizer`
  - `max_features=5000`
  - `ngram_range=(1,2)`
  - Remove English stopwords
- **Logistic Regression**
  - GridSearchCV (C, penalty, solver, max_iter=1000)
- **SVM**
  - Linear SVM + GridSearchCV (C, max_iter=1000)

### 2. Transformer Models
Process:
1. Tokenizer
2. DataLoader
3. Model building
4. Hyperparameter tuning
5. Training (3 epochs, ~15 mins/epoch)
6. Evaluation
7. Inference

#### BERT (`bert-base-uncased`)

#### XLNet (`xlnet-base-cased`)

#### RoBERTa (`roberta-base`)

---

## 📊 Results
| Model | Accuracy | F1 Score |
|-------|----------|----------|
| Logistic Regression | 0.945 | 0.9457 |
| SVM | 0.94 | 0.9452 |
| BERT | 0.99 | 0.99 |
| XLNet | 0.99 | 0.99 |
| RoBERTa | 0.99 | 0.99 |

---

## ⚠️ Limitations & Challenges
- High training time and resource requirements (GPU recommended).

- Dataset may be outdated → models may underperform on very recent news.

- Data covers limited topics (politics, economics).

- Models trained on English → performance may drop for other languages.

---

## 🔮 Future Work
- Update datasets with more recent and diverse content.

- Test on Vietnamese or other languages.

- Deploy as an API or web application.

- Use ensemble methods for higher accuracy.

---

##  Installation

 Clone the repo:
```bash
git clone https://github.com/tachirevlt/detect_fake_news.git
```


## Author

[![GitHub](https://img.shields.io/badge/GitHub-tachirevlt-blue?logo=github)](https://github.com/tachirevlt)
