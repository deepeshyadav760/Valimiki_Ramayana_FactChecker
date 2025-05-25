# 🧠 Ramayana Fact Verification

This project aims to verify factual statements related to the Ramayana using semantic search and natural language processing (NLP) techniques. It includes data preprocessing, embedding generation, and fact-checking functionalities through custom scripts and models.

---

## 📁 Project Structure

```
ramayana-fact-verification/
│
├── .venv/                       # Python virtual environment (ignored by Git)
├── __pycache__/                # Cache files (ignored by Git)
│
├── ramayana_data/
│   ├── merged_ramayana.csv          # Combined data from Ramayana for reference
│   ├── ramayana_embeddings.pkl      # Precomputed sentence embeddings
│
├── data_scrapper.py            # Script to scrape or load data from external sources
├── embeddings.py               # Embedding generation using language models
├── evaluation.py               # Evaluation metrics and testing
├── fact_verification.py        # Main logic to verify facts based on embeddings
├── main_function.py            # Entry point script to run the project
├── output_verified.csv         # Output CSV with verification results
├── preprocessing.py            # Text preprocessing (tokenization, cleaning, etc.)
├── ramayana_verify.py          # Script to run batch verification on dataset
├── semantic_search.py          # Implements vector-based semantic search
│
├── requirements.txt            # List of Python dependencies
├── statements.csv              # Input file with factual statements to verify
├── README.md                   # Project documentation (this file)
```

---

## 🚀 How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/deepeshyadav760/Valimiki_Ramayana_FactChecker.git
cd ramayana-fact-verification
```

### 2. Set Up Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Main Script

```bash
python main_function.py
```

---

## ✅ Features

- Preprocessing of mythological texts
- Sentence embedding generation using pre-trained models
- Semantic similarity-based fact verification
- Batch processing and CSV output of verification results

---

## 📊 Sample Input

**statements.csv**
```csv
statement
"Rama was the son of Dasharatha."
"Sita was kidnapped by Ravana."
```

---

## 📤 Output

**output_verified.csv**
```csv
statement,verified,score
"Rama was the son of Dasharatha.",True,0.92
"Sita was kidnapped by Ravana.",True,0.88
```

---

## 🛠️ Technologies Used

- Python 🐍
- Sentence Transformers / SBERT
- Pandas, NumPy
- Scikit-learn
- VS Code

---

## 📚 Dataset

The reference content is extracted and preprocessed from structured summaries or texts of the Ramayana. The merged content is stored in `merged_ramayana.csv`.

---

## 📌 To Do

- Improve accuracy with custom fine-tuned models
- Add UI to verify facts interactively
- Add support for other epics like Mahabharata

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you'd like to change.

---
