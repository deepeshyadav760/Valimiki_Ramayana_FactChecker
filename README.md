#                                                        🧠 Ramayana Fact Verification

This project is designed to verify factual claims against the verses of the **Ramayana**. It uses semantic search and natural language processing techniques to determine whether a given statement is **True**, **False**, or **Unverifiable** based on the scripture.

## 📂 Project Structure

```
ramayana-fact-verification/
│
├── ramayana_data/
│   ├── merged_ramayana.csv               # Scraped verses with translations
│   ├── ramayana_embeddings.pkl           # Embeddings generated for search
│
├── data_scrapper.py                      # Script to scrape Ramayana verses
├── embeddings.py                         # Script to generate embedding
├── output_verified.csv                   # Output of verified facts
├── ramayana_verify.py                    # Entry point for verifying user statements
├── requirements.txt                      # Required Python packages
├── statements.csv                        # Input CSV with statements to verify
├── README.md                             # Project documentation
```

## Algorithm Flow
                                                STATEMENT INPUT
                                                      │
                                                      ▼
                                                ┌─────────────┐
                                                │   STEP 1    │ ──── Text Cleaning & Normalization
                                                │  PREPROCESS │      ├── Remove special characters
                                                └─────────────┘      ├── Convert to lowercase
                                                      │              └── Handle whitespace
                                                      ▼
                                                ┌─────────────┐
                                                │   STEP 2    │ ──── Relevance Filtering
                                                │   FILTER    │      ├── Detect modern content
                                                └─────────────┘      └── Check for vague statements
                                                      │
                                                      ▼
                                                ┌─────────────┐
                                                │   STEP 3    │ ──── Semantic Embedding
                                                │   EMBED     │      ├── Generate statement vector
                                                └─────────────┘      └── Load verse embeddings
                                                      │
                                                      ▼
                                                ┌─────────────┐
                                                │   STEP 4    │ ──── Similarity Computation
                                                │  SIMILARITY │      ├── Calculate cosine similarity
                                                └─────────────┘      └── Rank verses by similarity
                                                      │
                                                      ▼
                                                ┌─────────────┐
                                                │   STEP 5    │ ──── Semantic Analysis
                                                │   ANALYZE   │      ├── Word overlap analysis
                                                └─────────────┘      └── Negation conflict detection
                                                      │
                                                      ▼
                                                ┌─────────────┐
                                                │   STEP 6    │ ──── Truth Classification
                                                │  CLASSIFY   │      ├── Statistical scoring
                                                └─────────────┘      └── Threshold-based decision
                                                      │
                                                      ▼
                                                RESULT OUTPUT (True/False/None)

## 📌 Features

- 📜 Scrapes all verses of Ramayana from online sources.
- 🔎 Embeds and stores the verses using semantic embeddings.
- ✅ Checks factual accuracy of input statements using semantic similarity.
- 📄 Outputs results with labels: `True`, `False`, or `None`.

---

## 🧰 Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/deepeshyadav760/Valimiki_Ramayana_FactChecker.git
   cd ramayana-fact-verification
   ```

2. (Optional but recommended) Create a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

---

## 📥 Data Scraping

To scrape the data from the Srimad Valmiki Ramayana -> `https://valmikiramayan.net/`

1. Run the `data_scrapper.py` script:

   ```bash
   python data_scrapper.py
   ```

2. This will generate a CSV file named `merged_ramayana.csv` with the following columns:

   - `Kanda/Book`
   - `Sarga/Chapter`
   - `Shloka/Verse Number`
   - `English Translation`

   ✅ **All verses from the 6 Kandas will be scraped and stored in this single CSV file.**

---

## 🚀 How to Run the Fact Verification

1. Prepare a `statements.csv` file with **only one column**:

   ```
   Statement
   ```
   Example content:
   ```
   Rama went to exile for 14 years.
   Hanuman burned Lanka.
   ```

2. Provide the path to input file -> `statements.csv` and `output csv file` inside the ramayana_verify.py. Then Run the `ramayana_verify.py` script.

   ```bash
   python ramayana_verify.py
   ```

3. The script will process the statements and generate an `output.csv` file with the results:

   - `True` – the statement matches a known verse
   - `False` – contradicts known verses
   - `None` – unverifiable / ambiguous

---

## 📊 Example Output

| ID | Statement                                                                 | Truth |
|----|---------------------------------------------------------------------------|--------|
| 1  | Rama is the eldest son of King Dasharatha.                                | True   |
| 2  | Sita was discovered by King Janaka in a furrow during ploughing and was later adopted by him. | True   |
| 3  | Lakshmana, Rama’s devoted younger brother, accompanied him into exile.    | True   |
| 4  | Bharata, another brother of Rama, revered him and ruled as regent in his absence. | True   |

---

## 🤝 Contributing

Contributions are welcome! Please feel free to open issues or submit pull requests for improvements.
