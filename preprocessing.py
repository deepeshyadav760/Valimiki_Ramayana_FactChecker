import os
import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# def load_ramayana_dataset(file_path="ramayana_data\merged_ramayana.csv"):
def load_ramayana_dataset(file_path=os.path.join("ramayana_data", "merged_ramayana.csv")):
    """
    Load Ramayana dataset from CSV file
    
    Args:
        file_path: Path to the CSV file containing Ramayana verses
        
    Returns:
        DataFrame containing all verses
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CSV file not found at {file_path}")
    
    # Load the CSV with the column names from the provided data
    df = pd.read_csv(file_path)
    
    # Check if we have the expected columns
    expected_cols = ['Kanda/Book', 'Sarga/Chapter', 'Shloka/Verse Number', 'English Translation']
    missing_cols = [col for col in expected_cols if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Rename columns to match the processing functions
    column_mapping = {
        'Kanda/Book': 'kanda',
        'Sarga/Chapter': 'sarga',
        'Shloka/Verse Number': 'verse_num',
        'English Translation': 'english_text'
    }
    
    df = df.rename(columns=column_mapping)
    
    # Create a reference column (used in the original code)
    df['reference'] = df.apply(lambda row: f"{row['kanda']}, Sarga {row['sarga']}, Verse {row['verse_num']}", axis=1)
    
    return df

def clean_text(text):
    """
    Clean and normalize text
    
    Args:
        text: Text to clean
        
    Returns:
        Cleaned text
    """
    if pd.isna(text) or not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and numbers
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def preprocess_text(text, remove_stopwords=True, lemmatize=True):
    """
    Preprocess text with tokenization, stopword removal, and lemmatization
    
    Args:
        text: Text to preprocess
        remove_stopwords: Whether to remove stopwords
        lemmatize: Whether to lemmatize words
        
    Returns:
        List of processed tokens
    """
    if pd.isna(text) or not isinstance(text, str):
        return []
    
    # Clean the text
    text = clean_text(text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatize
    if lemmatize:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return tokens

def preprocess_dataset(df, text_column='english_text'):
    """
    Preprocess the entire dataset
    
    Args:
        df: DataFrame containing Ramayana verses
        text_column: Column containing the text to preprocess
        
    Returns:
        DataFrame with added preprocessed text columns
    """
    # Create a copy to avoid modifying the original
    processed_df = df.copy()
    
    # Add cleaned text
    processed_df['cleaned_text'] = processed_df[text_column].apply(clean_text)
    
    # Add tokenized text
    processed_df['tokens'] = processed_df[text_column].apply(preprocess_text)
    
    # Add processed text (rejoined tokens)
    processed_df['processed_text'] = processed_df['tokens'].apply(lambda x: ' '.join(x))
    
    return processed_df

def preprocess_statement_for_verification(statement):
    """
    Preprocess a statement for verification using only clean_text function.
    This uses only basic text cleaning without tokenization, stopword removal, or lemmatization.
    
    Args:
        statement (str): Original statement to preprocess
        
    Returns:
        str: Preprocessed statement ready for verification (clean_text only)
    """
    if pd.isna(statement) or not isinstance(statement, str):
        return ""
    
    # Use only the clean_text function - no tokenization, stopwords, or lemmatization
    processed = clean_text(statement)
    
    return processed

def preprocess_statement_batch(statements):
    """
    Preprocess a batch of statements for verification using clean_text only.
    
    Args:
        statements (list): List of statements to preprocess
        
    Returns:
        list: List of preprocessed statements
    """
    return [preprocess_statement_for_verification(stmt) for stmt in statements]

def compare_preprocessing_methods(statement):
    """
    Compare different preprocessing methods on a statement.
    Useful for debugging and choosing the best preprocessing approach.
    
    Args:
        statement (str): Statement to test
        
    Returns:
        dict: Dictionary with different preprocessing results
    """
    results = {
        'original': statement,
        'clean_only': clean_text(statement),  # This is what we're using now
        'tokens_only': preprocess_text(statement),
        'dataset_style': ' '.join(preprocess_text(statement))  # Full dataset preprocessing for comparison
    }
    
    return results

# Main preprocessing function
# def prepare_ramayana_data(file_path="ramayana_data\merged_ramayana.csv"):
def prepare_ramayana_data(file_path=os.path.join("ramayana_data", "merged_ramayana.csv")):
    """
    Load and preprocess the Ramayana dataset
    
    Args:
        file_path: Path to the CSV file containing Ramayana verses
        
    Returns:
        Preprocessed DataFrame
    """
    print("Loading Ramayana dataset...")
    df = load_ramayana_dataset(file_path)
    
    print(f"Loaded {len(df)} verses from {len(df['kanda'].unique())} kandas")
    
    print("Preprocessing text...")
    processed_df = preprocess_dataset(df)
    
    print("Preprocessing complete!")
    return processed_df

def test_statement_preprocessing():
    """
    Test the statement preprocessing with example statements using clean_text only.
    """
    test_statements = [
        "Rama is the eldest son of King Dasharatha.",
        "Lakshmana, Rama's devoted younger brother, accompanied him into exile.",
        "Bharata, another brother of Rama, revered him and ruled as regent in his absence.",
    ]
    
    print("=== Statement Preprocessing Test (Clean Text Only) ===")
    for i, stmt in enumerate(test_statements, 1):
        print(f"\nTest Statement {i}:")
        
        original = stmt
        clean_only = clean_text(stmt)
        
        print(f"  Original:   {original}")
        print(f"  Clean Only: {clean_only}")
    
    return test_statements

if __name__ == "__main__":
    # Test the main preprocessing function
    processed_data = prepare_ramayana_data()
    print(f"Processed dataframe shape: {processed_data.shape}")
    print(processed_data.head(2))
    
    # Test statement preprocessing
    print("\n" + "="*80)
    test_statement_preprocessing()