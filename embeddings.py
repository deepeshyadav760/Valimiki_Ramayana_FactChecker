import numpy as np
import os
import pickle
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

def load_embedding_model(model_name="all-MiniLM-L6-v2"):
    """
    Load a sentence transformer model for creating embeddings
    
    Args:
        model_name: Name of the SentenceTransformer model to use
        
    Returns:
        Loaded model
    """
    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    return model

def create_verse_embeddings(df, model, text_column='processed_text', batch_size=32):
    """
    Create embeddings for all verses in the dataset
    
    Args:
        df: DataFrame containing preprocessed verses
        model: SentenceTransformer model
        text_column: Column containing the text to embed
        batch_size: Batch size for embedding generation
        
    Returns:
        DataFrame with added embeddings
    """
    # Create a copy to avoid modifying the original
    embedding_df = df.copy()
    
    # Get all texts to embed
    texts = embedding_df[text_column].tolist()
    
    # Generate embeddings in batches
    print(f"Generating embeddings for {len(texts)} verses...")
    all_embeddings = []
    
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i+batch_size]
        batch_embeddings = model.encode(batch_texts)
        all_embeddings.extend(batch_embeddings)
    
    # Add embeddings to DataFrame
    embedding_df['embedding'] = all_embeddings
    
    print("Embedding generation complete!")
    return embedding_df

def save_embeddings(df, embeddings_path):
    """
    Save the DataFrame with embeddings
    
    Args:
        df: DataFrame with embeddings
        embeddings_path: Path to save the DataFrame
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(embeddings_path), exist_ok=True)
    
    # Save with pickle to preserve numpy arrays
    with open(embeddings_path, 'wb') as f:
        pickle.dump(df, f)
    
    print(f"Embeddings saved to {embeddings_path}")

def load_embeddings(embeddings_path):
    """
    Load the DataFrame with embeddings
    
    Args:
        embeddings_path: Path to the saved DataFrame
        
    Returns:
        DataFrame with embeddings
    """
    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(f"Embeddings file not found at {embeddings_path}")
    
    with open(embeddings_path, 'rb') as f:
        df = pickle.load(f)
    
    print(f"Embeddings loaded from {embeddings_path}")
    return df

def create_and_save_embeddings(processed_df, model_name="all-MiniLM-L6-v2", 
                              embeddings_path=os.path.join("ramayana_data", "ramayana_embeddings.pkl")):
    """
    Create and save embeddings for the Ramayana dataset
    
    Args:
        processed_df: Preprocessed DataFrame
        model_name: Name of the SentenceTransformer model to use
        embeddings_path: Path to save the embeddings
        
    Returns:
        DataFrame with embeddings
    """
    # Check if embeddings already exist
    if os.path.exists(embeddings_path):
        print(f"Loading existing embeddings from {embeddings_path}")
        return load_embeddings(embeddings_path)
    
    # Load model and create embeddings
    model = load_embedding_model(model_name)
    df_with_embeddings = create_verse_embeddings(processed_df, model)
    
    # Save embeddings
    save_embeddings(df_with_embeddings, embeddings_path)
    
    return df_with_embeddings

if __name__ == "__main__":
    # Import preprocessing to get the processed data
    from preprocessing import prepare_ramayana_data
    
    # Get the processed dataframe - specify the path to your CSV file
    processed_df = prepare_ramayana_data("ramayana_data/merged_ramayana.csv")
    
    # Create and save embeddings
    df_with_embeddings = create_and_save_embeddings(processed_df)
    
    # Print some information about the embeddings
    print(f"\nEmbedding shape: {df_with_embeddings['embedding'][0].shape}")
    print(f"Total number of verses with embeddings: {len(df_with_embeddings)}")