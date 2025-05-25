import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from preprocessing import clean_text, preprocess_text

def semantic_search(query, df_with_embeddings, model, top_k=5):
    """
    Find the most semantically similar verses to a query
    
    Args:
        query: Input query text
        df_with_embeddings: DataFrame with verse embeddings
        model: SentenceTransformer model for creating query embedding
        top_k: Number of top matches to return
        
    Returns:
        List of dictionaries containing matched verses and similarity scores
    """
    # Preprocess the query
    processed_query = ' '.join(preprocess_text(query))
    
    # Get query embedding
    query_embedding = model.encode([processed_query])[0]
    
    # Get all verse embeddings
    verse_embeddings = np.array(df_with_embeddings['embedding'].tolist())
    
    # Calculate cosine similarity
    similarities = cosine_similarity([query_embedding], verse_embeddings)[0]
    
    # Get top_k indices
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    # Create results
    results = []
    for idx in top_indices:
        verse = df_with_embeddings.iloc[idx]
        
        # Handle potential column name differences
        text = verse['english_text'] if 'english_text' in verse else verse.get('text', '')
        
        # Access kanda directly or extract from reference if needed
        kanda = verse['kanda'] if 'kanda' in verse else verse['reference'].split(',')[0].strip() if 'reference' in verse else ''
        
        # Create result dictionary
        results.append({
            'kanda': kanda,
            'sarga': verse.get('sarga', ''),
            'verse_num': verse.get('verse_num', ''),
            'text': text,
            'reference': verse.get('reference', f"{kanda}, Sarga {verse.get('sarga', '')}, Verse {verse.get('verse_num', '')}"),
            'similarity': similarities[idx]
        })
    
    return results

def get_relevant_verses(statement, df_with_embeddings, model, threshold=0.5, top_k=10):
    """
    Get relevant verses for a statement with a minimum similarity threshold
    
    Args:
        statement: Input statement to check
        df_with_embeddings: DataFrame with verse embeddings
        model: SentenceTransformer model
        threshold: Minimum similarity threshold
        top_k: Maximum number of verses to return
        
    Returns:
        List of relevant verses that exceed the threshold
    """
    # Find similar verses
    matches = semantic_search(statement, df_with_embeddings, model, top_k=top_k)
    
    # Filter by threshold
    relevant_verses = [match for match in matches if match['similarity'] >= threshold]
    
    return relevant_verses

def is_statement_relevant_to_ramayana(statement, df_with_embeddings, model, 
                                     relevance_threshold=0.4, top_k=3):
    """
    Check if a statement is relevant to the Ramayana
    
    Args:
        statement: Input statement to check
        df_with_embeddings: DataFrame with verse embeddings
        model: SentenceTransformer model
        relevance_threshold: Threshold for determining relevance
        top_k: Number of top matches to consider
        
    Returns:
        Boolean indicating whether the statement is relevant to Ramayana and the relevance score
    """
    # Get the most similar verses
    matches = semantic_search(statement, df_with_embeddings, model, top_k=top_k)
    
    # Get the maximum similarity score
    max_similarity = max([match['similarity'] for match in matches]) if matches else 0
    
    # Check if the maximum similarity exceeds the threshold
    is_relevant = max_similarity >= relevance_threshold
    
    return is_relevant, max_similarity