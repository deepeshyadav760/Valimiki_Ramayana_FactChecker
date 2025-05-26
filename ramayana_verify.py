# ramayana_verify.py

import csv
import os
import traceback
import re
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Hardcoded file paths
CSV_FILE_PATH = "statements.csv"
OUTPUT_CSV_PATH = "output_verified.csv"
COLUMN_NAME = "Statement"

def clean_text_for_verification(text):
    """
    Clean and normalize text for verification.
    Simple preprocessing: lowercase + remove special chars + clean whitespace
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

def debug_embeddings_structure(df_with_embeddings):
    """
    Debug function to understand the structure of the embeddings DataFrame.
    """
    pass

def load_verification_resources():
    """
    Load the model and embeddings once at the beginning.
    """
    try:
        from embeddings import load_embedding_model, load_embeddings
        
        print("Loading embedding model and data...")
        
        model_name = "all-MiniLM-L6-v2"
        embeddings_path = os.path.join("ramayana_data", "ramayana_embeddings.pkl")

        model = load_embedding_model(model_name)

        if os.path.exists(embeddings_path):
            df_with_embeddings = load_embeddings(embeddings_path)
            debug_embeddings_structure(df_with_embeddings)
        else:
            print("Error: Embeddings file not found. Please create embeddings first.")
            return None, None
        
        print("Model and embeddings loaded successfully!")
        return model, df_with_embeddings
        
    except Exception as e:
        print(f"Error loading verification resources: {e}")
        traceback.print_exc()
        return None, None

def detect_modern_irrelevant_content(statement):
    """
    Detect if statement contains clearly modern concepts that wouldn't be in ancient texts
    """
    statement_lower = statement.lower()
    
    # Only the most obvious modern terms that clearly don't belong in ancient texts
    modern_terms = [
        'computer', 'internet', 'smartphone', 'mobile', 'email', 'website',
        'television', 'radio', 'movie', 'airplane', 'car', 'train',
        'dollar', 'euro', 'america', 'europe', 'china', 'japan'
    ]
    
    modern_count = sum(1 for term in modern_terms if term in statement_lower)
    return modern_count >= 1

def is_statement_too_vague(statement):
    """
    Check if statement is too vague to be verifiable
    """
    statement_lower = statement.lower().strip()
    
    if len(statement_lower) < 10:
        return True
    
    # Very generic patterns
    vague_patterns = [
        r'^(this|that|it)\s+(is|was)',
        r'^(there|here)\s+(is|are|was|were)',
        r'i love you',
        r'hello',
        r'how are you'
    ]
    
    return any(re.search(pattern, statement_lower) for pattern in vague_patterns)

def get_top_similar_verses(statement, model, df_with_embeddings, top_k=10):
    """
    Get the most similar verses to the statement using embeddings
    """
    # Get statement embedding
    statement_embedding = model.encode([statement])
    
    # Find embeddings column
    embeddings_col = None
    possible_names = ['embeddings', 'embedding', 'vectors', 'features']
    
    for col_name in possible_names:
        if col_name in df_with_embeddings.columns:
            embeddings_col = col_name
            break
    
    if embeddings_col is None:
        return [], 0.0
    
    # Get embeddings matrix
    embeddings_data = df_with_embeddings[embeddings_col].values
    if isinstance(embeddings_data[0], np.ndarray):
        embeddings_matrix = np.stack(embeddings_data)
    else:
        try:
            embeddings_matrix = np.array(embeddings_data.tolist())
        except:
            return [], 0.0
    
    # Calculate similarities
    similarities = cosine_similarity(statement_embedding, embeddings_matrix)[0]
    max_similarity = np.max(similarities)
    
    # Get top verses
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    top_verses = []
    
    for idx in top_indices:
        verse_data = df_with_embeddings.iloc[idx]
        top_verses.append({
            'text': verse_data.get('english_text', ''),
            'similarity': similarities[idx]
        })
    
    return top_verses, max_similarity

def analyze_semantic_consistency(statement, top_verses):
    """
    Analyze if the statement is semantically consistent with the verses
    Uses only semantic analysis, no hardcoded patterns
    """
    statement_words = set(clean_text_for_verification(statement).split())
    
    consistency_scores = []
    
    for verse in top_verses:
        if verse['similarity'] < 0.3:  # Skip very low similarity verses
            continue
            
        verse_text = clean_text_for_verification(verse['text'])
        verse_words = set(verse_text.split())
        
        # Calculate word overlap
        common_words = statement_words.intersection(verse_words)
        overlap_ratio = len(common_words) / len(statement_words) if statement_words else 0
        
        # Look for potential contradictions using negation patterns
        statement_lower = statement.lower()
        verse_lower = verse['text'].lower()
        
        # Simple negation detection
        statement_has_not = any(neg in statement_lower for neg in [' not ', ' never ', ' no ', "n't"])
        verse_has_not = any(neg in verse_lower for neg in [' not ', ' never ', ' no ', "n't"])
        
        # If one has negation and other doesn't, check if they're talking about same thing
        negation_conflict = False
        if statement_has_not != verse_has_not and len(common_words) >= 2:
            negation_conflict = True
        
        # Combine similarity, overlap, and negation analysis
        base_score = verse['similarity'] * 0.7 + overlap_ratio * 0.3
        
        if negation_conflict:
            base_score *= 0.5  # Reduce score for potential contradiction
        
        consistency_scores.append(base_score)
    
    return consistency_scores

def verify_ramayana_statement_generalized(statement, model, df_with_embeddings, verbose=False):
    """
    Generalized verification that doesn't rely on hardcoded patterns
    """
    try:
        # Check for obviously irrelevant content
        if detect_modern_irrelevant_content(statement):
            return None
        
        if is_statement_too_vague(statement):
            return None
        
        # Get most similar verses
        top_verses, max_similarity = get_top_similar_verses(statement, model, df_with_embeddings, top_k=15)
        
        if verbose:
            print(f"Max similarity: {max_similarity:.3f}")
        
        # If very low similarity, statement is not about this text
        if max_similarity < 0.25:
            return None
        
        # Analyze semantic consistency
        consistency_scores = analyze_semantic_consistency(statement, top_verses)
        
        if not consistency_scores:
            return None
        
        # Calculate overall scores
        avg_consistency = np.mean(consistency_scores)
        max_consistency = np.max(consistency_scores)
        high_consistency_count = sum(1 for score in consistency_scores if score > 0.6)
        
        if verbose:
            print(f"Avg consistency: {avg_consistency:.3f}, Max consistency: {max_consistency:.3f}")
            print(f"High consistency count: {high_consistency_count}")
        
        # Decision logic based purely on semantic analysis
        
        # Very high consistency - likely true
        if max_consistency >= 0.75 and avg_consistency >= 0.6:
            return True
        
        # High consistency with multiple supporting verses
        if max_consistency >= 0.65 and high_consistency_count >= 2:
            return True
        
        # Good consistency and similarity
        if max_consistency >= 0.6 and max_similarity >= 0.5:
            return True
        
        # Moderate consistency but very high similarity
        if max_similarity >= 0.7 and max_consistency >= 0.5:
            return True
        
        # Medium range - could go either way, use conservative threshold
        if max_consistency >= 0.55 and max_similarity >= 0.45:
            return True
        
        # Lower consistency or similarity suggests false or irrelevant
        if max_consistency < 0.4 or avg_consistency < 0.35:
            if max_similarity >= 0.3:  # Has some topical relevance
                return False
            else:
                return None
        
        # Default for borderline cases
        return False
        
    except Exception as e:
        if verbose:
            print(f"Error in verification: {e}")
        return None

def verify_statement(original_statement, model, df_with_embeddings):
    """
    Main verification function using generalized approach
    """
    try:
        result = verify_ramayana_statement_generalized(
            original_statement,
            model=model,
            df_with_embeddings=df_with_embeddings,
            verbose=False
        )

        if result is None:
            return "None"
        elif result is True:
            return "True"
        else:
            return "False"
            
    except Exception as e:
        print(f"Error verifying statement: {e}")
        traceback.print_exc()
        return "Error"

def verify_from_csv():
    results = []
    
    # Load model and embeddings ONCE at the beginning
    model, df_with_embeddings = load_verification_resources()
    
    if model is None or df_with_embeddings is None:
        print("Failed to load verification resources. Exiting.")
        return
    
    try:
        print(f"Looking for input file: {CSV_FILE_PATH}")
        if not os.path.exists(CSV_FILE_PATH):
            print(f"Error: File '{CSV_FILE_PATH}' not found in current directory.")
            print(f"Current directory: {os.getcwd()}")
            print(f"Files in current directory: {os.listdir('.')}")
            return

        print(f"Reading from {CSV_FILE_PATH}...")
        
        try:
            with open(CSV_FILE_PATH, 'r', encoding='utf-8') as csvfile:
                lines = csvfile.readlines()
                
                if len(lines) < 2:
                    print("Error: File has no data rows")
                    return
                
                statement_id = 1
                for line in lines[1:]:  # Skip header
                    line = line.strip()
                    if line:
                        raw_statement = line
                        
                        print(f"Processing statement {statement_id}: {raw_statement[:60]}...")
                        
                        truth_value = verify_statement(raw_statement, model, df_with_embeddings)
                        
                        result_entry = {
                            'ID': statement_id,
                            'Statement': raw_statement,
                            'Truth': truth_value
                        }
                        results.append(result_entry)
                        
                        print(f"Verification Result: {truth_value}")
                        statement_id += 1
                        
        except Exception as e:
            print(f"Error reading file: {e}")
            traceback.print_exc()
            return

        print(f"Finished processing. Total results collected: {len(results)}")

        if results:
            print(f"Writing results to {OUTPUT_CSV_PATH}...")
            try:
                with open(OUTPUT_CSV_PATH, 'w', newline='', encoding='utf-8') as output_file:
                    fieldnames = ['ID', 'Statement', 'Truth']
                    writer = csv.DictWriter(output_file, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
                    
                    writer.writeheader()
                    writer.writerows(results)
                
                print(f"✓ Results successfully saved to '{OUTPUT_CSV_PATH}'")
                print(f"Total statements processed: {len(results)}")
                
                # Print summary
                true_count = sum(1 for r in results if r['Truth'] == 'True')
                false_count = sum(1 for r in results if r['Truth'] == 'False')
                none_count = sum(1 for r in results if r['Truth'] == 'None')
                error_count = sum(1 for r in results if r['Truth'] == 'Error')
                
                print(f"Verification Summary:")
                print(f"  True: {true_count}")
                print(f"  False: {false_count}")
                print(f"  None: {none_count}")
                if error_count > 0:
                    print(f"  Errors: {error_count}")
                    
            except Exception as e:
                print(f"Error writing to output file: {e}")
                traceback.print_exc()
        else:
            print("No valid statements found to process.")
            
    except FileNotFoundError:
        print(f"Error: File '{CSV_FILE_PATH}' not found.")
    except Exception as e:
        print(f"Unexpected error: {e}")
        traceback.print_exc()

def main():
    print(f"=== Ramayana Statement Verification ===")
    print(f"Starting verification...")
    print(f"{'='*40}")
    
    verify_from_csv()

if __name__ == "__main__":
    main()
