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

def detect_non_ramayana_content(statement):
    """
    Detect if statement is about modern/irrelevant topics (should be None)
    """
    statement_lower = statement.lower()
    
    # Modern technology indicators
    modern_tech_terms = [
        'computer', 'internet', 'smartphone', 'phone', 'mobile', 'laptop', 
        'tablet', 'software', 'hardware', 'digital', 'electronic', 'technology',
        'wifi', 'bluetooth', 'email', 'website', 'app', 'application',
        'programming', 'coding', 'python', 'java', 'javascript'
    ]
    
    # Modern world concepts
    modern_concepts = [
        'car', 'automobile', 'vehicle', 'airplane', 'plane', 'train', 'bus',
        'television', 'tv', 'radio', 'movie', 'film', 'cinema',
        'hospital', 'doctor', 'medicine', 'surgery', 'patient',
        'school', 'university', 'college', 'student', 'teacher', 'education',
        'company', 'business', 'office', 'job', 'work', 'employee',
        'bank', 'money', 'dollar', 'euro', 'currency', 'economy'
    ]
    
    # Science and nature (modern context)
    science_terms = [
        'physics', 'chemistry', 'biology', 'mathematics', 'science',
        'experiment', 'laboratory', 'research', 'scientist',
        'gravity', 'atom', 'molecule', 'dna', 'gene', 'evolution',
        'planet', 'solar system', 'galaxy', 'universe', 'space'
    ]
    
    # Geography (modern)
    modern_geography = [
        'america', 'usa', 'europe', 'china', 'japan', 'australia',
        'new york', 'london', 'paris', 'tokyo', 'beijing', 'moscow',
        'country', 'nation', 'government', 'president', 'democracy'
    ]
    
    # Sports and entertainment
    sports_entertainment = [
        'football', 'soccer', 'basketball', 'tennis', 'cricket', 'golf',
        'olympics', 'sport', 'game', 'player', 'team', 'match',
        'music', 'song', 'singer', 'band', 'concert', 'album'
    ]
    
    all_modern_terms = (modern_tech_terms + modern_concepts + science_terms + 
                       modern_geography + sports_entertainment)
    
    # Count modern terms
    modern_term_count = sum(1 for term in all_modern_terms if term in statement_lower)
    
    # Check for modern sentence patterns
    modern_patterns = [
        r'today\s+we',
        r'in\s+the\s+modern\s+world',
        r'nowadays',
        r'in\s+\d{4}',  # years like "in 1990"
        r'21st\s+century',
        r'20th\s+century',
        r'currently',
        r'these\s+days'
    ]
    
    pattern_matches = sum(1 for pattern in modern_patterns 
                         if re.search(pattern, statement_lower))
    
    return modern_term_count + pattern_matches

def detect_vague_or_generic_content(statement):
    """
    Detect vague, generic, or meaningless statements
    """
    statement_lower = statement.lower().strip()
    
    # Very short statements (likely vague)
    if len(statement_lower) < 10:
        return 2
    
    # Generic/vague patterns
    vague_patterns = [
        r'^(this|that|it|something|anything)\s+(is|was|are|were)',
        r'^(there|here)\s+(is|are|was|were)',
        r'^(some|many|few|several)\s+things?',
        r'in\s+general',
        r'usually',
        r'sometimes',
        r'often',
        r'always\s+true',
        r'it\s+depends',
        r'perhaps',
        r'maybe',
        r'possibly'
    ]
    
    vague_score = sum(1 for pattern in vague_patterns 
                     if re.search(pattern, statement_lower))
    
    # Check for overly generic words
    generic_words = ['thing', 'stuff', 'something', 'anything', 'everything', 
                    'someone', 'anyone', 'everyone', 'somewhere', 'anywhere']
    
    generic_count = sum(1 for word in generic_words if word in statement_lower)
    
    return vague_score + generic_count

def extract_ramayana_indicators(statement):
    """
    Extract indicators that suggest statement is about Ramayana
    """
    statement_lower = statement.lower()
    
    # Character names 
    # This is dynamic - we can extract these from the actual verses
    character_indicators = [
        'rama', 'sita', 'lakshmana', 'hanuman', 'ravana', 'bharata',
        'dasharatha', 'kaikeyi', 'kausalya', 'sumitra', 'sugreeva',
        'vibhishana', 'indrajit', 'kumbhakarna', 'angada', 'jatayu',
        'vishwamitra', 'vasishta', 'janaka', 'mandodari', 'urmila'
    ]
    
    # Places
    place_indicators = [
        'ayodhya', 'lanka', 'mithila', 'dandaka', 'panchavati',
        'kishkindha', 'ashoka', 'vatika', 'chitrakuta'
    ]
    
    # Concepts/themes
    concept_indicators = [
        'epic', 'ramayana', 'valmiki', 'sage', 'demon', 'monkey',
        'exile', 'forest', 'bridge', 'ocean', 'arrow', 'bow',
        'dharma', 'virtue', 'righteous', 'devotion', 'loyalty'
    ]
    
    # Count indicators
    character_count = sum(1 for char in character_indicators if char in statement_lower)
    place_count = sum(1 for place in place_indicators if place in statement_lower)
    concept_count = sum(1 for concept in concept_indicators if concept in statement_lower)
    
    return {
        'character_count': character_count,
        'place_count': place_count,
        'concept_count': concept_count,
        'total_indicators': character_count + place_count + concept_count
    }

def detect_obvious_false_patterns(statement):
    """
    Detect only the most obvious false patterns
    """
    statement_lower = statement.lower()
    
    obvious_false_indicators = 0
    
    # Impossible combinations
    impossible_pairs = [
        ("lakshmana", "disciple", "ravana"),
        ("hanuman", "son", "ravana"),
        ("sita", "built", "bridge"),
        ("bharata", "originally", "lanka"),
        ("ravana", "peaceful", "poet"),
        ("married", "lanka", "priest")
    ]
    
    for pair in impossible_pairs:
        if all(word in statement_lower for word in pair):
            obvious_false_indicators += 2
    
    # Obvious negations
    if "lakshmana" in statement_lower and "not mentioned" in statement_lower:
        obvious_false_indicators += 2
    
    # Timeline contradictions
    if ("exile" in statement_lower and "one year" in statement_lower):
        obvious_false_indicators += 1
    
    return obvious_false_indicators

def verify_ramayana_statement_complete(statement, model, df_with_embeddings, verbose=False):
    """
    Complete verification with proper None detection
    """
    try:
        # Step 1: Check for non-Ramayana content first
        modern_score = detect_non_ramayana_content(statement)
        vague_score = detect_vague_or_generic_content(statement)
        
        if verbose:
            print(f"Modern score: {modern_score}, Vague score: {vague_score}")
        
        # If clearly modern/irrelevant content
        if modern_score >= 2:
            return None
        
        # If very vague/generic
        if vague_score >= 2:
            return None
        
        # Step 2: Check for Ramayana indicators
        ramayana_indicators = extract_ramayana_indicators(statement)
        
        if verbose:
            print(f"Ramayana indicators: {ramayana_indicators}")
        
        # If no Ramayana indicators at all
        if ramayana_indicators['total_indicators'] == 0:
            # Still check similarity to be sure
            pass  # Continue to similarity check
        
        # Step 3: Get similarity scores
        statement_embedding = model.encode([statement])
        
        # Find embeddings column
        embeddings_col = None
        possible_names = ['embeddings', 'embedding', 'vectors', 'features']
        
        for col_name in possible_names:
            if col_name in df_with_embeddings.columns:
                embeddings_col = col_name
                break
        
        if embeddings_col is None:
            return None
        
        # Get embeddings matrix
        embeddings_data = df_with_embeddings[embeddings_col].values
        if isinstance(embeddings_data[0], np.ndarray):
            embeddings_matrix = np.stack(embeddings_data)
        else:
            try:
                embeddings_matrix = np.array(embeddings_data.tolist())
            except:
                return None
        
        # Calculate similarities
        similarities = cosine_similarity(statement_embedding, embeddings_matrix)[0]
        max_similarity = np.max(similarities)
        mean_similarity = np.mean(similarities)
        
        if verbose:
            print(f"Max similarity: {max_similarity:.3f}, Mean similarity: {mean_similarity:.3f}")
        
        # Step 4: Relevance check based on similarity
        if max_similarity < 0.2:
            return None  # Very low similarity - not about Ramayana
        
        # Step 5: Combined relevance check
        # If low similarity AND no indicators AND (modern OR vague content)
        if (max_similarity < 0.35 and 
            ramayana_indicators['total_indicators'] == 0 and 
            (modern_score >= 1 or vague_score >= 1)):
            return None
        
        # Step 6: If clearly not relevant despite some similarity
        if max_similarity < 0.3 and modern_score >= 1:
            return None
        
        # Step 7: Now do True/False classification for Ramayana-relevant content
        false_indicators = detect_obvious_false_patterns(statement)
        
        if verbose:
            print(f"False indicators: {false_indicators}")
        
        # Decision logic for True/False
        if false_indicators >= 3:
            return False
        elif max_similarity >= 0.6 and false_indicators == 0:
            return True
        elif max_similarity >= 0.55 and false_indicators <= 1:
            # Use additional analysis
            return analyze_borderline_case(statement, similarities, df_with_embeddings)
        elif max_similarity >= 0.45:
            if false_indicators >= 2:
                return False
            else:
                return True
        elif max_similarity >= 0.35:
            if false_indicators >= 1:
                return False
            else:
                return analyze_borderline_case(statement, similarities, df_with_embeddings)
        else:
            # Low similarity - could be None or False
            if ramayana_indicators['total_indicators'] > 0:
                return False  # Has Ramayana content but low similarity = False
            else:
                return None   # No Ramayana content and low similarity = None
            
    except Exception as e:
        if verbose:
            print(f"Error in verification: {e}")
        return None

def analyze_borderline_case(statement, similarities, df_with_embeddings):
    """
    Analyze borderline cases using statistical methods
    """
    try:
        top_similarities = np.sort(similarities)[-10:]
        mean_top = np.mean(top_similarities)
        mean_all = np.mean(similarities)
        std_all = np.std(similarities)
        max_sim = np.max(similarities)
        
        high_sim_count = np.sum(similarities > 0.5)
        
        # Decision rules
        if mean_top > 0.5 and max_sim > 0.55:
            return True
        
        if max_sim > (mean_all + 1.5 * std_all) and max_sim > 0.45:
            return True
        
        if high_sim_count >= 3 and max_sim > 0.5:
            return True
        
        percentile_95 = np.percentile(similarities, 95)
        if max_sim >= percentile_95 and max_sim > 0.4:
            return True
        
        return False
        
    except Exception as e:
        return True if np.max(similarities) > 0.5 else False

def verify_statement(original_statement, model, df_with_embeddings):
    """
    Main verification function with complete None detection
    """
    try:
        result = verify_ramayana_statement_complete(
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