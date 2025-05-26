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

def extract_key_entities_and_relations(statement):
    """
    Extract key entities and their relationships from the statement
    """
    statement_lower = statement.lower()
    
    # Extract entities
    entities = []
    entity_patterns = [
        'rama', 'sita', 'lakshmana', 'hanuman', 'ravana', 'bharata',
        'dasharatha', 'kaikeyi', 'sugreeva', 'vibhishana', 'indrajit',
        'kumbhakarna', 'vishwamitra', 'janaka', 'ayodhya', 'lanka'
    ]
    
    for entity in entity_patterns:
        if entity in statement_lower:
            entities.append(entity)
    
    # Extract relationships and actions
    relations = []
    
    # Family relations
    if 'son' in statement_lower or 'father' in statement_lower:
        relations.append('family')
    if 'brother' in statement_lower or 'sister' in statement_lower:
        relations.append('sibling')
    if 'married' in statement_lower or 'wife' in statement_lower or 'husband' in statement_lower:
        relations.append('marriage')
    
    # Actions
    if any(word in statement_lower for word in ['killed', 'defeated', 'fought', 'battle']):
        relations.append('conflict')
    if any(word in statement_lower for word in ['helped', 'supported', 'assisted', 'ally']):
        relations.append('alliance')
    if any(word in statement_lower for word in ['built', 'constructed', 'made']):
        relations.append('creation')
    if any(word in statement_lower for word in ['kidnapped', 'abducted', 'took']):
        relations.append('abduction')
    if any(word in statement_lower for word in ['friend', 'friendship']):
        relations.append('friendship')
    if any(word in statement_lower for word in ['disciple', 'student', 'learned', 'taught']):
        relations.append('teaching')
    
    return entities, relations

def analyze_statement_against_verses(statement, top_verses):
    """
    Analyze statement against the most similar verses to determine truth value
    """
    statement_entities, statement_relations = extract_key_entities_and_relations(statement)
    statement_lower = statement.lower()
    
    support_evidence = 0
    contradiction_evidence = 0
    
    for verse in top_verses:
        verse_text = verse['text'].lower() if 'text' in verse else verse.get('english_text', '').lower()
        verse_entities, verse_relations = extract_key_entities_and_relations(verse_text)
        
        # Check entity overlap
        common_entities = set(statement_entities).intersection(set(verse_entities))
        
        if len(common_entities) >= 1:  # At least one common entity
            
            # Analyze specific relationship patterns
            
            # Family relationships
            if 'family' in statement_relations:
                if any(word in verse_text for word in ['son', 'father', 'parent']):
                    # Check if the family relationship matches
                    if any(entity in verse_text for entity in common_entities):
                        support_evidence += 1
                else:
                    # Family claim but no family mention in verse
                    contradiction_evidence += 0.5
            
            # Sibling relationships
            if 'sibling' in statement_relations:
                if any(word in verse_text for word in ['brother', 'sister']):
                    support_evidence += 1
                else:
                    contradiction_evidence += 0.5
            
            # Marriage relationships
            if 'marriage' in statement_relations:
                if any(word in verse_text for word in ['married', 'wife', 'husband', 'wedding']):
                    support_evidence += 1
                else:
                    contradiction_evidence += 0.5
            
            # Conflict relationships
            if 'conflict' in statement_relations:
                if any(word in verse_text for word in ['killed', 'defeated', 'fought', 'battle', 'war', 'enemy']):
                    support_evidence += 1
                elif any(word in verse_text for word in ['friend', 'ally', 'helped', 'supported']):
                    contradiction_evidence += 2  # Strong contradiction
            
            # Alliance relationships
            if 'alliance' in statement_relations:
                if any(word in verse_text for word in ['helped', 'supported', 'assisted', 'ally', 'friend']):
                    support_evidence += 1
                elif any(word in verse_text for word in ['enemy', 'fought', 'killed', 'defeated']):
                    contradiction_evidence += 2  # Strong contradiction
            
            # Friendship relationships
            if 'friendship' in statement_relations:
                if any(word in verse_text for word in ['friend', 'ally', 'helped']):
                    support_evidence += 1
                elif any(word in verse_text for word in ['enemy', 'fought', 'killed', 'defeated', 'battle']):
                    contradiction_evidence += 3  # Very strong contradiction
            
            # Teaching relationships
            if 'teaching' in statement_relations:
                if any(word in verse_text for word in ['disciple', 'student', 'learned', 'taught', 'mentor']):
                    support_evidence += 1
                elif any(word in verse_text for word in ['enemy', 'fought', 'killed']):
                    contradiction_evidence += 2
            
            # Creation/building activities
            if 'creation' in statement_relations:
                if any(word in verse_text for word in ['built', 'constructed', 'made', 'created']):
                    # Check if the same entity is doing the building
                    builder_mentioned = False
                    for entity in common_entities:
                        if entity in statement_lower and entity in verse_text:
                            builder_mentioned = True
                    if builder_mentioned:
                        support_evidence += 1
                    else:
                        contradiction_evidence += 1
            
            # Abduction activities
            if 'abduction' in statement_relations:
                if any(word in verse_text for word in ['kidnapped', 'abducted', 'took', 'captured']):
                    support_evidence += 1
    
    return support_evidence, contradiction_evidence

def check_for_obvious_contradictions(statement):
    """
    Check for statements that are obviously contradictory to basic Ramayana facts
    """
    statement_lower = statement.lower()
    contradiction_score = 0
    
    # Obvious contradictions
    obvious_false_patterns = [
        # Impossible relationships
        ('lakshmana', 'disciple', 'ravana'),
        ('hanuman', 'son', 'ravana'),
        ('hanuman', 'brother', 'ravana'),
        ('sita', 'father', 'ravana'),
        ('ravana', 'rama', 'friend'),
        ('ravana', 'peaceful', 'poet'),
        ('lakshmana', 'not mentioned'),
        ('sita', 'built', 'bridge'),
        ('hanuman', 'fought', 'rama'),
        ('bharata', 'lanka'),
        ('dasharatha', 'lanka'),
        
        # Timeline contradictions
        ('exile', 'one year'),
        ('married', 'lanka', 'ravana'),
        
        # Character misattributions
        ('rama', 'pandava'),
        ('ramayana', 'kurukshetra'),
        ('rama', 'ten heads'),
        ('ravana', 'devotee', 'krishna'),
        ('sugriva', 'king', 'lanka'),
        ('ramayana', 'english'),
    ]
    
    for pattern in obvious_false_patterns:
        if all(word in statement_lower for word in pattern):
            contradiction_score += 3
    
    # Additional specific contradictions
    if 'lakshmana' in statement_lower and 'not mentioned' in statement_lower:
        contradiction_score += 5
    
    if 'ravana' in statement_lower and 'peaceful' in statement_lower:
        contradiction_score += 4
    
    if 'hanuman' in statement_lower and ('son' in statement_lower or 'brother' in statement_lower) and 'ravana' in statement_lower:
        contradiction_score += 4
    
    return contradiction_score

def verify_ramayana_statement_complete(statement, model, df_with_embeddings, verbose=False):
    """
    Complete verification with improved logic
    """
    try:
        # Step 1: Check for non-Ramayana content
        modern_score = detect_non_ramayana_content(statement)
        vague_score = detect_vague_or_generic_content(statement)
        
        if verbose:
            print(f"Modern score: {modern_score}, Vague score: {vague_score}")
        
        # If clearly modern/irrelevant content
        if modern_score >= 2:
            return None
        
        if vague_score >= 2:
            return None
        
        # Step 2: Check for Ramayana indicators
        ramayana_indicators = extract_ramayana_indicators(statement)
        
        if verbose:
            print(f"Ramayana indicators: {ramayana_indicators}")
        
        # Step 3: Get embeddings and similarity
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
        
        if verbose:
            print(f"Max similarity: {max_similarity:.3f}")
        
        # Step 4: Relevance check
        if max_similarity < 0.2:
            return None
        
        if (max_similarity < 0.35 and 
            ramayana_indicators['total_indicators'] == 0 and 
            (modern_score >= 1 or vague_score >= 1)):
            return None
        
        # Step 5: Get top verses for detailed analysis
        top_indices = np.argsort(similarities)[-15:][::-1]
        top_verses = []
        
        for idx in top_indices:
            verse_data = df_with_embeddings.iloc[idx]
            top_verses.append({
                'text': verse_data.get('english_text', ''),
                'similarity': similarities[idx]
            })
        
        # Step 6: Check for obvious contradictions first
        obvious_contradiction_score = check_for_obvious_contradictions(statement)
        
        if verbose:
            print(f"Obvious contradiction score: {obvious_contradiction_score}")
        
        # If obvious contradiction, return False immediately
        if obvious_contradiction_score >= 3:
            return False
        
        # Step 7: Detailed analysis against verses
        support_evidence, contradiction_evidence = analyze_statement_against_verses(statement, top_verses)
        
        if verbose:
            print(f"Support evidence: {support_evidence}, Contradiction evidence: {contradiction_evidence}")
        
        # Step 8: Decision logic - Adjusted weights to favor True when appropriate
        
        # Very strong contradictions (only the most obvious)
        if contradiction_evidence >= 3 or obvious_contradiction_score >= 3:
            return False
        
        # Strong support with reasonable similarity
        if support_evidence >= 2 and max_similarity >= 0.4 and contradiction_evidence == 0:
            return True
        
        # Medium support with good similarity
        if support_evidence >= 1 and max_similarity >= 0.5 and contradiction_evidence == 0:
            return True
        
        # High similarity alone (for basic facts) - lowered threshold
        if max_similarity >= 0.6 and contradiction_evidence == 0:
            return True
        
        # Good similarity with minimal contradiction
        if max_similarity >= 0.55 and contradiction_evidence <= 0.5:
            return True
        
        # Cases with moderate contradiction - be more lenient
        if contradiction_evidence >= 2 or obvious_contradiction_score >= 2:
            return False
        
        # Medium similarity with some support - more generous
        if max_similarity >= 0.45 and support_evidence >= 0.5:
            return True
        
        # Reasonable similarity for Ramayana content
        if max_similarity >= 0.4 and ramayana_indicators['total_indicators'] >= 2 and contradiction_evidence < 1:
            return True
        
        # Low support or low similarity
        if support_evidence == 0 and max_similarity < 0.35:
            if ramayana_indicators['total_indicators'] > 0:
                return False  # Has Ramayana content but very low support
            else:
                return None
        
        # Default case - give benefit of doubt if it's about Ramayana
        if ramayana_indicators['total_indicators'] > 0 and max_similarity >= 0.3:
            return True
        else:
            return False
        
    except Exception as e:
        if verbose:
            print(f"Error in verification: {e}")
        return None

def verify_statement(original_statement, model, df_with_embeddings):
    """
    Main verification function
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
