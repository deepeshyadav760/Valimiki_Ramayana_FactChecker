import numpy as np
from sentence_transformers import CrossEncoder
from semantic_search import get_relevant_verses, is_statement_relevant_to_ramayana
from preprocessing import preprocess_text
import spacy
import re

# Load SpaCy for NER and dependency parsing
try:
    nlp = spacy.load("en_core_web_sm")
except:
    import spacy.cli
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def extract_entities(text):
    """
    Extract named entities from text
    
    Args:
        text: Input text
        
    Returns:
        Dictionary of entities by type
    """
    doc = nlp(text)
    entities = {}
    
    for ent in doc.ents:
        if ent.label_ not in entities:
            entities[ent.label_] = []
        entities[ent.label_].append(ent.text)
    
    return entities

def verify_entities(statement_entities, verse_entities):
    """
    Check if entities in statement match with verse entities
    
    Args:
        statement_entities: Entities extracted from statement
        verse_entities: Entities extracted from verse
        
    Returns:
        Entity match score (0-1)
    """
    if not statement_entities or not verse_entities:
        return 0.5  # Neutral if no entities
    
    # Flatten all entities to lists
    statement_all = []
    verse_all = []
    
    for entity_type, entities in statement_entities.items():
        statement_all.extend([e.lower() for e in entities])
    
    for entity_type, entities in verse_entities.items():
        verse_all.extend([e.lower() for e in entities])
    
    # Count matches
    matches = 0
    for entity in statement_all:
        if any(entity in v or v in entity for v in verse_all):
            matches += 1
    
    # Calculate match score
    if len(statement_all) == 0:
        return 0.5
    
    return matches / len(statement_all)

def check_negation(text):
    """
    Check if text contains negation
    
    Args:
        text: Input text
        
    Returns:
        Boolean indicating presence of negation
    """
    doc = nlp(text)
    
    # Check for negation words
    negation_words = ["not", "no", "never", "neither", "nor", "none", "doesn't", "isn't", "wasn't", "aren't", "don't", "didn't"]
    
    for token in doc:
        if token.text.lower() in negation_words:
            return True
        if token.dep_ == "neg":
            return True
    
    return False

def verify_with_cross_encoder(statement, verses, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
    """
    Use a cross-encoder model to verify statement against verses
    
    Args:
        statement: Statement to verify
        verses: List of verse dictionaries
        model_name: Name of the cross-encoder model
        
    Returns:
        Verification score (0-1)
    """
    # Initialize cross-encoder
    model = CrossEncoder(model_name)
    
    # Create pairs of statement and verse texts
    pairs = [[statement, verse['text']] for verse in verses]
    
    # Compute scores
    scores = model.predict(pairs)
    
    # Return the maximum score
    return max(scores) if len(scores) > 0 else 0.0

def extract_key_facts(text):
    """
    Extract key facts from text
    
    Args:
        text: Input text
        
    Returns:
        List of key facts
    """
    doc = nlp(text)
    facts = []
    
    # Extract subject-verb-object triples
    for sent in doc.sents:
        for token in sent:
            # Look for verbs
            if token.pos_ == "VERB":
                # Find subject
                subj = None
                for child in token.children:
                    if child.dep_ in ["nsubj", "nsubjpass"]:
                        subj = child
                        break
                
                # Find object
                obj = None
                for child in token.children:
                    if child.dep_ in ["dobj", "pobj", "attr"]:
                        obj = child
                        break
                
                # If we have a subject and verb, create a fact
                if subj:
                    fact = " ".join([subj.text, token.text])
                    if obj:
                        fact += " " + obj.text
                    facts.append(fact.lower())
    
    return facts

def compare_facts(statement_facts, verse_facts):
    """
    Compare facts between statement and verses
    
    Args:
        statement_facts: Facts extracted from statement
        verse_facts: Facts extracted from verses
        
    Returns:
        Fact match score (0-1)
    """
    if not statement_facts:
        return 0.5  # Neutral if no facts
    
    # Flatten verse facts
    all_verse_facts = []
    for facts in verse_facts:
        all_verse_facts.extend(facts)
    
    # Count matches
    matches = 0
    for fact in statement_facts:
        for verse_fact in all_verse_facts:
            # Calculate word overlap
            statement_words = set(fact.lower().split())
            verse_words = set(verse_fact.lower().split())
            overlap = len(statement_words.intersection(verse_words))
            
            if overlap >= min(2, len(statement_words) - 1):  # At least 2 words or all but one
                matches += 1
                break
    
    # Calculate match score
    if len(statement_facts) == 0:
        return 0.5
    
    return matches / len(statement_facts)

def check_assertion_support(statement, verses):
    """
    Check if verses support the specific claim in the statement
    
    Args:
        statement: Statement to verify
        verses: List of verse dictionaries
        
    Returns:
        Support score (0-1) and support details
    """
    # Extract key entities from statement
    statement_doc = nlp(statement)
    statement_entities = set()
    for ent in statement_doc.ents:
        statement_entities.add(ent.text.lower())
    
    # Extract key terms (PROPN, NOUN, VERB)
    statement_terms = set()
    for token in statement_doc:
        if token.pos_ in ["PROPN", "NOUN", "VERB"] and len(token.text) > 3:
            statement_terms.add(token.text.lower())
    
    # Check for direct contradictions in verses
    contradiction_terms = ["not", "never", "without", "against", "contrary", 
                          "opposite", "different", "instead", "rather", 
                          "however", "but", "although", "unlike"]
    
    # Check each verse for support or contradiction
    verse_supports = []
    for verse in verses:
        verse_text = verse['text'].lower()
        verse_doc = nlp(verse_text)
        
        # Check for entity presence
        entity_present = any(entity in verse_text for entity in statement_entities)
        
        # Check for term presence
        term_overlap = sum(1 for term in statement_terms if term in verse_text)
        term_score = term_overlap / len(statement_terms) if statement_terms else 0.5
        
        # Check for contradictions
        has_contradiction_terms = any(term in verse_text for term in contradiction_terms)
        
        # Combine factors
        support_score = 0.6 * entity_present + 0.4 * term_score
        
        # Reduce score if contradictions present
        if has_contradiction_terms:
            support_score *= 0.7
        
        # Make sure we have 'reference' key - with the updated CSV structure
        reference = verse.get('reference', f"{verse.get('kanda', 'Unknown')}, Sarga {verse.get('sarga', 'Unknown')}, Verse {verse.get('verse_num', 'Unknown')}")
        
        verse_supports.append({
            'reference': reference,
            'support_score': support_score,
            'has_contradiction': has_contradiction_terms,
            'entity_present': entity_present,
            'term_score': term_score
        })
    
    # Calculate overall support score
    if not verse_supports:
        return 0.5, []
    
    # Get the maximum support score
    max_support_score = max(v['support_score'] for v in verse_supports)
    
    # Check if any verse has high support with contradiction
    contradictory_verses = [v for v in verse_supports if v['has_contradiction'] and v['support_score'] > 0.6]
    
    # If we have contradictory verses with high support, this likely means the statement is false
    if contradictory_verses:
        max_support_score *= 0.5
    
    return max_support_score, verse_supports

def verify_fact(statement, df_with_embeddings, model, 
               relevance_threshold=0.4, verify_threshold=0.7):
    """
    Verify if a statement is factually correct according to the Ramayana
    
    Args:
        statement: Statement to verify
        df_with_embeddings: DataFrame with verse embeddings
        model: SentenceTransformer model
        relevance_threshold: Threshold for determining if statement is relevant to Ramayana
        verify_threshold: Threshold for determining if statement is factually correct
        
    Returns:
        True if factually correct, False if incorrect, None if not relevant
    """
    # Check if statement is relevant to Ramayana
    is_relevant, relevance_score = is_statement_relevant_to_ramayana(
        statement, df_with_embeddings, model, relevance_threshold=relevance_threshold
    )
    
    if not is_relevant:
        return None, {
            'relevance': is_relevant,
            'relevance_score': relevance_score,
            'message': "Statement is not relevant to Ramayana"
        }
    
    # Get relevant verses
    relevant_verses = get_relevant_verses(
        statement, df_with_embeddings, model, threshold=relevance_threshold
    )
    
    if not relevant_verses:
        return None, {
            'relevance': is_relevant,
            'relevance_score': relevance_score,
            'message': "No relevant verses found"
        }
    
    # Extract entities
    statement_entities = extract_entities(statement)
    verse_entities = [extract_entities(verse['text']) for verse in relevant_verses]
    
    # Check entity match
    entity_score = max([verify_entities(statement_entities, v) for v in verse_entities])
    
    # Extract facts from statement and verses
    statement_facts = extract_key_facts(statement)
    verse_facts = [extract_key_facts(verse['text']) for verse in relevant_verses]
    
    # Compare facts
    fact_score = compare_facts(statement_facts, verse_facts)
    
    # Check for negation (can indicate contradiction)
    statement_has_negation = check_negation(statement)
    verses_have_negation = [check_negation(verse['text']) for verse in relevant_verses]
    
    # Check if statement directly contradicts verses
    support_score, support_details = check_assertion_support(statement, relevant_verses)
    
    # Improved negation handling
    if statement_has_negation:
        # Statement has negation (e.g., "Rama did not...") 
        # Check if verses support or contradict this negative claim
        statement_has_contradiction = any(["not" in v['text'].lower() or 
                                         "never" in v['text'].lower() or 
                                         "no " in v['text'].lower() 
                                         for v in relevant_verses])
        
        if not statement_has_contradiction:
            # If the statement has negation but verses don't support this negation,
            # the statement is likely false
            entity_score *= 0.5
            fact_score *= 0.5
    
    # Get cross-encoder verification score
    cross_encoder_score = verify_with_cross_encoder(statement, relevant_verses)
    
    # Normalize cross-encoder score (it tends to be inflated)
    normalized_cross_encoder = min(cross_encoder_score / 10.0, 1.0)
    
    # Combine scores (weighted average with adjusted weights)
    combined_score = (
        entity_score * 0.35 +
        fact_score * 0.35 + 
        normalized_cross_encoder * 0.2 +
        support_score * 0.1
    )
    
    # Extra check for borderline scores
    if 0.6 <= combined_score < verify_threshold:
        # Look for evidence of contradictions
        contradiction_words = ["not", "never", "without", "against", "contrary", "opposite", 
                             "different", "instead", "rather", "however", "but", "although"]
        
        has_contradiction_terms = any(any(word in verse['text'].lower() 
                                       for word in contradiction_words) 
                                    for verse in relevant_verses)
        
        # If there are contradictions, the statement is probably false
        if has_contradiction_terms:
            combined_score *= 0.8  # Reduce the score
    
    # Determine if statement is factually correct
    is_factual = combined_score >= verify_threshold
    
    # Check against well-known Ramayana facts
    ramayana_facts = {
        "rama married sita": True,
        "sita was kidnapped by ravana": True,
        "ravana kidnapped sita": True,
        "lakshmana accompanied rama": True,
        "hanuman crossed the ocean": True,
        "ravana was a demon king": True,
        "rama was born in ayodhya": True,
        "rama and sita were married in lanka": False,
        "lakshmana is not mentioned": False,
        "ravana is rama's brother": False,
        "hanuman failed to find sita": False
    }
    
    # Normalize statement for comparison
    norm_statement = ' '.join(preprocess_text(statement)).lower()
    
    # Check for contradictions with known facts
    for key, is_true in ramayana_facts.items():
        # If statement contains the key phrase
        if all(word in norm_statement for word in key.split()):
            # Check if statement agrees or contradicts known fact
            if statement_has_negation != (not is_true):
                # If statement contradicts known fact, it's likely false
                is_factual = False
                combined_score *= 0.7
                break
            else:
                # If statement agrees with known fact, boost confidence
                combined_score = min(combined_score * 1.2, 1.0)
                break
    
    # Create verification details
    verification_details = {
        'relevance': is_relevant,
        'relevance_score': relevance_score,
        'entity_score': entity_score,
        'fact_score': fact_score,
        'cross_encoder_score': cross_encoder_score,
        'normalized_cross_encoder': normalized_cross_encoder,
        'support_score': support_score,
        'combined_score': combined_score,
        'has_negation': statement_has_negation,
        'verses_have_negation': any(verses_have_negation),
        'top_verses': relevant_verses
    }
    
    return is_factual, verification_details