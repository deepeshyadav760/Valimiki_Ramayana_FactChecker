import os
import argparse
import json
from embeddings import create_and_save_embeddings, load_embedding_model
from fact_verification import verify_fact

def verify_ramayana_statement(statement, model=None, df_with_embeddings=None, 
                             data_file=os.path.join("ramayana_data", "merged_ramayana.csv"), 
                             embeddings_path=os.path.join("ramayana_data", "ramayana_embeddings.pkl"),
                             model_name="all-MiniLM-L6-v2",
                             relevance_threshold=0.5,
                             verify_threshold=0.5,
                             verbose=False):
    """
    Main function to verify if a statement about Ramayana is factually correct
    
    Args:
        statement: Statement to verify
        model: Pre-loaded SentenceTransformer model (optional)
        df_with_embeddings: Pre-loaded DataFrame with embeddings (optional)
        data_file: CSV file containing Ramayana data
        embeddings_path: Path to save/load embeddings
        model_name: Name of the SentenceTransformer model
        relevance_threshold: Threshold for determining relevance
        verify_threshold: Threshold for determining factual correctness
        verbose: Whether to print detailed verification information
        
    Returns:
        True if factually correct, False if incorrect, None if not relevant
    """
    # Load model and embeddings if not provided
    if model is None:
        model = load_embedding_model(model_name)
    
    if df_with_embeddings is None:
        # Check if embeddings exist
        if os.path.exists(embeddings_path):
            from embeddings import load_embeddings
            df_with_embeddings = load_embeddings(embeddings_path)
        else:
            print(f"Error: Embeddings file not found at {embeddings_path}")
            print("Please create embeddings first using the embeddings module.")
            return None
    
    # Verify the statement
    result, details = verify_fact(
        statement, df_with_embeddings, model,
        relevance_threshold=relevance_threshold,
        verify_threshold=verify_threshold
    )
    
    # Print details if verbose
    if verbose:
        print("\n" + "="*50)
        print(f"Statement: {statement}")
        print("-"*50)
        print(f"Relevance: {details['relevance']} (Score: {details['relevance_score']:.4f})")
        
        if result is not None:
            print(f"Entity Match Score: {details['entity_score']:.4f}")
            print(f"Fact Match Score: {details['fact_score']:.4f}")
            print(f"Cross-Encoder Score: {details['cross_encoder_score']:.4f}")
            print(f"Combined Score: {details['combined_score']:.4f}")
            print(f"Statement has negation: {details['has_negation']}")
            print("-"*50)
            print(f"Verification Result: {result}")
            
            print("\nTop matching verses:")
            for i, verse in enumerate(details['top_verses'][:3]):
                print(f"\n{i+1}. {verse['reference']} (Similarity: {verse['similarity']:.4f})")
                print(f"   {verse['text']}")
        else:
            print(f"Message: {details.get('message', 'Not relevant to Ramayana')}")
        
        print("="*50)
    
    return result

def batch_verify_statements(statements, output_file=None, **kwargs):
    """
    Verify a batch of statements and optionally save results to a file
    
    Args:
        statements: List of statements to verify
        output_file: File to save results (optional)
        **kwargs: Additional arguments for verify_ramayana_statement
        
    Returns:
        List of results
    """
    # Load resources once for efficiency
    model = load_embedding_model(kwargs.get('model_name', "all-MiniLM-L6-v2"))
    
    embeddings_path = kwargs.get('embeddings_path', os.path.join("ramayana_data", "ramayana_embeddings.pkl"))
    if os.path.exists(embeddings_path):
        from embeddings import load_embeddings
        df_with_embeddings = load_embeddings(embeddings_path)
    else:
        print(f"Error: Embeddings file not found at {embeddings_path}")
        print("Please create embeddings first using the embeddings module.")
        return []
    
    # Verify each statement
    results = []
    for i, statement in enumerate(statements):
        print(f"Verifying statement {i+1}/{len(statements)}")
        result = verify_ramayana_statement(
            statement, 
            model=model, 
            df_with_embeddings=df_with_embeddings,
            verbose=kwargs.get('verbose', False)
        )
        results.append({
            'statement': statement,
            'result': result
        })
    
    # Save results if output file provided
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_file}")
    
    return results

# Command-line interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify statements about the Ramayana")
    parser.add_argument("statement", nargs="?", help="Statement to verify")
    parser.add_argument("--input-file", "-i", help="Input file with statements (one per line)")
    parser.add_argument("--output-file", "-o", help="Output file for batch verification results")
    parser.add_argument("--data-file", default=os.path.join("ramayana_data", "merged_ramayana.csv"), 
                       help="CSV file containing Ramayana data")
    parser.add_argument("--embeddings-path", default=os.path.join("ramayana_data", "ramayana_embeddings.pkl"), 
                      help="Path to save/load embeddings")
    parser.add_argument("--model-name", default="all-MiniLM-L6-v2", 
                      help="Name of the SentenceTransformer model")
    parser.add_argument("--relevance-threshold", type=float, default=0.4, 
                      help="Threshold for determining relevance")
    parser.add_argument("--verify-threshold", type=float, default=0.65, 
                      help="Threshold for determining factual correctness")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print detailed information")
    
    args = parser.parse_args()
    
    if args.input_file:
        # Batch verification
        with open(args.input_file, 'r') as f:
            statements = [line.strip() for line in f if line.strip()]
        
        batch_verify_statements(
            statements,
            output_file=args.output_file,
            data_file=args.data_file,
            embeddings_path=args.embeddings_path,
            model_name=args.model_name,
            relevance_threshold=args.relevance_threshold,
            verify_threshold=args.verify_threshold,
            verbose=args.verbose
        )
    elif args.statement:
        # Single statement verification
        verify_ramayana_statement(
            args.statement,
            data_file=args.data_file,
            embeddings_path=args.embeddings_path,
            model_name=args.model_name,
            relevance_threshold=args.relevance_threshold,
            verify_threshold=args.verify_threshold,
            verbose=True
        )
    else:
        parser.print_help()