import json
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from main_function import batch_verify_statements, verify_ramayana_statement
import os

def evaluate_on_test_set(test_file, ground_truth_file=None, output_file=None, **kwargs):
    """
    Evaluate the model on a test set
    
    Args:
        test_file: File containing test statements
        ground_truth_file: File containing ground truth labels (optional)
        output_file: File to save results (optional)
        **kwargs: Additional arguments for batch_verify_statements
        
    Returns:
        Evaluation metrics
    """
    # Load test statements
    with open(test_file, 'r') as f:
        if test_file.endswith('.json'):
            # If test file is JSON
            test_data = json.load(f)
            statements = [item['statement'] for item in test_data]
            
            if ground_truth_file is None and 'label' in test_data[0]:
                # Extract ground truth from test data if available
                ground_truth = [item['label'] for item in test_data]
                
                # Convert text labels to boolean/None
                ground_truth = [
                    True if label == "True" or label is True else 
                    False if label == "False" or label is False else None 
                    for label in ground_truth
                ]
        else:
            # If test file is text (one statement per line)
            statements = [line.strip() for line in f if line.strip()]
            ground_truth = None
    
    # Load ground truth if provided separately
    if ground_truth_file and ground_truth is None:
        with open(ground_truth_file, 'r') as f:
            if ground_truth_file.endswith('.json'):
                ground_truth_data = json.load(f)
                ground_truth = [item['label'] for item in ground_truth_data]
            else:
                ground_truth = [line.strip() for line in f if line.strip()]
            
            # Convert text labels to boolean/None
            ground_truth = [
                True if label == "True" or label is True else 
                False if label == "False" or label is False else None 
                for label in ground_truth
            ]
    
    # Verify statements
    results = batch_verify_statements(statements, output_file=output_file, **kwargs)
    predictions = [item['result'] for item in results]
    
    # If no ground truth, just return predictions
    if ground_truth is None:
        return {
            'predictions': predictions,
            'statements': statements
        }
    
    # Calculate metrics
    metrics = calculate_metrics(ground_truth, predictions)
    
    # Create detailed results
    detailed_results = []
    for i, (statement, true_label, pred_label) in enumerate(zip(statements, ground_truth, predictions)):
        detailed_results.append({
            'id': i + 1,
            'statement': statement,
            'true_label': true_label,
            'predicted_label': pred_label,
            'correct': true_label == pred_label
        })
    
    # Save detailed results if output file is provided
    if output_file:
        with open(output_file, 'w') as f:
            json.dump({
                'metrics': metrics,
                'detailed_results': detailed_results
            }, f, indent=2)
    
    return {
        'metrics': metrics,
        'detailed_results': detailed_results
    }

def calculate_metrics(ground_truth, predictions):
    """
    Calculate evaluation metrics
    
    Args:
        ground_truth: List of true labels
        predictions: List of predicted labels
        
    Returns:
        Dictionary of metrics
    """
    # Filter out None values (not relevant to Ramayana)
    filtered_data = [(true, pred) for true, pred in zip(ground_truth, predictions) 
                    if true is not None and pred is not None]
    
    if not filtered_data:
        return {
            'accuracy': 0,
            'precision': 0,
            'recall': 0,
            'f1': 0,
            'none_accuracy': 0,
            'confusion_matrix': [[0, 0], [0, 0]]
        }
    
    filtered_true, filtered_pred = zip(*filtered_data)
    
    # Calculate metrics for True/False predictions
    accuracy = accuracy_score(filtered_true, filtered_pred)
    precision = precision_score(filtered_true, filtered_pred, zero_division=0)
    recall = recall_score(filtered_true, filtered_pred, zero_division=0)
    f1 = f1_score(filtered_true, filtered_pred, zero_division=0)
    conf_matrix = confusion_matrix(filtered_true, filtered_pred, labels=[True, False])
    
    # Calculate accuracy for None predictions
    none_indices = [i for i, label in enumerate(ground_truth) if label is None]
    none_preds = [predictions[i] for i in none_indices]
    none_accuracy = none_preds.count(None) / len(none_preds) if none_preds else 1.0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'none_accuracy': none_accuracy,
        'confusion_matrix': conf_matrix.tolist()
    }

def plot_confusion_matrix(metrics, output_file=None):
    """
    Plot confusion matrix
    
    Args:
        metrics: Metrics dictionary containing confusion matrix
        output_file: File to save the plot (optional)
    """
    conf_matrix = np.array(metrics['confusion_matrix'])
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
               xticklabels=['True', 'False'],
               yticklabels=['True', 'False'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix plot saved to {output_file}")
    else:
        plt.show()

def tune_thresholds(test_file, ground_truth_file=None, 
                   relevance_thresholds=None, verify_thresholds=None,
                   **kwargs):
    """
    Tune relevance and verification thresholds
    
    Args:
        test_file: File containing test statements
        ground_truth_file: File containing ground truth labels (optional)
        relevance_thresholds: List of relevance thresholds to try
        verify_thresholds: List of verification thresholds to try
        **kwargs: Additional arguments for evaluate_on_test_set
        
    Returns:
        Best thresholds and corresponding metrics
    """
    if relevance_thresholds is None:
        relevance_thresholds = [0.3, 0.35, 0.4, 0.45, 0.5]
    
    if verify_thresholds is None:
        verify_thresholds = [0.6, 0.65, 0.7, 0.75, 0.8]
    
    best_accuracy = 0
    best_f1 = 0
    best_thresholds = None
    all_results = []
    
    for rel_threshold in relevance_thresholds:
        for ver_threshold in verify_thresholds:
            print(f"Trying relevance_threshold={rel_threshold}, verify_threshold={ver_threshold}")
            
            # Update thresholds in kwargs
            kwargs_copy = kwargs.copy()
            kwargs_copy['relevance_threshold'] = rel_threshold
            kwargs_copy['verify_threshold'] = ver_threshold
            
            # Evaluate with these thresholds
            results = evaluate_on_test_set(test_file, ground_truth_file, **kwargs_copy)
            
            metrics = results['metrics']
            accuracy = metrics['accuracy']
            f1 = metrics['f1']
            
            all_results.append({
                'relevance_threshold': rel_threshold,
                'verify_threshold': ver_threshold,
                'accuracy': accuracy,
                'f1': f1,
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'none_accuracy': metrics['none_accuracy']
            })
            
            # Update best thresholds based on F1 score
            if f1 > best_f1:
                best_f1 = f1
                best_accuracy = accuracy
                best_thresholds = {
                    'relevance_threshold': rel_threshold,
                    'verify_threshold': ver_threshold
                }
            
            print(f"Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
    
    # Sort results by F1 score
    all_results.sort(key=lambda x: x['f1'], reverse=True)
    
    print("\nTop 5 threshold combinations:")
    for i, result in enumerate(all_results[:5]):
        print(f"{i+1}. relevance={result['relevance_threshold']}, verify={result['verify_threshold']}")
        print(f"   Accuracy: {result['accuracy']:.4f}, F1: {result['f1']:.4f}")
        print(f"   Precision: {result['precision']:.4f}, Recall: {result['recall']:.4f}")
        print(f"   None Accuracy: {result['none_accuracy']:.4f}")
        print()
    
    print(f"Best thresholds: relevance={best_thresholds['relevance_threshold']}, verify={best_thresholds['verify_threshold']}")
    print(f"Best accuracy: {best_accuracy:.4f}")
    print(f"Best F1 score: {best_f1:.4f}")
    
    return best_thresholds, all_results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate Ramayana fact verification model")
    parser.add_argument("--test-file", "-t", required=True, help="File containing test statements")
    parser.add_argument("--ground-truth", "-g", help="File containing ground truth labels")
    parser.add_argument("--output-file", "-o", help="File to save evaluation results")
    parser.add_argument("--data-file", default="ramayana_data.csv", help="CSV file containing Ramayana data")
    parser.add_argument("--embeddings-path", default="ramayana_embeddings.pkl", 
                      help="Path to save/load embeddings")
    parser.add_argument("--model-name", default="all-MiniLM-L6-v2", 
                      help="Name of the SentenceTransformer model")
    parser.add_argument("--tune", action="store_true", help="Tune thresholds")
    parser.add_argument("--plot", action="store_true", help="Plot confusion matrix")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print detailed information")
    
    args = parser.parse_args()
    
    if args.tune:
        # Tune thresholds
        best_thresholds, all_results = tune_thresholds(
            args.test_file,
            args.ground_truth,
            data_file=args.data_file,
            embeddings_path=args.embeddings_path,
            model_name=args.model_name,
            verbose=args.verbose
        )
        
        # Save tuning results
        if args.output_file:
            with open(args.output_file, 'w') as f:
                json.dump({
                    'best_thresholds': best_thresholds,
                    'all_results': all_results
                }, f, indent=2)
    else:
        # Evaluate with current thresholds
        results = evaluate_on_test_set(
            args.test_file,
            args.ground_truth,
            args.output_file,
            data_file=args.data_file,
            embeddings_path=args.embeddings_path,
            model_name=args.model_name,
            verbose=args.verbose
        )
        
        if 'metrics' in results and args.plot:
            # Plot confusion matrix
            plot_confusion_matrix(
                results['metrics'],
                os.path.splitext(args.output_file)[0] + '_cm.png' if args.output_file else None
            )