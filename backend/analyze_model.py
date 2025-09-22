"""
Analyze model performance and class balance.
"""
import os
import json
import numpy as np
from pathlib import Path
from PIL import Image
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from species_classifier import predict_species, _lazy_init, _preprocess_for_onnx, _onnx_predict, _softmax
from image_processing import load_image_bytes

# Initialize the model
_lazy_init()

def analyze_predictions(image_dir: str, labels_path: str = None):
    """Analyze model predictions on a directory of test images."""
    # Get ground truth if available
    ground_truth = {}
    if labels_path and os.path.exists(labels_path):
        with open(labels_path) as f:
            ground_truth = json.load(f)
    
    # Collect predictions
    results = []
    image_dir = Path(image_dir)
    
    for img_path in image_dir.glob('*.jpg'):
        img_id = img_path.stem
        true_label = ground_truth.get(img_id, 'unknown')
        
        try:
            # Get prediction
            with open(img_path, 'rb') as f:
                image_bytes = f.read()
            
            # Get raw prediction scores
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"Warning: Could not read {img_path}")
                continue
                
            # Preprocess
            input_img = _preprocess_for_onnx(img)
            
            # Get raw scores
            ort_session = _lazy_init()
            input_name = ort_session.get_inputs()[0].name
            output = ort_session.run(None, {input_name: input_img.astype(np.float32)})
            scores = _softmax(output[0][0])
            
            # Get predicted class and confidence
            predicted_idx = np.argmax(scores)
            confidence = float(scores[predicted_idx])
            predicted_class = ["cattle", "buffalo"][predicted_idx]
            
            results.append({
                'image_id': img_id,
                'true_label': true_label,
                'predicted': predicted_class,
                'cattle_score': float(scores[0]),
                'buffalo_score': float(scores[1]),
                'confidence': confidence,
                'is_correct': predicted_class.lower() == true_label.lower()
            })
            
            print(f"{img_id}: Predicted {predicted_class} (Confidence: {confidence:.2f}), "
                  f"True: {true_label}")
            
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
    
    if not results:
        print("No results to analyze.")
        return
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame(results)
    
    # Print basic stats
    print("\n=== Prediction Statistics ===")
    print(f"Total images: {len(df)}")
    print(f"Predicted as cattle: {(df['predicted'] == 'cattle').sum()}")
    print(f"Predicted as buffalo: {(df['predicted'] == 'buffalo').sum()}")
    
    if 'true_label' in df.columns and df['true_label'].nunique() > 1:
        print("\n=== Classification Report ===")
        print(classification_report(df['true_label'], df['predicted'], 
                                 target_names=['cattle', 'buffalo']))
        
        # Plot confusion matrix
        cm = confusion_matrix(df['true_label'], df['predicted'], 
                            labels=['cattle', 'buffalo'])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                    display_labels=['cattle', 'buffalo'])
        disp.plot()
        plt.title('Confusion Matrix')
        plt.show()
    
    # Analyze confidence scores
    print("\n=== Confidence Analysis ===")
    print("Average confidence by prediction:")
    print(df.groupby('predicted')['confidence'].describe())
    
    # Check for misclassified examples
    if 'true_label' in df.columns:
        misclassified = df[df['true_label'] != 'unknown']
        misclassified = misclassified[misclassified['predicted'] != misclassified['true_label']]
        
        if not misclassified.empty:
            print("\n=== Misclassified Examples ===")
            print(misclassified[['image_id', 'true_label', 'predicted', 'confidence']])
    
    return df

def visualize_decision_boundary(image_paths, output_dir='output/analysis'):
    """Visualize model's decision boundary on sample images."""
    os.makedirs(output_dir, exist_ok=True)
    
    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
            
        # Get prediction
        with open(img_path, 'rb') as f:
            image_bytes = f.read()
        
        # Get raw scores
        input_img = _preprocess_for_onnx(img)
        ort_session = _lazy_init()
        input_name = ort_session.get_inputs()[0].name
        output = ort_session.run(None, {input_name: input_img.astype(np.float32)})
        scores = _softmax(output[0][0])
        
        # Create visualization
        plt.figure(figsize=(10, 5))
        
        # Show image
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(f"{img_path.name}")
        plt.axis('off')
        
        # Show prediction scores
        plt.subplot(1, 2, 2)
        classes = ['Cattle', 'Buffalo']
        colors = ['blue', 'orange']
        plt.bar(classes, scores, color=colors)
        plt.ylim(0, 1.0)
        plt.title('Prediction Scores')
        plt.ylabel('Confidence')
        
        # Save the figure
        output_path = os.path.join(output_dir, f"analysis_{img_path.stem}.png")
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        print(f"Saved analysis to {output_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze model predictions.')
    parser.add_argument('--image-dir', type=str, default='data/images',
                       help='Directory containing test images')
    parser.add_argument('--labels', type=str, default=None,
                       help='Path to JSON file with ground truth labels')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualizations for sample images')
    
    args = parser.parse_args()
    
    # Run analysis
    df = analyze_predictions(args.image_dir, args.labels)
    
    # Generate visualizations if requested
    if args.visualize and os.path.exists(args.image_dir):
        image_paths = list(Path(args.image_dir).glob('*.jpg'))
        if image_paths:
            visualize_decision_boundary(image_paths[:10])  # Visualize first 10 images
