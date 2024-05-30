import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def plot_confusion_matrix(csv_file):
    """
    Reads a CSV file containing actual and predicted labels and plots a confusion matrix.
    """
    # Read the CSV file
    df = pd.read_csv(csv_file)
    print("File read successfully")

    # Extract unique labels
    labels = sorted(df['Actual'].unique())
    print(f"Unique labels: {labels}")

    # Create the confusion matrix using the nPredictions column
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    label_to_index = {label: index for index, label in enumerate(labels)}
    for _, row in df.iterrows():
        actual_index = label_to_index[row['Actual']]
        predicted_index = label_to_index[row['Predicted']]
        cm[actual_index, predicted_index] += int(row['nPredictions'])
    print("Confusion matrix created")
    print("Confusion Matrix:")
    print(cm)

    # Plot the confusion matrix using seaborn
    plt.figure(figsize=(12, 10))
    print("Plotting confusion matrix...")
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    print("Plotting completed")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
    plt.close()  # Close the plot after showing
    print("Plot displayed and closed")

    # Save the plot
    output_dir = r'C:\Users\adib4\OneDrive\Documents\Travail\X\MODAL DL\cheese_classification_challenge\results'
    output_filename = 'confusion_matrix.png'
    output_path = os.path.join(output_dir, output_filename)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"Saving plot to {output_path}...")
    plt.savefig(output_path)
    plt.close()
    print(f"Confusion matrix plot saved to {output_path}")

if __name__ == "__main__":
    csv_file = r'C:\Users\adib4\OneDrive\Documents\Travail\X\MODAL DL\cheese_classification_challenge\results\conf_matric_doubletune.csv'
    if os.path.isfile(csv_file):
        plot_confusion_matrix(csv_file)
    else:
        print(f"CSV file {csv_file} not found.")
