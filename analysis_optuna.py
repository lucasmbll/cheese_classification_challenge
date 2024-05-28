import pandas as pd
import matplotlib.pyplot as plt
import json
import ast
import os

def plot_accuracies(df):
    """
    Plots the training and validation accuracies per epoch for each trial, dividing them into separate plots.
    """
    print("Starting to plot accuracies...")
    try:
        # Sort trials by validation accuracy
        df_sorted = df.sort_values(by='best_val_acc', ascending=False)
        num_trials = len(df_sorted)

        # Define number of trials per plot (maximum 5)
        trials_per_plot = 5

        # Plotting loop for training accuracies
        for i in range(0, num_trials, trials_per_plot):
            plt.figure(figsize=(10, 6))

            for j in range(trials_per_plot):
                idx = i + j
                if idx >= num_trials:
                    break

                row = df_sorted.iloc[idx]
                trial_num = int(row['trial_number']) + 1

                train_acc_per_epoch = ast.literal_eval(row['train_acc_per_epoch'])

                plt.plot(train_acc_per_epoch, label=f'Trial {trial_num}')

            plt.xlabel('Epoch')
            plt.ylabel('Training Accuracy')
            plt.title('Training Accuracies per Epoch')
            plt.legend()

            # Define the output path
            output_dir = r'C:\Users\adib4\OneDrive\Documents\Travail\X\MODAL DL\cheese_classification_challenge\scp_copy'
            output_filename = f'training_accuracy_per_epoch_{i//trials_per_plot}.png'
            output_path = os.path.join(output_dir, output_filename)

            # Ensure the output directory exists
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                print(f"Created directory: {output_dir}")

            # Save the plot
            try:
                print(f"Saving plot to {output_path}...")
                plt.savefig(output_path)
                print(f"Plotting completed and saved as {output_path}")
                plt.close()  # Close the plot to free up memory
            except Exception as e:
                print(f"Error saving plot: {e}")

        # Plotting loop for validation accuracies
        for i in range(0, num_trials, trials_per_plot):
            plt.figure(figsize=(10, 6))

            for j in range(trials_per_plot):
                idx = i + j
                if idx >= num_trials:
                    break

                row = df_sorted.iloc[idx]
                trial_num = int(row['trial_number']) + 1

                val_acc_per_epoch = ast.literal_eval(row['val_acc_per_epoch'])

                plt.plot(val_acc_per_epoch, label=f'Trial {trial_num}')

            plt.xlabel('Epoch')
            plt.ylabel('Validation Accuracy')
            plt.title('Validation Accuracies per Epoch')
            plt.legend()

            # Define the output path
            output_dir = r'C:\Users\adib4\OneDrive\Documents\Travail\X\MODAL DL\cheese_classification_challenge\scp_copy'
            output_filename = f'validation_accuracy_per_epoch_{i//trials_per_plot}.png'
            output_path = os.path.join(output_dir, output_filename)

            # Ensure the output directory exists
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                print(f"Created directory: {output_dir}")

            # Save the plot
            try:
                print(f"Saving plot to {output_path}...")
                plt.savefig(output_path)
                print(f"Plotting completed and saved as {output_path}")
                plt.close()  # Close the plot to free up memory
            except Exception as e:
                print(f"Error saving plot: {e}")

    except Exception as e:
        print(f"Error while plotting accuracies: {e}")


def analyze_results(df):
    """
    Analyzes the results of the trials and prints the best hyperparameters and validation accuracy.
    """
    print("Starting analysis of results...")
    try:
        best_trial = df.iloc[df['best_val_acc'].idxmax()]

        print("Analysis of Results:")
        print("====================")
        print("The best hyperparameters found:")
        print(json.dumps(ast.literal_eval(best_trial['params']), indent=4))
        print(f"The best validation accuracy found: {best_trial['best_val_acc']}")
    except Exception as e:
        print(f"Error while analyzing results: {e}")

if __name__ == "__main__":
    csv_file = r'C:\Users\adib4\OneDrive\Documents\Travail\X\MODAL DL\cheese_classification_challenge\scp_copy/trial_results.csv'
    if os.path.isfile(csv_file):
        print(f"Loading trial results from {csv_file}")
        try:
            df = pd.read_csv(csv_file)
            print(f"Successfully loaded {len(df)} trials from the CSV file.")
        except Exception as e:
            print(f"Error loading CSV file: {e}")
            exit(1)

        # Plot the training and validation accuracies per epoch for each trial
        plot_accuracies(df)

        # Analyze the results and print the best hyperparameters and validation accuracy
        analyze_results(df)
    else:
        print(f"CSV file {csv_file} not found.")
