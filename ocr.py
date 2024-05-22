import os
import hydra
import torch
import difflib
from fuzzywuzzy import fuzz, process
from PIL import Image
import numpy as np
import jellyfish
import rapidfuzz.process
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend for non-GUI environments
import pytesseract
import logging
import unidecode
import unicodedata

logging.getLogger().setLevel(logging.ERROR)

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def load_cheese_names(file_path):
    with open(file_path, 'r') as file:
        cheeses = file.read().splitlines()
    return cheeses

def decode_strange_chars(text):
    # Manually decode strange sequences into the correct French characters
    text = text.replace("BÃ›CHE DE CHÃˆVRE", "BÛCHETTE DE CHÈVRE")
    text = text.replace("BÃ›CHE", "BÛCHE")
    text = text.replace("CHÃˆVRE", "CHÈVRE")
    text = text.replace("COMTÃ‰", "COMTÉ")
    text = text.replace("GRUYÃˆRE", "GRUYÈRE")
    text = text.replace("TÃŠTE", "TÊTE")
    text = text.replace("SAINT- FÃ‰LICIEN", "SAINT- FÉLICIEN")
    text = text.replace("MONT Dâ€™OR", "MONT D’OR")

    return text

def check_cheese_name(detected_text, cheese_names, method='difflib', threshold=0.7):
    matches = []
    for text in detected_text:
        if method == 'difflib':
            match = difflib.get_close_matches(text, cheese_names, n=1, cutoff=threshold)
            if match:
                matches.append((text, match[0]))
        elif method == 'fuzzywuzzy':
            match, score = process.extractOne(text, cheese_names, scorer=fuzz.ratio)
            if score >= 0.5 * 100:  # fuzzywuzzy scores are out of 100
                match = decode_strange_chars(match)
                matches.append((text, match, score))
        elif method == 'jellyfish':
            max_length = max(len(text), max(len(cheese) for cheese in cheese_names))
            match_scores = [(cheese, jellyfish.levenshtein_distance(text, cheese) / max_length) for cheese in cheese_names]
            match, score = min(match_scores, key=lambda x: x[1])
            if score >= threshold:
                matches.append((text, match))
        elif method == 'rapidfuzz':
            match, score, _ = rapidfuzz.process.extractOne(text, cheese_names, scorer=rapidfuzz.fuzz.ratio)
            if score >= threshold * 100:  # rapidfuzz scores are out of 100
                matches.append((text, match))
    return matches

def initialize_ocr(ocr_method):
    if ocr_method == 'easyocr':
        import easyocr
        return easyocr.Reader(['fr'], gpu=torch.cuda.is_available())
    elif ocr_method == 'tesseract':
        return pytesseract
    else:
        raise ValueError(f"Unsupported OCR method: {ocr_method}")

def perform_ocr(ocr, img, ocr_method):
    if ocr_method == 'easyocr':
        result = ocr.readtext(np.array(img))
        detected_text = [line[1] for line in result]
    elif ocr_method == 'tesseract':
        detected_text = pytesseract.image_to_string(img, lang='fra').split('\n')
        detected_text = [text.strip() for text in detected_text if text.strip()]
    return detected_text

def classify_results_fuzzy(detected_text, cheese_names, high_treshold_cheese_names, threshold_base=0.8, increment=0.05, f=None):
    label = None
    matches = check_cheese_name(detected_text, cheese_names, method='fuzzywuzzy', threshold=threshold_base)
    if not matches: return label
    matches.sort(key=lambda x: x[2], reverse=True)
    if f:
        f.write("DEBUG : ", matches)     
    if (matches[0][1] == "FROMAGE FRAIS" or matches[0][1] == "FROMAGE BLANC"):
        threshold = min(threshold_base + increment*2, 0.95)
        if (len(matches)>1 and matches[1][1] != "FROMAGE BLANC" and matches[1][1]!= "FROMAGE FRAIS"):
            if (matches[1][2] > threshold_base * 100): 
                matches[0], matches[1] = matches[1], matches[0]
                threshold = threshold_base
    if (matches[0][1] in high_treshold_cheese_names):
        threshold = threshold_base + increment
    else:
        threshold = threshold_base

    if (matches[0][2] >= threshold * 100):
        if (matches[0][1] == "CHÈVRE" and len(matches)>1):
            if ((matches[1][1]  == "BÛCHE" or matches[1][1]  == "BÛCHETTE DE CHÈVRE"
                or matches[1][1]== "BÛCHE DE CHÈVRE") and matches[1][2] >= threshold_base * 100):  
                label = "BÛCHETTE DE CHÈVRE"
            else:  label = matches[0][1]
        elif (matches[0][1] =='PARMIGIANO'): label = "PARMESAN"
        elif (matches[0][1] =='BERTHAUT'): label = "EPOISSES"
        else: label = matches[0][1]

    return label

def classify_image(image, ocr, cheese_names, high_treshold_cheese_names, threshold_base=0.8, increment=0.05, ocr_method='easyocr', comparison_method='fuzzywuzzy'):
    detected_text = perform_ocr(ocr, image, ocr_method)
    label = classify_results_fuzzy(detected_text, cheese_names, high_treshold_cheese_names, threshold_base=threshold_base, increment=increment, f=f)
    
    return label

@hydra.main(config_path="configs/train", config_name="config")
def test_ocr(cfg):
    cheese_names = load_cheese_names('C:/Users/adib4/OneDrive/Documents/Travail/X/MODAL DL/cheese_classification_challenge/list_of_cheese.txt')
    ocr_images_dir = 'C:/Users/adib4/OneDrive/Documents/Travail/X/MODAL DL/cheese_classification_challenge/ocr_images'
    ocr_methods = ['easyocr', 'tesseract']
    comparison_methods = ['difflib', 'fuzzywuzzy', 'jellyfish', 'rapidfuzz']
    thresholds = np.arange(0.6, 0.95, 0.05)

    for ocr_method in ocr_methods:
        ocr = initialize_ocr(ocr_method)  # Initialize the OCR engine

        results = {method: {'thresholds': [], 'accuracies': [], 'total_guesses': []} for method in comparison_methods}

        for comparison_method in comparison_methods:
            for threshold in thresholds:
                print(f"Start of tests for OCR Method: {ocr_method}, Comparison Method: {comparison_method}, Threshold: {threshold:.2f}")
                correct_guesses = 0
                total_guesses = 0

                for cheese_label in os.listdir(ocr_images_dir):
                    label_path = os.path.join(ocr_images_dir, cheese_label)
                    if os.path.isdir(label_path):
                        for image_name in os.listdir(label_path):
                            image_path = os.path.join(label_path, image_name)
                            img = Image.open(image_path).convert('RGB')

                            detected_text = perform_ocr(ocr, img, ocr_method)

                            matches = check_cheese_name(detected_text, cheese_names, method=comparison_method, threshold=threshold)
                            
                            if matches:
                                total_guesses += 1
                                # print(f"Image: {image_name}, Label: {cheese_label}, Match: {matches[0]}")
                                if matches[0][1] == cheese_label:  # Check if the first match is correct
                                    correct_guesses += 1

                accuracy = correct_guesses / total_guesses * 100 if total_guesses > 0 else 0
                results[comparison_method]['thresholds'].append(threshold)
                results[comparison_method]['accuracies'].append(accuracy)
                results[comparison_method]['total_guesses'].append(total_guesses)

                print(f"OCR Method: {ocr_method}, Comparison Method: {comparison_method}, Threshold: {threshold:.2f}, Total Guesses: {total_guesses}, Correct Guesses: {correct_guesses}, Accuracy: {accuracy:.2f}%")

        plot_results(results, ocr_method)

def plot_results(results, ocr_method):
    output_dir = f'ocr_test_param/{ocr_method}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for method, data in results.items():
        plt.figure()
        plt.plot(data['thresholds'], data['accuracies'], label='Accuracy')
        plt.xlabel('Threshold')
        plt.ylabel('Accuracy (%)')
        plt.title(f'Accuracy vs Threshold for {method} ({ocr_method})')
        plt.legend()
        plt.savefig(f'{output_dir}/accuracy_{method}.png')
        plt.close()

        plt.figure()
        plt.plot(data['thresholds'], data['total_guesses'], label='Total Guesses')
        plt.xlabel('Threshold')
        plt.ylabel('Number of Guesses')
        plt.title(f'Number of Guesses vs Threshold for {method} ({ocr_method})')
        plt.legend()
        plt.savefig(f'{output_dir}/guesses_{method}.png')
        plt.close()

    plt.figure()
    for method, data in results.items():
        plt.plot(data['thresholds'], data['accuracies'], label=method)
    plt.xlabel('Threshold')
    plt.ylabel('Accuracy (%)')
    plt.title(f'Accuracy Comparison Across Methods ({ocr_method})')
    plt.legend()
    plt.savefig(f'{output_dir}/accuracy_comparison.png')
    plt.close()

def save_plots(results):
    output_dir = "plots"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    results = np.round(results, 2)
    plt.figure()
    plt.plot(results[:, 0], results[:, 2], label="Accuracy")
    plt.xlabel("Threshold")
    plt.ylabel("Accuracy (%)")
    plt.title(f"Accuracy vs Threshold for {ocr_method} and {comparison_method}")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "accuracy_plot.png"))



if __name__ == "__main__":
    f = open("ocr_errors.txt", "w")
    f.write("OCR Errors for easyocr/fuzzywuzzy classification\n\n")
    cheese_names = load_cheese_names('list_of_cheese.txt')  # Assuming the file is in the same directory
    simple_names = []
    ocr_images_dir = 'ocr_images'
    thresholds = np.arange(0.6, 0.95, 0.05)
    ocr_method = 'easyocr'
    ocr = initialize_ocr(ocr_method)
    comparison_method = 'fuzzywuzzy'
    results = []

    for threshold in thresholds:
        print(f"Start of tests for OCR Method: {ocr_method}, Comparison Method: {comparison_method}, Threshold: {threshold:.2f}")
        f.write("Errors for threshold: " + str(threshold) + "\n")
        correct_guesses = 0
        total_guesses = 0

        for cheese_label in os.listdir(ocr_images_dir):
            label_path = os.path.join(ocr_images_dir, cheese_label)
            if os.path.isdir(label_path):
                for image_name in os.listdir(label_path):
                    image_path = os.path.join(label_path, image_name)
                    img = Image.open(image_path).convert('RGB')

                    detected_text = perform_ocr(ocr, img, ocr_method)

                    matches = check_cheese_name(detected_text, cheese_names, method=comparison_method, threshold=threshold)
                    label = classify_results_fuzzy(detected_text, cheese_names, simple_names, threshold_base=threshold, increment=0.05)

                    if label:
                        label = decode_strange_chars(label)
                        total_guesses += 1
                        if label == cheese_label:  # Check if the first match is correct
                            correct_guesses += 1
                        else:
                            print(f"Image: {image_name}, Label: {cheese_label}, Detected: {label}")
                            f.write(f"Error for Image: {image_name}, Label: {cheese_label}, Detected: {label}\n")

        accuracy = correct_guesses / total_guesses * 100 if total_guesses > 0 else 0
        results.append(threshold, total_guesses, correct_guesses, accuracy)

        print(f"OCR Method: {ocr_method}, Comparison Method: {comparison_method}, Threshold: {threshold:.2f}, Total Guesses: {total_guesses}, Correct Guesses: {correct_guesses}, Accuracy: {accuracy:.2f}%")

    print(results)
    # Save results in plots
    save_plots(np.array(results))
