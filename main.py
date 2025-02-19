from Repository.repo_handler import repo_handler
from Logging.logger import main_log
from web_attack_detector_model.model_trainer import WebAttackDetector
from torch import cuda
import numpy as np
import os
import csv
import time
device = 'cuda' if cuda.is_available() else 'cpu'
#print(cuda.is_available())
main_log.warning(f"Current: {device}")


def track_misclassifications(real_payloads, true_label):
    for payload in real_payloads:
        for category in categories:
            if category != true_label and payload in globals()[category.lower()]:
                misclassification_counts[true_label][category] += 1
                break

def update_counts_and_lists(category_list, category_real):
    matches = set(category_list).intersection(category_real)
    count = len(matches)
    category_list = [item for item in category_list if item not in matches]
    category_real = [item for item in category_real if item not in matches]
    return count, category_list, category_real

if __name__ == "__main__":
    main_log.info("Starting the model...")

    try:
        data = repo_handler()
        unique_values = data.full_data.unique()
        main_log.info(f"Unique UriQuery values: {len(unique_values)}")
        main_log.info("Initializing Model")
        model = WebAttackDetector(data.full_data_df)
        #model.train()
        model.load_model()
        np.set_printoptions(suppress=True)
        #model.load_model()
        categories = {0: 'XSS', 1: 'Benign',  2: 'LFI', 3: 'SQLi',  4: 'RCE', 5: 'RFI'}
        #while True:
        #    test = input("Enter URI:\n")

        test_results = {}
        test_cases = []
        test_dir = os.path.join("Repository", "test")
        xss_real = set()
        rfi_real = set()
        lfi_real = set()
        rce_real = set()
        sqli_real = set()
        benign_real = set()
        try:
            for filename in os.listdir(test_dir):
                if filename.endswith(".csv"):
                    path = os.path.join(test_dir, filename)
                    category = os.path.splitext(filename)[0]

                    try:
                        with open(path, mode='r', encoding='latin-1') as file:
                            reader = csv.reader(file)
                            data = [item for row in reader for item in row]
                            if category == "XSS":
                                xss_real = set(data)
                            if category == "RFI":
                                rfi_real = set(data)
                            if category == "RCE":
                                rce_real = set(data)
                            if category == "SQLi":
                                sqli_real = set(data)
                            if category == "Benign":
                                benign_real = set(data)
                            if category == "LFI":
                                lfi_real = set(data)
                            test_cases.extend(data)
                            test_results[category] = len(data)
                            main_log.debug(f"{category}: {len(data)} samples")
                    except Exception as e:
                        main_log.error(f"Error reading file {filename}: {str(e)}")

        except Exception as e:
            main_log.error(f"Error accessing test directory: {str(e)}")

        xss = set()
        rfi = set()
        lfi = set()
        rce = set()
        sqli = set()
        benign = set()
        main_log.info(f"Len: {len(test_cases)}")
        start = time.time()
        for test in test_cases:
            category,pred = model.predict(test,categories)
            if category == "XSS":
                xss.add(test)
            if category == "RFI":
                rfi.add(test)
            if category == "RCE":
                rce.add(test)
            if category == "SQLi":
                sqli.add(test)
            if category == "Benign":
                benign.add(test)
            if category == "LFI":
                lfi.add(test)
            #log.debug(f"Category: {category}\nPredication: {pred}")
        end = time.time()
        xss_count = 0
        benign_count = 0
        sqli_count = 0
        lfi_count = 0
        rce_count = 0
        rfi_count = 0

        xss_count, xss, xss_real = update_counts_and_lists(xss, xss_real)
        benign_count, benign, benign_real = update_counts_and_lists(benign, benign_real)
        sqli_count, sqli, sqli_real = update_counts_and_lists(sqli, sqli_real)
        lfi_count, lfi, lfi_real = update_counts_and_lists(lfi, lfi_real)
        rce_count, rce, rce_real = update_counts_and_lists(rce, rce_real)
        rfi_count, rfi, rfi_real = update_counts_and_lists(rfi, rfi_real)

        counts = {
            "XSS": xss_count,
            "RFI": rfi_count,
            "LFI": lfi_count,
            "RCE": rce_count,
            "SQLi": sqli_count,
            "Benign": benign_count
        }

        for category, count in counts.items():
            main_log.info(f"Found {count} out of {test_results[category]}")

        categories = {
            "XSS": (xss, xss_real),
            "Benign": (benign, benign_real),
            "SQLi": (sqli, sqli_real),
            "LFI": (lfi, lfi_real),
            "RCE": (rce, rce_real),
            "RFI": (rfi, rfi_real)
        }

        for name, (predicted, real) in categories.items():
            main_log.warning(f"Mismatched {len(predicted)} {name}, Remaining: {len(real)}")

        categories = ["XSS", "RFI", "LFI", "RCE", "SQLi", "Benign"]

        misclassification_counts = {
            true_cat: {pred_cat: 0 for pred_cat in categories if pred_cat != true_cat}
            for true_cat in categories
        }

        track_misclassifications(xss_real, "XSS")
        track_misclassifications(sqli_real, "SQLi")
        track_misclassifications(rce_real, "RCE")
        track_misclassifications(lfi_real, "LFI")
        track_misclassifications(rfi_real, "RFI")
        track_misclassifications(benign_real, "Benign")

        for true_cat, predictions in misclassification_counts.items():
            for pred_cat, count in predictions.items():
                main_log.debug(f"{true_cat} misclassified as {pred_cat}: {count}")

        main_log.info(f"Running Time: {end-start}")




    except Exception as e:
        main_log.error(f"Error in model execution: {e}")


