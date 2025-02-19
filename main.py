from Repository.repo_handler import repo_handler
from Logging.logger import log
from web_attack_detector_model.model_trainer import WebAttackDetector
from torch import cuda
import numpy as np

device = 'cuda' if cuda.is_available() else 'cpu'
#print(cuda.is_available())
print(device)



if __name__ == "__main__":
    log.info("Starting the model...")

    try:
        data = repo_handler()
        unique_values = data.full_data.unique()
        log.info(f"Unique UriQuery values: {len(unique_values)}")
        log.info("Initializing Model")
        model = WebAttackDetector(data.full_data_df)
        #model.train()
        model.load_model()
        np.set_printoptions(suppress=True)
        #model.load_model()
        categories = {0: 'XSS', 1: 'Benign',  2: 'LFI', 3: 'SQLi',  4: 'RCE', 5: 'RFI'}
        while True:
            test = input("Enter URI:\n")
            category,pred = model.predict(test,categories)
            log.info(f"Category: {category}\nPrediction: {pred}")

    except Exception as e:
        log.error(f"Error in model execution: {e}")

