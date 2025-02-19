import pandas as pd
from Logging.logger import log
class repo_handler:

    def __init__(self):
        log.info("Initializing Repository Handler")
        self.malicious_data = []
        self.malicious_data_df = []
        self.LFI_df = []
        self.SQLi_df = []
        self.XSS_df = []
        self.RFI_df = []
        self.RCE_df = []
        self.normal_data = []
        self.normal_data_df = []
        self.excel_file = "./Repository/Dataset.xlsx"

        try:
            self.load_data()
            self.full_data = pd.concat([self.normal_data, self.malicious_data], ignore_index=True)
            self.malicious_data_df = pd.concat([self.RCE_df,self.LFI_df,self.RFI_df,self.SQLi_df,self.XSS_df])
            self.normal_data_df = pd.DataFrame(self.normal_data, columns=["UriQuery"])
            self.normal_data_df["category"] = "Benign"
            self.malicious_data_df["isVulnerable"] = 1 # 1 for malicious
            self.normal_data_df["isVulnerable"] = 0 # 0 for normal
            self.full_data_df = pd.concat([self.malicious_data_df,self.normal_data_df],ignore_index=True)
            self.randomize_n_time(5)
            log.info("Repository successfully initialized!")
        except Exception as e:
            log.error(f"Error during initialization: {e}")

    def load_data(self):
        """ Loads and processes data from the Excel file. """
        log.info("Loading Data from Excel...")

        try:
            excel_data = pd.read_excel(self.excel_file, sheet_name=None)
            data = {
                sheet_name: d["UriQuery"].drop_duplicates().dropna()
                for sheet_name, d in excel_data.items() if "UriQuery" in d.columns
            }

            if not data:
                log.warning("No data found in the Excel file!")

            self.normal_data = data["Benign"]
            self.RCE_df = pd.DataFrame(data["RCE"], columns=["UriQuery"])
            self.RCE_df["category"] = "RCE"

            self.LFI_df = pd.DataFrame(data["LFI"], columns=["UriQuery"])
            self.LFI_df["category"] = "LFI"

            self.SQLi_df = pd.DataFrame(data["SQLi"], columns=["UriQuery"])
            self.SQLi_df["category"] = "SQLi"

            self.XSS_df = pd.DataFrame(data["XSS"], columns=["UriQuery"])
            self.XSS_df["category"] = "XSS"

            self.RFI_df = pd.DataFrame(data["RFI"], columns=["UriQuery"])
            self.RFI_df["category"] = "RFI"


            self.malicious_data = pd.concat([data["RCE"],data["LFI"],data["SQLi"],data["XSS"],data["RFI"]], ignore_index=True)
            log.info(f"Loaded {len(data['SQLi'])} SQLi, {len(data['RFI'])} RFI, {len(data['LFI'])} LFI, {len(data['XSS'])} XSS and {len(data['RCE'])} RCE entries.")
            log.info(f"Loaded {len(self.normal_data)} normal and {len(self.malicious_data)} malicious entries.")
        except Exception as e:
            log.error(f"Error loading data: {e}")

    def randomize_n_time(self, counter):
        """ Randomizes the dataset multiple times. """
        log.info(f"Randomizing dataset {counter} times...")

        try:
            for _ in range(counter):
                self.full_data_df = self.full_data_df.sample(frac=1).reset_index(drop=True)
            log.info("Dataset successfully randomized!")
        except Exception as e:
            log.error(f"Error during randomization: {e}")
