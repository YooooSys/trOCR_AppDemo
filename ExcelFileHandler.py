import pandas as pd


class ExcelFileHandler():
    def __init__(self, excel_path: str):
        self.excel_path = excel_path
        self.df = pd.read_excel(self.excel_path, engine='openpyxl')

    def get(self, mssv_title):
        return self.df[mssv_title].astype(str).tolist()
    
    def check(self, mssv_title, score_title):
        mssv_title = " ".join(mssv_title.split())
        score_title = " ".join(score_title.split())

        if mssv_title in self.df.columns and score_title in self.df.columns:
            return mssv_title, score_title
        
        else:
            return None
        