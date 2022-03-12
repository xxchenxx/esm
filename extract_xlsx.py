import pandas as pd

xl_file = pd.ExcelFile("41592_2018_138_MOESM4_ESM.xlsx")

dfs = {sheet_name: xl_file.parse(sheet_name) 
          for sheet_name in xl_file.sheet_names}

for key in dfs:
    sheet = dfs[key]
    sheet.to_csv(f"{key}.csv")