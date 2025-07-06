# Cargar EXCEL

import pandas as pd
import os

def load_dataset():
    excel_path = os.path.join(os.path.dirname(__file__), '../data/excel/setData.xlsx')
    
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"El archivo {excel_path} no existe.")
    
    df = pd.read_excel(excel_path)
    return df

# # Reemplazar los nulos
# df['Relationship_Status'] = df['Relationship_Status'].fillna("undefined")

# df[df.select_dtypes(include='number').columns] = df.select_dtypes(include='number').fillna(0)

# print(df.isnull().sum())

# df.to_excel('../data/excel/setData.xlsx', index=False)