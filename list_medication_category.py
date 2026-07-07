import pandas as pd
import numpy as np

def main():
    df = pd.read_csv('microbiology_combined_clean.csv', low_memory=False)
    col = 'medication_category'
    print('medication_category present:', col in df.columns)
    if col in df.columns:
        vc = df[col].astype(str).str.strip().replace({'': np.nan}).dropna().value_counts()
        print('Top medication_category values:')
        for k, v in vc.head(50).items():
            print(f"{k}\t{v}")
    else:
        print('Column not found. Columns sample:')
        print(', '.join(list(df.columns)[:50]))

if __name__ == '__main__':
    main()
