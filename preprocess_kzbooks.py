import pandas as pd

kz = pd.read_csv('kazakhBooks.csv')

kz_nagyz = kz[(kz['contains_kaz_symbols'] == 1)&(kz['predicted_language']=='kaz')]['text']
kz_nagyz.to_csv('kz_nagyz.txt',header=False,index=False)