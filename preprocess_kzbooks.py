
#kz = pd.read_csv('kazakhBooks.csv')

#kz_nagyz = kz[(kz['contains_kaz_symbols'] == 1)&(kz['predicted_language']=='kaz')]['text']
#kz_nagyz.to_csv('kz_nagyz.txt',header=False,index=False)


with open('kz_nagyz.txt','r',encoding='UTF-8') as f:
    data = f.read()

data = data[:2000000]

print(f"length of dataset in characters: {len(data):,}")

# get all the unique characters tha
#  occur in this text
chars = sorted(list(set(data)))
vocab_size = len(chars)
print("all the unique characters:", ''.join(chars))
print(f"vocab size: {vocab_size:,}")

with open('twomillion_kz.txt','w',encoding='UTF-8') as f:
    f.write(data)