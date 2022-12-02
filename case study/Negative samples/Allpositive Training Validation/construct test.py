
import pandas as pd
import numpy as np
from tqdm import trange

if __name__ == '__main__':
    mirna = pd.read_csv(r'../RNAInter miRNA name.csv',header=None).values.tolist()
    drug = pd.read_csv(r'../RNAInter drug name.csv',header=None).values.tolist()

    positive = pd.read_csv(r'../RNAInter miRNA drug interaction.csv',header=None).values.tolist()
    negative = pd.read_csv(r'NegativeSample.csv',header=None).values.tolist()

    for a in positive:
        print(a)

    allinteraction = []
    for a in mirna:
        for b in drug:
            item = []
            item.append(a)
            item.append(b)
            if item in positive:
                print(item)
                break
            elif item in negative:
                print(item)
                break
            elif item in allinteraction:
                break
            else:
                test_intera = a[0],b[0]
                print(len(allinteraction))
                allinteraction.append(test_intera)

    pd.DataFrame(allinteraction).to_csv(r'cs global test.csv',header=None,index=None)



