# @Time  : 2022/3/28 10:13
# @File  : 检查Confirmed.py

import pandas as pd
import numpy as np
from tqdm import trange

if __name__ == '__main__':
    csstd_result = pd.read_csv(r'csstd all result.csv',header=None).values.tolist()
    csstd_test = pd.read_csv(r'../cs all test.csv').values.tolist()
    csstd_confirmed = pd.read_csv(r'../../Negative samples/ confirmed.csv').values.tolist()

    confirmed = []
    for a in csstd_result:
        item = []
        item.append(a[0])
        item.append(a[1])
        if [a[0],a[1]] in csstd_test:
            item.append(1)
        else:
            item.append(0)
        confirmed.append(item)
    pd.DataFrame(confirmed).to_csv(r'31703 result.csv',header=None,index=None)

