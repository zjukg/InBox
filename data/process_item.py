import pandas as pd
import os    
    

data = pd.read_csv('/home/xzz/I2B/data/yelp2018/item_list.txt', sep = ' ')

data = data.drop(['freebase_id'], axis=1)

data.to_csv('/home/xzz/I2B/data/yelp2018/item_list1.txt', sep = ' ', index=False)