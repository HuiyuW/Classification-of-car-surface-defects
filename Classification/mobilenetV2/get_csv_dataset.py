import pandas as pd
import numpy as np
import random
import csv
import os.path
from pathlib import Path
from time import sleep
f = pd.read_csv('label0-896.csv')
l = len(f)
x = random.sample(range(int(l)), int(l))
# determine the large of train data
train_num = int(0.8 * l)
# print how many data in train set
print(train_num)
# determine the large of test data
test_num = int(0.1*l)
print(test_num)
#
train_list = x[:train_num]
test_list = x[train_num: (test_num + train_num)]
val_list = x[(test_num + train_num):]
print(len(train_list), len(test_list), len(val_list))
# if .csv file exists, delete it and make a new one
if Path('train_dataset.csv').is_file():
    os.remove('train_dataset.csv')
train_file = open('train_dataset.csv', 'w')
train_writer = csv.writer(train_file)
# write the chosen data in train set
for i in range(len(train_list)):
    train_writer.writerow(f.iloc[train_list[i], :])
train_file.close()
train_f = pd.read_csv('train_dataset.csv')
# add column names in .csv file
train_f.columns = ['annotation_index', 'image_id', 'human_label', 'label_name']
train_f.to_csv('train_dataset.csv', index=False)

if Path('test_dataset.csv').is_file():
    os.remove('test_dataset.csv')
test_file = open('test_dataset.csv', 'w')
test_writer = csv.writer(test_file)
for i in range(len(test_list)):
    test_writer.writerow(f.iloc[test_list[i], :])
test_file.close()
test_f = pd.read_csv('test_dataset.csv')
test_f.columns = ['annotation_index', 'image_id', 'human_label', 'label_name']
test_f.to_csv('test_dataset.csv', index=False)

if Path('val_dataset.csv').is_file():
    os.remove('val_dataset.csv')
# sleep(10)
val_file = open('val_dataset.csv', 'w')
val_writer = csv.writer(val_file)
for i in range(len(val_list)):
    val_writer.writerow(f.iloc[val_list[i], :])
val_file.close()
sleep(3)
val_f = pd.read_csv('val_dataset.csv')
val_f.columns = ['annotation_index', 'image_id', 'human_label', 'label_name']
val_f.to_csv('val_dataset.csv', index=False)
