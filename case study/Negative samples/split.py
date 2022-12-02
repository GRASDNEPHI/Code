import numpy as np
import pandas as pd
import csv
import math
import random
csv.field_size_limit(500 * 1024 * 1024)

def read_csv(save_list, file_name):
    csv_reader = csv.reader(open(file_name))
    for row in csv_reader:
        save_list.append(row)
    return

def store_csv(data, file_name):
    with open(file_name, "w", newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(data)
    return

def generate_negative_sample(relationship_pd):
    relationship_matrix = pd.pivot_table(relationship_pd, index='Pair1', columns='Pair2', values='Rating', fill_value=0)
    negative_sample = []
    counter = 0
    while counter < len(relationship_pd):
        print(counter)
        temp_1 = random.randint(0, len(relationship_matrix.index) - 1)
        temp_2 = random.randint(0, len(relationship_matrix.columns) - 1)
        if relationship_matrix.iloc[temp_1, temp_2] == 0 and temp_1 != 3385:
            relationship_matrix.iloc[temp_1, temp_2] = -1
            row = []
            row.append(np.array(relationship_matrix.index).tolist()[temp_1])
            row.append(np.array(relationship_matrix.columns).tolist()[temp_2])
            negative_sample.append(row)

            counter = counter + 1

        else:
            pass

    return negative_sample, relationship_matrix

if __name__ == '__main__':

    relationship = pd.read_csv(r'RNAInter miRNA drug interaction.csv').values.tolist()

    random.shuffle(relationship)    # random shuffle
    relationship_train = relationship[0: int(0.8 * len(relationship))]
    pd.DataFrame(relationship_train).to_csv(r'Allpositive Training Validation/PositiveSample_Train.csv', index=None, header=None)
    relationship_validation = relationship[int(0.8 * len(relationship)):]
    pd.DataFrame(relationship_validation).to_csv(r'Allpositive Training Validation/PositiveSample_Validation.csv', index=None, header=None)

    relationship_pd = pd.DataFrame(relationship, columns=['Pair1', 'Pair2'])
    relationship_pd['Rating'] = [1] * len(relationship_pd)
    negative_sample, relationship_matrix = generate_negative_sample(relationship_pd)
    relationship_matrix.to_csv('Allpositive Training Validation/Relationship_Matrix.csv')
    store_csv(negative_sample, 'Allpositive Training Validation/NegativeSample.csv')

    negative_sample_train = negative_sample[0: int(0.8 * len(negative_sample))]
    store_csv(negative_sample_train, 'Allpositive Training Validation/NegativeSample_Train.csv')
    negative_sample_validation = negative_sample[int(0.8 * len(negative_sample)):]
    store_csv(negative_sample_validation, 'Allpositive Training Validation/NegativeSample_Validation.csv')
