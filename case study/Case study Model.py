import random
import keras
from keras import regularizers
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input
from keras import backend as K
from keras.callbacks import ReduceLROnPlateau
from keras.utils import np_utils
import csv
import pandas as pd
import numpy as np
from tqdm import trange

csv.field_size_limit(500 * 1024 * 1024)

# 定义函数
def ReadMyCsv1(SaveList, fileName):
    csv_reader = csv.reader(open(fileName))
    for row in csv_reader:  # 把每个rna疾病对加入OriginalData，注意表头
        SaveList.append(row)
    return

def ReadMyCsv2(SaveList, fileName):
    csv_reader = csv.reader(open(fileName))
    for row in csv_reader:
        for i in range(len(row)):       # 转换数据类型
            row[i] = float(row[i])
        SaveList.append(row)
    return

def ReadMyCsv3(SaveList, fileName):
    csv_reader = csv.reader(open(fileName))
    for row in csv_reader:
        counter = 1
        while counter < len(row):
            row[counter] = float(row[counter])
            counter = counter + 1
        SaveList.append(row)
    return

def StorFile(data, fileName):
    with open(fileName, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
    return

def GenerateEmbeddingFeature(SequenceList, EmbeddingList, PaddingLength):   # 产生所有miRNA/drug的embedding表示
    SampleFeature = []

    counter = 0
    while counter < len(SequenceList):
        PairFeature = []
        PairFeature.append(SequenceList[counter][0])        # 加入名称

        FeatureMatrix = []
        counter1 = 0            # 生成特征矩阵
        while counter1 < PaddingLength:  # 截取长度
            row = []
            counter2 = 0
            while counter2 < len(EmbeddingList[0]) - 1:  # embedding长度
                row.append(0)
                counter2 = counter2 + 1
            FeatureMatrix.append(row)
            counter1 = counter1 + 1

        try:
            counter3 = 0
            while counter3 < PaddingLength:
                counter4 = 0
                while counter4 < len(EmbeddingList):
                    if SequenceList[counter][1][counter3] == EmbeddingList[counter4][0]:
                        FeatureMatrix[counter3] = EmbeddingList[counter4][1:]
                        break
                    counter4 = counter4 + 1
                counter3 = counter3 + 1
        except:
            pass

        PairFeature.append(FeatureMatrix)
        SampleFeature.append(PairFeature)
        counter = counter + 1
    return SampleFeature

def GenerateSampleFeature(InteractionList, EmbeddingFeature1, EmbeddingFeature2):
    SampleFeature1 = []
    SampleFeature2 = []

    counter = 0
    while counter < len(InteractionList):
        Pair1 = InteractionList[counter][0]
        Pair2 = InteractionList[counter][1]

        counter1 = 0
        while counter1 < len(EmbeddingFeature1):
            if EmbeddingFeature1[counter1][0] == Pair1:
                SampleFeature1.append(EmbeddingFeature1[counter1][1])
                break
            counter1 = counter1 + 1

        counter2 = 0
        while counter2 < len(EmbeddingFeature2):
            if EmbeddingFeature2[counter2][0] == Pair2:
                SampleFeature2.append(EmbeddingFeature2[counter2][1])
                break
            counter2 = counter2 + 1

        counter = counter + 1
    SampleFeature1, SampleFeature2 = np.array(SampleFeature1), np.array(SampleFeature2)
    SampleFeature1.astype('float32')
    SampleFeature2.astype('float32')
    SampleFeature1 = SampleFeature1.reshape(SampleFeature1.shape[0], SampleFeature1.shape[1], SampleFeature1.shape[2], 1)
    SampleFeature2 = SampleFeature2.reshape(SampleFeature2.shape[0], SampleFeature2.shape[1], SampleFeature2.shape[2], 1)
    return SampleFeature1, SampleFeature2

def GenerateBehaviorFeature(InteractionPair, NodeBehavior):
    SampleFeature1 = []
    SampleFeature2 = []
    for i in range(len(InteractionPair)):
        Pair1 = str(InteractionPair[i][0])#miRNA
        Pair2 = str(InteractionPair[i][1])#drug

        for m in range(len(NodeBehavior)):
            if Pair1 == NodeBehavior[m][0]:
                SampleFeature1.append(NodeBehavior[m][1:])
                break

        for n in range(len(NodeBehavior)):
            if Pair2 == NodeBehavior[n][0]:
                SampleFeature2.append(NodeBehavior[n][1:])
                break

    SampleFeature1 = np.array(SampleFeature1).astype('float32')
    SampleFeature2 = np.array(SampleFeature2).astype('float32')

    return SampleFeature1, SampleFeature2

def GenerateAttributeFeature(InteractionPair, drug_feature, miRNA_feature):
    SampleFeature1 = []
    SampleFeature2 = []
    miss1 = []
    miss2 = []
    for i in trange(len(InteractionPair)):
        Pair1 = str(InteractionPair[i][1])  #drug
        Pair2 = str(InteractionPair[i][0])  #mirna
        for m in range(len(drug_feature)):#drug
            if int(Pair1) == int(drug_feature[m][0]):
                SampleFeature1.append(drug_feature[m][1:])
                break
            if Pair1 not in miss1:
                miss1.append(Pair1)
        for n in range(len(miRNA_feature)):#mirna
            if Pair2 == str(miRNA_feature[n][0]):
                SampleFeature2.append(miRNA_feature[n][1:])
                break
            if Pair2 not in miss2:
                miss2.append(Pair2)
    print('miss1:',miss1)
    print('miss2:',miss2)
        # pd.DataFrame(miss1).to_csv(r'attribue miss1,csv',header=None,index=None)
        # pd.DataFrame(miss2).to_csv(r'attribue miss2,csv', header=None, index=None)

    SampleFeature1 = np.array(SampleFeature1).astype('float32')
    SampleFeature2 = np.array(SampleFeature2).astype('float32')

    return SampleFeature1, SampleFeature2

def MyLabel(Sample):
    label = []
    for i in range(int(len(Sample) / 2)):
        label.append(1)
    for i in range(int(len(Sample) / 2)):
        label.append(0)
    return label

def scheduler(epoch):
    # 每隔100个epoch，学习率减小为原来的1/2
    if epoch % 2 == 0 and epoch != 0:
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, lr * 0.6)
        print("lr changed to {}".format(lr * 0.6))
    return K.get_value(model.optimizer.lr)

def MyChange(list):
    List = []
    for i in range(len(list)):
        row = []
        row.append(i)               # x
        row.append(float(list[i]))  # y
        List.append(row)
    return List

if __name__ == '__main__':
    # ker+MACCSfp
    AllNodeBehavior = pd.read_csv(r'All_phi_SDNE64.csv', header=None).values.tolist()
    AllDrugFingerPrint = pd.read_csv(r'../MACCS/3385RNAInter drug MACCS2.csv', header=None).astype(np.float).values.tolist()
    AllMiRNAKer = pd.read_csv(r'../kmer/All miRNA Kmer.csv', header=None).values.tolist()

    PositiveSample_Train = pd.read_csv(r'../Allpositive Training Validation/PositiveSample_Train.csv', header=None).values.tolist()
    PositiveSample_Validation = pd.read_csv(r'../Allpositive Training Validation/PositiveSample_Validation.csv', header=None).values.tolist()
    NegativeSample_Train = pd.read_csv(r'../Allpositive Training Validation/NegativeSample_Train.csv', header=None).values.tolist()
    NegativeSample_Validation = pd.read_csv(r'../Allpositive Training Validation/NegativeSample_Validation.csv', header=None).values.tolist()
    #CaseStudy Test
    casestudy_test = pd.read_csv(r'../Allpositive Training Validation/cs global test.csv', header=None).values.tolist()
    casestudy_test = random.sample(casestudy_test, 10000)
    pd.DataFrame(casestudy_test).to_csv(r'cs all test.csv',header=None,index=None)



    x_train_pair = []
    x_train_pair.extend(PositiveSample_Train)
    x_train_pair.extend(NegativeSample_Train)

    x_validation_pair = []
    x_validation_pair.extend(PositiveSample_Validation)
    x_validation_pair.extend(NegativeSample_Validation)

    # drug and miRNA feature. matrix and vector
    x_train_1_Attribute, x_train_2_Attribute = GenerateAttributeFeature(x_train_pair, AllDrugFingerPrint,AllMiRNAKer)
    x_validation_1_Attribute, x_validation_2_Attribute = GenerateAttributeFeature(x_validation_pair,AllDrugFingerPrint,AllMiRNAKer)
    x_train_1_Behavior, x_train_2_Behavior = GenerateBehaviorFeature(x_train_pair, AllNodeBehavior)
    x_validation_1_Behavior, x_validation_2_Behavior = GenerateBehaviorFeature(x_validation_pair, AllNodeBehavior)

    x_train_1_Attribute, x_train_2_Attribute = GenerateAttributeFeature(x_train_pair, AllDrugFingerPrint,AllMiRNAKer)
    x_validation_1_Attribute, x_validation_2_Attribute = GenerateAttributeFeature(x_validation_pair,AllDrugFingerPrint,AllMiRNAKer)
    x_train_1_Behavior, x_train_2_Behavior = GenerateBehaviorFeature(x_train_pair, AllNodeBehavior)
    x_validation_1_Behavior, x_validation_2_Behavior = GenerateBehaviorFeature(x_validation_pair, AllNodeBehavior)


    #CaseStudy Test
    x_test_1_Attribute, x_test_2_Attribute = GenerateAttributeFeature(casestudy_test, AllDrugFingerPrint,AllMiRNAKer)
    x_test_1_Behavior, x_test_2_Behavior = GenerateBehaviorFeature(casestudy_test, AllNodeBehavior)


    num_classes = 2
    y_train_Pre = MyLabel(x_train_pair)     # Label->one hot
    y_validation_Pre = MyLabel(x_validation_pair)
    #y_test_Pre = MyLabel(x_test_pair)

    y_train = np_utils.to_categorical(y_train_Pre, num_classes)
    y_validation = np_utils.to_categorical(y_validation_Pre, num_classes)
    #y_test = np_utils.to_categorical(y_test_Pre, num_classes)

    print('x_train_1_Attribute shape', x_train_1_Attribute.shape)
    print('x_train_2_Attribute shape', x_train_2_Attribute.shape)
    print('x_train_1_Behavior shape', x_train_1_Behavior.shape)
    print('x_train_2_Behavior shape', x_train_2_Behavior.shape)
    print('y_train shape:', y_train.shape)

    print('x_validation_1_Attribute shape', x_validation_1_Attribute.shape)
    print('x_validation_2_Attribute shape', x_validation_2_Attribute.shape)
    print('x_validation_1_Behavior shape', x_validation_1_Behavior.shape)
    print('x_validation_2_Behavior shape', x_validation_2_Behavior.shape)
    print('y_validation shape:', y_validation.shape)

    print('x_test_1_Attribute shape', x_test_1_Attribute.shape)
    print('x_test_2_Attribute shape', x_test_2_Attribute.shape)
    print('x_test_1_Behavior shape', x_test_1_Behavior.shape)
    print('x_test_2_Behavior shape', x_test_2_Behavior.shape)


    # ———————————————————— 5 times —————————————————————
    CounterT = 0
    while CounterT < 1:
        # ———————————————————— define ————————————————————
        # input1 = Input(shape=(len(x_train_1_Attribute[0]),), name='input1')
        # x1 = Dense(64, activation='relu', activity_regularizer=regularizers.l2(0.001))(input1)
        # x1 = Dropout(rate=0.3)(x1)
        # ——输入1 miRNA
        input1 = Input(shape=(len(x_train_1_Attribute[0]),), name='input1')
        x1 = Dense(64, activation='relu', activity_regularizer=regularizers.l2(0.001))(input1)
        x1 = Dropout(rate=0.3)(x1)
        # ——输入2—— drug
        input2 = Input(shape=(len(x_train_2_Attribute[0]),), name='input2')
        x2 = Dense(64, activation='relu', activity_regularizer=regularizers.l2(0.001))(input2)
        x2 = Dropout(rate=0.3)(x2)
        # ——输入3—— miRNA
        #input3 = Input(shape=(len(x_train_1_Behavior[0]),), name='input3')
        #x3 = Dense(128, activation='relu', activity_regularizer=regularizers.l2(0.05))(input3)
        #x3 = Dropout(rate=0.3)(x3)
        input3 = Input(shape=(len(x_train_1_Behavior[0]),), name='input3')
        x3 = Dense(64, activation='relu', activity_regularizer=regularizers.l2(0.005))(input3)
        x3 = Dropout(rate=0.3)(x3)
        # ——输入4—— drug
        #input4 = Input(shape=(len(x_train_2_Behavior[0]),), name='input4')
        #x4 = Dense(128, activation='relu', activity_regularizer=regularizers.l2(0.05))(input4)
        #x4 = Dropout(rate=0.3)(x4)
        input4 = Input(shape=(len(x_train_2_Behavior[0]),), name='input4')
        x4 = Dense(64, activation='relu', activity_regularizer=regularizers.l2(0.001))(input4)
        x4 = Dropout(rate=0.3)(x4)
        # ——连接——
        flatten = keras.layers.concatenate([x1, x2, x3, x4])
        # flatten = Reshape((256, 1))(flatten)
        # flatten = Reshape((256,))(flatten)
        # ——全连接——
        # hidden = Dense(32, activation='relu', name='hidden2', activity_regularizer=regularizers.l2(0.001))(flatten)
        # hidden = Dropout(rate=0.3)(hidden)
        hidden = Dense(32, activation='relu', name='hidden2', activity_regularizer=regularizers.l2(0.001))(flatten)
        hidden = Dropout(rate=0.3)(hidden)
        output = Dense(num_classes, activation='softmax', name='output')(hidden)  # category
        model = Model(inputs=[input1, input2, input3, input4], outputs=output)
        # 打印网络结构
        model.summary()
        # ——编译——
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        # —————————————————————— train ——————————————————————
        reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', patience=10, mode='auto')  # Automatically adjust the learning rate
        history = model.fit({'input1': x_train_1_Attribute, 'input2': x_train_2_Attribute, 'input3': x_train_1_Behavior, 'input4': x_train_2_Behavior}, y_train,
                            validation_data=({'input1': x_validation_1_Attribute, 'input2': x_validation_2_Attribute,
                                              'input3': x_validation_1_Behavior, 'input4': x_validation_2_Behavior}, y_validation),
                            callbacks=[reduce_lr],
                            epochs=60, batch_size=128,   #epochs=50, batch_size=128,
                            )
        # —————————————————————— 训练模型 ——————————————————————

        # StorFile(MyChange(history.history['val_loss']), 'Val_Loss.csv')
        # StorFile(MyChange(history.history['val_accuracy']), 'Val_ACC.csv')
        # StorFile(MyChange(history.history['loss']), 'Loss.csv')
        # StorFile(MyChange(history.history['acc']), 'ACC.csv')

        model.save(r'my_model' + str(CounterT) + '.h5')  # 保存模型

        # 输出预测值
        ModelTest = Model(inputs=model.input, outputs=model.get_layer('output').output)
        ModelTestOutput = ModelTest.predict([x_test_1_Attribute, x_test_2_Attribute, x_test_1_Behavior, x_test_2_Behavior])
        print(ModelTestOutput.shape)
        print(type(ModelTestOutput))
        #StorFile(ModelTestOutput, 'csstd all result.csv')
        # 输出值为label、1的概率
        LabelPredictionProb = []

        for counter in range(len(ModelTestOutput)):
            rowProb = []
            rowProb.append(ModelTestOutput[counter][1])
            LabelPredictionProb.append(rowProb)

        pd.DataFrame(LabelPredictionProb).to_csv(r'casestudy result/csstd all result.csv', header=None, index=None)

        CounterT = CounterT + 1