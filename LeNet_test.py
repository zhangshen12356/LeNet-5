import torch
import torch.nn as nn
from LeNET_5 import LeNet
import pandas as pd
import numpy as np

model = LeNet()
test_data = pd.DataFrame(pd.read_csv(r'C:\Users\zs\Desktop\python_practice\classical_model\data\mnist_test.csv'))
model.load_state_dict(torch.load('LeNet.pkl'))  # 加载模型参数
with torch.no_grad():  # 测试不需要反向传播
    batch_data = test_data.sample(n=50, replace=False)
    batch_x = torch.from_numpy(batch_data.iloc[:, 1::].values).float().view(-1, 1, 28, 28)
    batch_y = batch_data.iloc[:, 0].values
    prediction = np.argmax(model(batch_x).numpy(), axis=1)  # 在pytorch中.numpy()的意思是将tensor转化为numpy
    print(prediction)
    for i in range(100):
        batch_data = test_data.sample(n=50, replace=False)
        batch_x = torch.from_numpy(batch_data.iloc[:,1::].values).float().view(-1,1,28,28)
        batch_y = batch_data.iloc[:, 0].values
        prediction = np.argmax(model(batch_x).numpy(),axis=1)
        acccurcy = np.mean(prediction == batch_y)
    print("第%d组测试集，准确率为%.3f" % (i, acccurcy))