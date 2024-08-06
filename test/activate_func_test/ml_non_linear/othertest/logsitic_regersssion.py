
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.utils import _array_api

import numpy as np
import math
from matplotlib import pyplot as plt

# In[2]:


# def sigmoid(z):
#     return 1 / (1 + np.exp(-z))


def sigmoid(z):
    return np.divide(1, 1 + np.exp(-np.clip(z, -500, 500)))


def our_sigmoid(x):
    return 


def bolt_sigmoid(x):
    return np.divide(1, 1 + (0.385*((-x)+ 1.353)**2 + 0.344))

def sigmoid_talor(z, terms=13):
    result = 0
    for n in range(terms):
        result += z**n / math.factorial(n)
    return 1 / (1 + result)

# In[ ]:

def error_test():
    np.set_printoptions(suppress=True, precision=4)
    x = np.linspace(-5, 5, 100)
    error1 = abs(bolt_sigmoid(x) - sigmoid(x))
    print("average: ", sum(error1)/ 100)

    error2 = abs(sigmoid_talor(x) - sigmoid(x))
    print("average: ", sum(error2)/100)



def indexing_dtype(xp):
    """Return a platform-specific integer dtype suitable for indexing.
    On 32-bit platforms, this will typically return int32 and int64 otherwise.

    """

    return xp.asarray(0).dtype

class CustomLogisticRegression(LogisticRegression):

    def custom_predict(self, X):
        ret =np.zeros(X.shape[0])
        predictions = sigmoid(X)

        for i, x in  enumerate(predictions):
            ret[i] = np.argmax(x)
        return ret
    

    def predict_one_vs_all(theta, x):
        """
        :param theta: 权值（优化后的）
        :param x: 测试变量
        :return: 预测值
        """
        m = x.shape[0]
        p = np.zeros((m, 1))
        cal = sigmoid(x @ theta)
        p = np.argmax(cal, axis=1)
        p[p == 0] = 10
        return p






 
# 加载乳腺癌数据集
breast_cancer = load_breast_cancer()
 
# 获取X特征向量数据以及y标签向量数据
cancer_x = breast_cancer.data
cancer_y = breast_cancer.target
 
# 划分训练集和测试集
# x_train, x_test, y_train, y_test = train_test_split(cancer_x,
#                                                     cancer_y,
#                                                     test_size=0.2)
# 训练逻辑回归模型
# log = LogisticRegression(solver='liblinear')
log = CustomLogisticRegression(solver='liblinear')
log.fit(cancer_x, cancer_y)


# y_predict = log.custom_predict(x_test)

y_predict = log.custom_predict(cancer_x)




count = 0
L = len(cancer_y)
print(f'Lenght of data: {L}')
for i in range(L):
    if log.custom_predict(cancer_x)[i] == cancer_y[i]:
    # if log.predict(x_test)[i] == y_test[i]:
        count += 1
print(f"correct number: {count}")
rate = (count / L) * 100

print("acc: %.2f%%" % rate)