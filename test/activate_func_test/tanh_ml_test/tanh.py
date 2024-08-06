import numpy as np
from numpy import tanh


def poly(split1, NotMin=True):

    split2 = 4.6
    end = 20
    x1 = np.linspace(0, split1, 1000)
    x1_neg = -x1
    x2 = np.linspace(split1, split2, 1000)
    x2_neg = -x2
    # x3 = np.linspace(split2, end, 1000)
    # x3_neg = -x3
    y1 = tanh(x1)
    y1_neg = tanh(x1_neg)
    y2 = tanh(x2)
    y2_neg = tanh(x2_neg)
    poly_degree1 = 4
    poly_degree2 = 6
    p1 = np.polyfit(x1, y1, poly_degree1)
    p2 = np.polyfit(x2, y2, poly_degree2)
    poly1 = np.poly1d(p1)
    poly2 = np.poly1d(p2)
    y1_fit = poly1(x1)
    y1_neg_fit = -y1_fit
    y2_fit = poly2(x2)
    y2_neg_fit = -y2_fit

    avg_err = 0
    for j in range(len(y1)):
        err = np.abs(y1[j] - y1_fit[j])
        avg_err += err
    for j in range(len(y1_neg)):
        err = np.abs(y1_neg[j] - y1_neg_fit[j])
        avg_err += err
    for j in range(len(y2)):
        err = np.abs(y2[j] - y2_fit[j])
        avg_err += err
    for j in range(len(y2_neg)):
        err = np.abs(y2_neg[j] - y2_neg_fit[j])
        avg_err += err
    # for j in range(len(x3)):
    #     err = 1 - sigmoid(x3[j])
    #     avg_err += err
    # for j in range(len(x3_neg)):
    #     err = sigmoid(x3_neg[j])
    #     avg_err += err
    
    ret  = avg_err / (len(y1) + len(y1_neg) + len(y2) + len(y2_neg))
    if NotMin:
        print("split1 =", split1)
        print("poly1: ", end = "")
        for i in p1:
            print("%.10f" % (i), end = " ")
        print("\npoly2: ", end = "")
        for i in p2:
            print("%.10f" % (i), end = " ")
        print()
        print("average error =", ret)

    return ret
poly(1)
poly(2)
poly(3)
poly(4)
# sigmoid: ln((sqrt(3)±1) /sqrt(2)) ≈ 0.6584789485
poly(np.log((1 + np.sqrt(3) + 1) / np.sqrt(2)))
poly(np.log((1 + np.sqrt(3) + 0.94) / np.sqrt(2)))
# window
a = []
start = 0
end = 4
for index, value in enumerate(np.arange(start, end, 0.01)):
    # print("Index:", index, " Value:", value)
    a.append(poly(np.log((1 + np.sqrt(3) + value) / np.sqrt(2)), False)) # The best point for seg!

if a:
    min_value = min(a)
    min_index = a.index(min_value)
    min_value_corresponding = np.arange(start, end, 0.01)[min_index]

    print("Min_value:", min_value)
    print("Value corresponding to Min_value:", min_value_corresponding)