import numpy as np
 
def sigmoid(x):
    return 1 / (np.exp(-x) + 1)

def d_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

def poly(split1):
    print("split1 =", split1)
    split2 = 6.480
    end = 20
    x1 = np.linspace(0, split1, 1000)
    x1_neg = -x1
    x2 = np.linspace(split1, split2, 1000)
    x2_neg = -x2
    # x3 = np.linspace(split2, end, 1000)
    # x3_neg = -x3
    y1 = sigmoid(x1)
    y1_neg = sigmoid(x1_neg)
    y2 = sigmoid(x2)
    y2_neg = sigmoid(x2_neg)
    poly_degree1 = 4
    poly_degree2 = 7
    p1 = np.polyfit(x1, y1, poly_degree1)
    p2 = np.polyfit(x2, y2, poly_degree2)
    poly1 = np.poly1d(p1)
    poly2 = np.poly1d(p2)
    y1_fit = poly1(x1)
    y1_neg_fit = 1 - poly1(x1)
    y2_fit = poly2(x2)
    y2_neg_fit = 1 - poly2(x2)

    print("poly1: ", end = "")
    for i in p1:
        print("%.10f" % (i), end = " ")
    print("\npoly2: ", end = "")
    for i in p2:
        print("%.10f" % (i), end = " ")
    print()
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
    print("average error =", avg_err / (len(y1) + len(y1_neg) + len(y2) + len(y2_neg)))

poly(1)
poly(2)
poly(3)
poly(4)
poly(5)
poly(np.log(2+np.sqrt(3)))