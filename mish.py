import numpy as np
def f(x):
    return ((x+2) * np.exp(x) + 2 * np.exp(2*x)) * np.cosh(np.log(1+np.exp(x)))-2 * x * np.exp(2*x) * np.sinh(np.log(1+np.exp(x)))

def binary_search(f, a, b, epsilon=5e-16):
    while np.abs(b - a) > epsilon:
        c = (a + b) / 2
        if f(c) == 0: return c
        elif f(a) * f(c) < 0: b = c
        else: a = c
    return (a + b) / 2

def mish(x):
    return x * np.tanh(np.log(1 + np.exp(x)))

def poly(begin=-11, end=8, point1=1.4905711794854284, point2=-2.2563763963607935 - 0.05):
    poly_degree2 = 7
    poly_degree3 = 7
    poly_degree4 = 7
    avg_err = 0

    x1 = np.linspace(-16, begin, 1000)
    y1 = mish(x1)
    avg_err1 = 0
    for j in range(len(y1)):
        err = 0 - y1[j]
        avg_err1 += err
    avg_err1 /= len(y1)
    avg_err += avg_err1
    print("average error in poly1 =", avg_err1)

    x2 = np.linspace(begin, point2, 1000)
    y2 = mish(x2)
    avg_err2 = 0
    p2 = np.polyfit(x2, y2, poly_degree2)
    poly2 = np.poly1d(p2)
    y2_fit = poly2(x2)
    print("poly2: ", end = "")
    for i in p2:
        print("%.10f" % (i), end = " ")
    print()
    for j in range(len(y2)):
        err = np.abs(y2[j] - y2_fit[j])
        avg_err2 += err
    avg_err2 /= len(y2)
    avg_err += avg_err2
    print("average error in poly2 =", avg_err2)

    x3 = np.linspace(point2, point1, 1000)
    y3 = mish(x3)
    avg_err3 = 0
    p3 = np.polyfit(x3, y3, poly_degree3)
    poly3 = np.poly1d(p3)
    y3_fit = poly3(x3)
    print("poly3: ", end = "")
    for i in p3:
        print("%.10f" % (i), end = " ")
    print()
    for j in range(len(y3)):
        err = np.abs(y3[j] - y3_fit[j])
        avg_err3 += err
    avg_err3 /= len(y3)
    avg_err += avg_err3
    print("average error in poly3 =", avg_err3)

    x4 = np.linspace(point1, end, 1000)
    y4 = mish(x4)
    avg_err4 = 0
    p4 = np.polyfit(x4, y4, poly_degree4)
    poly4 = np.poly1d(p4)
    y4_fit = poly4(x4)
    print("poly4: ", end = "")
    for i in p4:
        print("%.10f" % (i), end = " ")
    print()
    for j in range(len(y4)):
        err = np.abs(y4[j] - y4_fit[j])
        avg_err4 += err
    avg_err4 /= len(y4)
    avg_err += avg_err4
    print("average error in poly4 =", avg_err4)

    x5 = np.linspace(end, 16, 1000)
    y5 = mish(x5)
    avg_err5 = 0
    for j in range(len(y5)):
        err = np.abs(x5[j] - y5[j])
        avg_err5 += err
    avg_err4 /= len(y5)
    avg_err += avg_err5
    print("average error in poly5 =", avg_err5)

    print("average error =", avg_err)
    return avg_err

# start = 0
# end = 3
# a = []
# for index, value in enumerate(np.arange(start, end, 0.01)):
#     # print("Index:", index, " Value:", value)
#     a.append(poly(end=(7 - value))) # The best point for seg!

# if a:
#     min_value = min(a)
#     min_index = a.index(min_value)
#     min_value_corresponding = np.arange(start, end, 0.01)[min_index]

#     print("Min_value:", min_value)
#     print("Value corresponding to Min_value:", min_value_corresponding)
poly()

"""
poly3
Min_value: 0.00010084441579381256
Value corresponding to Min_value: 0.05
"""