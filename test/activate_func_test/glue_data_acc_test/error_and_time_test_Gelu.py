import numpy as np
import  random as rd
import time

def gelu(x, print_flag = False):

    res = 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))

    return  res

def bumbble_gelu(x, scale):   
    res = 0
    seg4gelu = []
    for x in x:
        if x <= -5.:
            res = - 10e-5
            seg4gelu.append(round(res, scale))
        if -5 < x and x <= -1.97:
            res = -0.5054031199708174 - 0.4222658115198386 * x  - 0.1180761295118195 * x**2 - 0.0110341340306157 * x ** 3  
            seg4gelu.append(round(res, scale))
        if -1.97 < x and x <= 3:
            res =  0.0085263215410380 +  0.5 * x  +  0.360329269278962 * x**2 + .0 * x ** 3  -0.037688200365904 * x ** 4 + 0.0018067462606141  *x **6
            seg4gelu.append(round(res, scale))
        if x > 3:
            res = x - 10e-5
            seg4gelu.append(round(res, scale))

    return seg4gelu

def seg5_gelu(x, scale):

    a1 = -0.568686678
    a2 = 0.529288810
    a3 = 0.183509590
    a4 = 0.028070202
    a5 = 0.001597741

    b1 = 0.001193207   
    b2 = 0.5 
    b3 = 0.385858026
    b4 = 0
    b5 = -0.045101361

    c1 = -0.438406187 
    c2 = 1.340789252
    c3 = - 0.087184212 
    c4 = 0.007334718


    # x  =      x /  2 ** 12
    seg5gelu = []
    res = 0

    for i in x:
        if i <= -5.075:
            seg5gelu.append(0.000002)
        if -5.075 < i and i <= -1.414: # poor
            res = -0.568686678 -  0.529288810 * i  - 0.183509590* i**2 - 0.028070202 * i ** 3  -0.001597741 * i ** 4
            seg5gelu.append(round(res, scale))
        if -1.414 < i and i < 1.414: # good! 
            res =   0.001193207 +  0.5 * i  + 0.385858026* i**2 + .0 * i ** 3  -0.045101361 * i ** 4
            seg5gelu.append(round(res, scale))
        if 1.414 <= i and i < 5.075: # bad
            res = -0.438406187 + 1.340789252 * i - 0.087184212 * i ** 2 + 0.007334718 * i ** 3
            seg5gelu.append( round(res, scale) )
        if i >= 5.075:
            res =  i + 0.000002
            seg5gelu.append(round(res, scale))
    return seg5gelu


def bolt_gelu2(x, scale):
    c1 = 0.14439048359960427
    c2 = 0.7077117131613893
    c3 = 4.5702822654246535
    c4 = 8.15444702051307
    c5 = 16.382265425072532
    
    abs_x = np.abs(x)
    bolt_gelu_res = []
    res = []
    
    for i in len(abs_x):
        if np.abs(i) <= 2.7:

            res = 0.020848611754127593 * i ** 4 - 0.183525061270 * i ** 3 + 0.5410550166368381 * i ** 2 - 0.03798164612714 * i + 0.0016208085 
            if i > 2.7:
                res = np.round(x, scale)
                print("res2 ",res)
            if i < -2.7:
                res = 0
                print("res3 ",res)
            bolt_gelu_res.append( np.round(res, scale) )

    return bolt_gelu_res

def bolt_gelu(x, print_flag=False):
    # x = truncate(x)

    # print(np.floor(softmax_input[4] + input_mask * 2**12)[8])
    # if print_flag:
    #     print(x)
    #     np.save('/home/qipang/mnt/d2/trash/compare/np.npy', x)

    c1 = 0.14439048359960427
    c2 = 0.7077117131613893
    c3 = 4.5702822654246535
    c4 = 8.15444702051307
    c5 = 16.382265425072532

    c1 = np.floor(c1 * 2 ** 12)
    c2 = np.floor(c2 * 2 ** 12)
    c3 = np.floor(c3 * 2 ** 12)
    c4 = np.floor(c4 * 2 ** 12)
    c5 = np.floor(c5 * 2 ** 12)

    abs_x = np.abs(x)
    # y = truncate((truncate(c1 * abs_x, bits=12) - c2) * abs_x, bits=12) + c3
    # res = truncate((y + truncate(c1 * abs_x, bits=12) - c4) * y, bits=12) + c5 + 0.5 * x
    # res = truncate(res, bits=12)
    # res[x > 2.7] = x[x > 2.7]
    # res[x < -2.7] = 0

    temp_y = np.floor(c1 * abs_x / 2 ** 12) - c2
    y = np.floor(temp_y * abs_x / 2 ** 12) + c3
    temp_res = y + np.floor(c1 * abs_x / 2 ** 12) - c4
    temp_res = temp_res * y
    res = np.floor(temp_res / 2 ** 12) + c5 + x / 2


    res[x > np.floor(2.7 * 2 ** 12)] = x[x > np.floor(2.7 * 2 ** 12)]
    res[x < np.floor(-2.7 * 2 ** 12)] = 0
    return res



def truncate(x, bits=12):
    return np.floor(x * 2 ** bits) / 2 ** bits



def test_error():
    iter_time = 10000 # vrctor lenght 
    scale = 5
    a = np.random.rand(iter_time)
    # print(a)
    time1 = time.time()
    true_y = gelu(a)
    time2 = time.time()  - time1
    # print("true res: ", true_y)
    # print("true res time: ", time2)
    
    
    time3 = time.time()
    bumbble_res = bumbble_gelu(a, scale)
    time4 = time.time() - time3
    # print("bumbble_res: ", bumbble_res)
    # print("bumbble_res time: ", time4)
    
    time7 = time.time()
    seg_5_res = seg5_gelu(a, scale)
    time8 = time.time() - time7
    # print("seg_5_res: ", seg_5_res)
    # print("seg_g_res time: ", time8)
    
    
    
    # time9 = time.time()
    # bolt_gelu2_res = bolt_gelu2(a, scale)
    # time10 = time.time() - time9
    # print("bolt_gelu2_res: ", bolt_gelu2_res)
    # print("bolt_gelu2_res time: ", time10)
    
    a = np.floor(a * 2 ** 12)
    time5 = time.time()
    Bbolt_res = bolt_gelu(a) / 2 ** 12
    time6 = time.time() - time5
    # print("bolt_res: ", Bbolt_res)
    # print("bolt res time: ", time6)
        
        
    seg5res =     sum(true_y - seg_5_res) / iter_time
        
    bumbbleres  = sum(true_y - bumbble_res) / iter_time
    boltres = sum(true_y - Bbolt_res) / iter_time
    
    # print("mean of error seg5Gelu: ", seg5res)

    # print("mean of error bumbbleGelu: ", bumbbleres)

    # print("mean of error2 boltGelu: ", boltres)
    return seg5res, bumbbleres, boltres

if __name__ == "__main__":
    seg5res = 0 
    bumbbleres = 0
    boltres = 0
    times = 10
    a = 0
    b = 0
    c = 0
    for i in range(times):
        a, b, c = test_error()
        seg5res += a 
        bumbbleres += b
        boltres += c
        
    meanerror_seg5 =     seg5res / times
    meanerror_bumbble =     bumbbleres / times
    meanerror_bolt =     boltres / times

    print("meanerror_seg5: ", meanerror_seg5)
    print("meanerror_bumbble: ", meanerror_bumbble)
    print("meanerror_bolt: ", meanerror_bolt)