import numpy as np
import matplotlib.pyplot as plt


def gelu(x):
    res = 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))
    return res


def bumbble_gelu(x, scale):
    res = 0
    seg4gelu = []
    for x in x:
        if x <= -5.0:
            res = -10e-5
            seg4gelu.append(round(res, scale))
        if -5 < x and x <= -1.97:
            res = -0.5054031199708174 - 0.4222658115198386 * x - 0.1180761295118195 * x**2 - 0.0110341340306157 * x**3
            seg4gelu.append(round(res, scale))
        if -1.97 < x and x <= 3:
            res = (
                0.0085263215410380
                + 0.5 * x
                + 0.360329269278962 * x**2
                + 0.0 * x**3
                - 0.037688200365904 * x**4
                + 0.0018067462606141 * x**6
            )
            seg4gelu.append(round(res, scale))
        if x > 3:
            res = x - 10e-5
            seg4gelu.append(round(res, scale))

    return seg4gelu


def seg5_gelu(x, scale):
    seg5gelu = []
    res = 0
    for i in x:
        if i <= -5.075:
            seg5gelu.append(round(0.00001,scale))
        elif -5.075 < i and i <= -1.414:  # poor
            res = -0.568686678 - 0.529288810 * i - 0.183509590 * i**2 - 0.028070202 * i**3 - 0.001597741 * i**4
            seg5gelu.append(round(res, scale))
        elif -1.414 < i and i < 1.414:  # good!
            res = 0.001193207 + 0.5 * i + 0.385858026 * i**2 + 0.0 * i**3 - 0.045101361 * i**4
            seg5gelu.append(round(res, scale))
        elif 1.414 <= i and i < 5.075:  # bad
            res = -0.438406187 + 1.340789252 * i - 0.087184212 * i**2 + 0.007334718 * i**3
            seg5gelu.append(round(res, scale))
        elif i >= 5.075:
            res = i + 0.00001
            seg5gelu.append(round(res, scale))
    return seg5gelu


def bolt_gelu2(x, scale):
    abs_x = np.abs(x)
    bolt_gelu_res = []
    res = []

    for i in len(abs_x): # type: ignore
        if np.abs(i) <= 2.7:

            res = (
                0.020848611754127593 * i**4
                - 0.183525061270 * i**3
                + 0.5410550166368381 * i**2
                - 0.03798164612714 * i
                + 0.0016208085
            )
            if i > 2.7:
                res = np.round(x, scale)
                print("res2 ", res)
            if i < -2.7:
                res = 0
                print("res3 ", res)
            bolt_gelu_res.append(np.round(res, scale))

    return bolt_gelu_res


def bolt_gelu(x):
    x = np.floor(x * 2**12)
    c1 = 0.14439048359960427
    c2 = 0.7077117131613893
    c3 = 4.5702822654246535
    c4 = 8.15444702051307
    c5 = 16.382265425072532

    c1 = np.floor(c1 * 2**12)
    c2 = np.floor(c2 * 2**12)
    c3 = np.floor(c3 * 2**12)
    c4 = np.floor(c4 * 2**12)
    c5 = np.floor(c5 * 2**12)

    abs_x = np.abs(x)

    temp_y = np.floor(c1 * abs_x / 2**12) - c2
    y = np.floor(temp_y * abs_x / 2**12) + c3
    temp_res = y + np.floor(c1 * abs_x / 2**12) - c4
    temp_res = temp_res * y
    res = np.floor(temp_res / 2**12) + c5 + x / 2

    res[x > np.floor(2.7 * 2**12)] = x[x > np.floor(2.7 * 2**12)]
    res[x < np.floor(-2.7 * 2**12)] = 0
    return np.array(res) / 2**12


def test_error():
    iter_time = 10000  # vrctor lenght
    scale = 5
    a = np.random.rand(iter_time)
    true_y = gelu(a)
    bumbble_res = bumbble_gelu(a, scale)
    seg_5_res = seg5_gelu(a, scale)
    Bbolt_res = bolt_gelu(a)

    seg5res = sum(true_y - seg_5_res) / iter_time
    bumbbleres = sum(true_y - bumbble_res) / iter_time
    boltres = sum(true_y - Bbolt_res) / iter_time

    return abs(seg5res), abs(bumbbleres), abs(boltres)


if __name__ == "__main__":
    plt.rcParams["font.family"] = "Times New Roman"

    granularity = 10000
    BOLT_granularity = 10
    scale = 12

    blocks = [(-6.0, -4.0), (-4.6, -2.25), (-2.5, 0.5), (0.0, 2.0), (2.0, 3.5), (-6.0, 6.0)]

    for a, b in blocks:
        assert a <= b
        x = np.array(range(int(a * granularity), int(b * granularity))) / granularity
        y = np.array(range(int(a * BOLT_granularity), int(b * BOLT_granularity))) / BOLT_granularity
        plt.clf()
        # plt.title("test")
        plt.plot(x, gelu(x), "-", color="black", label="Base")
        plt.plot(x, seg5_gelu(x, scale), "r--", label="Ours")
        plt.plot(x, bumbble_gelu(x, scale), "g--", label="Bumbble")

        # if -2.7 < a and b < 2.7:
        #     y = np.array(range(int(a * BOLT_granularity), int(b * BOLT_granularity))) / granularity
        #     plt.plot(y, bolt_gelu(y), "b--", label="BOLT")
        # elif a < -2.7 and 2.7 < b:
        #     y = np.hstack(
        #         (
        #             np.array(range(int(a * granularity), int(-2.7 * granularity))) / granularity,
        #             np.array(range(int(-2.7 * BOLT_granularity), int(2.7 * BOLT_granularity))) / BOLT_granularity,
        #             np.array(range(int(2.7 * BOLT_granularity), int(b * BOLT_granularity))) / BOLT_granularity,
        #         )
        #     )
        #     plt.plot(y, bolt_gelu(y), "b--", label="BOLT")
        # elif a < -2.7 and b < 2.7:
        #     y = np.append(
        #         np.array(range(int(a * granularity), int(-2.7 * granularity))) / granularity,
        #         np.array(range(int(-2.7 * BOLT_granularity), int(b * BOLT_granularity))) / BOLT_granularity,
        #     )
        #     plt.plot(y, bolt_gelu(y), "b--", label="BOLT")
        # elif -2.7 < a and 2.7 < b:
        #     y = np.append(
        #         np.array(range(int(a * BOLT_granularity), int(2.7 * BOLT_granularity))) / BOLT_granularity,
        #         np.array(range(int(2.7 * granularity), int(b * granularity))) / granularity,
        #     )
        # else:
        plt.plot(y, bolt_gelu(y), "b--", label="BOLT")



        legend = plt.legend(fontsize=16)
        for text in legend.get_texts():
            text.set_fontweight('bold')
        plt.xticks(fontsize=17)
        plt.yticks(fontsize=17)
        plt.subplots_adjust(left=0.17, right=0.9, top=0.9, bottom=0.12)
        plt.ylabel("Output", fontsize=18, fontweight='bold')
        plt.xlabel("Input", fontsize=18, fontweight='bold')
        # plt.show()

        plt.savefig(f"{a}_to_{b}_fig.pdf")

    blocks = [(-6.0, 6.0)]

    for a, b in blocks:
        assert a <= b
        x = np.array(range(int(a * granularity), int(b * granularity))) / granularity

        plt.clf()
        # plt.title("test")
        seg5_err = gelu(x) - seg5_gelu(x, scale)
        BOLT_err = gelu(x) - bolt_gelu(x)
        bumbble_err = gelu(x) - bumbble_gelu(x, scale)


        plt.plot(
            x,
            abs(seg5_err),
            "r-",
            label="Ours(avg err: %.2f × $\mathdefault{10^{-5}}$)" % abs(np.mean(seg5_err) * 10000)
        )
        if -2.7 < a and b < 2.7:
            y = np.array(range(int(a * BOLT_granularity), int(b * BOLT_granularity))) / granularity
            plt.plot(
                y,
                abs(gelu(y) - bolt_gelu(y)),
                "b-",
                label="BOLT(avg err: %.2f × $\mathdefault{10^{-5}}$)" % abs(np.mean(BOLT_err) * 10000)
            )
        elif a < -2.7 and 2.7 < b:
            y = np.hstack(
                (
                    np.array(range(int(a * granularity), int(-2.7 * granularity))) / granularity,
                    np.array(range(int(-2.7 * BOLT_granularity), int(2.7 * BOLT_granularity))) / BOLT_granularity,
                    np.array(range(int(2.7 * BOLT_granularity), int(b * BOLT_granularity))) / BOLT_granularity,
                )
            )
            plt.plot(
                y,
                abs(gelu(y) - bolt_gelu(y)),
                "b-",
                label="BOLT(avg err: %.2f × $\mathdefault{10^{-5}}$)" % abs(np.mean(BOLT_err) * 10000),
            )
        elif a < -2.7 and b < 2.7:
            y = np.append(
                np.array(range(int(a * granularity), int(-2.7 * granularity))) / granularity,
                np.array(range(int(-2.7 * BOLT_granularity), int(b * BOLT_granularity))) / BOLT_granularity,
            )
            plt.plot(
                y,
                abs(gelu(y) - bolt_gelu(y)),
                "b-",
                label="BOLT(avg err: %.2f × $\mathdefault{10^{-5}}$)" % abs(np.mean(BOLT_err) * 10000),
            )
        elif -2.7 < a and 2.7 < b:
            y = np.append(
                np.array(range(int(a * BOLT_granularity), int(2.7 * BOLT_granularity))) / BOLT_granularity,
                np.array(range(int(2.7 * granularity), int(b * granularity))) / granularity,
            )
        else:
            plt.plot(
                x,
                abs(BOLT_err),
                "b-",
                label="BOLT(avg err: %.2f × $\mathdefault{10^{-5}}$)" % abs(np.mean(BOLT_err) * 10000),
            )
        plt.plot(
            x,
            abs(bumbble_err),
            "g-",
            label="Bumbble(avg err: %.2f × $\mathdefault{10^{-5}}$)" % abs(np.mean(bumbble_err) * 10000),
        )

        legend3 = plt.legend(fontsize=16)
        for text in legend3.get_texts():
            text.set_fontweight('bold')
        plt.xticks(fontsize=17)
        plt.yticks(fontsize=17)
        plt.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.12)
        plt.ylabel("Error", fontsize=18, fontweight='bold')
        plt.xlabel("Input", fontsize=18, fontweight='bold')
        # plt.show()
        plt.savefig(f"{a}_to_{b}_err_fig.pdf")
