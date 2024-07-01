关于安全性证明的一些问题：
1.调研安全性证明（或者同态秘密共享论文的安全性证明）
A,secureTrain； B,FIT



## ckks的安全性
对于CKKS而言，其运算域实际上在整数域，其没有包含plaintext modulus，密文的数值空间与明文的数值空间一致，令其为Q，对于一个长度为n的明文vector，其所属空间为$Ring_Q^n$。首先对其进行encoding形成多项式，其次对多项式进行加密。形成密文$（A，B) \in (Ring_Q^n)^2 $。在microsoft SEAL库中，Q是多个prime的乘积，表示为$Q=q_{l-1}q_{l-2}...q_0$，最后的运算结果密文通过modulus switching使空间域变为$(Ring_{q_0}^n)^2$。因此在share此密文结果时，每一个share，令其为$y_s$，的抽样空间为$Z_{q_0}$，而解密的share，令其为$y_c$，所属空间为$Z_{Q}$，其满足$y_s +  y_c= y mod Q$，其中，y是desired output。
你反馈的问题非常好。目前的实现中确实没有反应这个性质，我们已将此问题进行标记，以便之后的开发。此外，个人推荐使用机器学习隐私保护计算开发工具CrypTFlow2:github: https://github.com/mpc-msri/EzPC/tree/master/SCI），
paper: https://eprint.iacr.org/2020/1002.pdf）。 

特别地，其对非线性函数进行了重要改进，使得性能显著提高。最后，个人认为目前基于transformer的大模型训练与预测是一个具有挑战性但意义重大的课（paper: https://www.computer.org/csdl/proceedingsarticle/sp/2024/313000a130/1Ub23O2X00U）

## softmax的安全性
由于softmax函数计算利用了multiplicative sectre sharing而非additive sectre sharing，我们需要$r/d_c$是可逆的，即存在$inv(r/d_c) \in Z_Q$，使得$r/d_c \times inv(r/d_c) = 1 mod Q$。此处$d_c \in Z_Q$，即$inv(d_c)=1/d_c$存在。给定$r \in Z_Q，r/d_c = r \times inv(d_c) ~mod ~Q$。其他涉及inverse的计算都需要满足类似的性质以形成random share(s)。
更多关于指数等浮点数运算可参考：

SecFloat: Accurate Floating-Point meets Secure 2-Party Computation
Deevashwer Rathee, Anwesh Bhattacharya, Rahul Sharma, Divya Gupta, Nishanth Chandran, Aseem Rastogi
IEEE S&P 2022

SIRNN: A Math Library for Secure RNN Inference
Deevashwer Rathee, Mayank Rathee, Rahul Kranti Kiran Goli, Divya Gupta, Rahul Sharma, Nishanth Chandran, Aseem Rastogi
IEEE S&P 2021


## 关于$[m]_C - shr$的计算domain的处理。

Hello同学你好，这是一个很好的问题，针对于不同的bit位，可以有相应的MPC的protocol，比如有针对$\ell$ bit的MPC in the ring，也有针对于prime 对应bit的MPC in the filed（详见CrypTFlow2针对$\ell$ bit以及任意space including prime number的millionaires’ protocol），MPC可选择$\ell$ bit，则在后续HE计算之前需要进行domain conversion,也可选择针对任意space的MPC以直接兼容后续计算。针对加密函数的share,在prime space 中直接产生并与加密结果相加，把另一个share设为prime modulus减去产生的share,当然所有结果都是在prime modulus 中的，具体的sharing细节可参考CrypTFlow2的实现。






