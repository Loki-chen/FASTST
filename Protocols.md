


## $Secure\,Attention$



$Client$: 拥有$x^C$和$W^C_I$；

$Server$: 拥有$x^S$和$W^S_I$；

以上$I\in\{Q, K, V\}$，以下$[\cdot]_P$表示使用参与方$P$的密钥加密的同态密文。

***

过程：

<!-- ## $sub-protocols ~~1 (liner~~ 1)$ -->

$Client$：

1. 生成随机数$r^C$；

2. 计算$r^Cx^C\bigotimes W^C_I$，$r^Cx^C$，$r^CW^C_I$，$[r^C]_C$;

3. 将$H_1=\left\{r^Cx^C\bigotimes  W^C_I, r^Cx^C, r^CW^C_I, [r^C]_C\right\}$发送给$Server$。

$Server$：
1. 收到$H_1$，计算$r^Cx^C\bigotimes W^C_I+r^Cx^C\bigotimes W^S_I+r^CW^C_I\bigotimes x^S+[r^C]_C\cdot x^S\bigotimes W^S_I=[r^CI]_C$；
2. 生成随机数$r^S_1$，计算$[r^CQ/r^S_1]_C$，$[r^CK/r^S_1]_C$， $[r^CV/r^S_1]_C$，$[(r^S_1)^2]_S$；
3. 将$H_2=\left\{[r^CQ/r^S_1]_C, [r^CK/r^S_1]_C, [(r^S_1)^2]_S\right\}$发送给$Client$。

<!-- ## $sub-protocols ~~2 (softmax~~ $ &&  $liner 2)$ -->
$Client$：

1. 收到$H_2$，可得$Q/r^S_1$，$K/r^S_1$，$[(r^S_1)^2]_S$；
2. 计算$[Z]_S=\left[\frac{Q\bigotimes K^T}{\sqrt{d_K}}\right]_S =(Q/r^S_1)\bigotimes(K/r^S_1)^T\cdot[(r^S_1)^2]_S/\sqrt{d_K}$，生成随机矩阵$Z^C$，计算$[Z^S]_S=[Z]_S-Z^C$，$Z^C=Z^C-\max(Z^C)$，$[\exp(Z^C)]_C$；
3. 将$H_3=\left\{[Z^S]_S, [\exp(Z^C)]_C\right\}$发送给$Server$。

$Server$：

1. 收到$H_3$，可得$Z^S$，$[\exp(Z^C)]_C$，计算${Z}^S=Z^S-\max(Z^S)$，生成随机数$r^S_2$和随机矩阵$D^S$，$O$，满足$\sum_jO_{ij}=0$；
2. 计算$[r^S_2\exp(Z)+O]_C=r^S_2\exp(Z^S)\cdot[\exp(Z^C)]_C+O$，$D^Sexp(Z^S)$，生成随机矩阵$R^S$，计算$[r^CR^SV]_C$
3. 将$H_4=\left\{[r^S_2\exp(Z)+O]_C, D^S\exp(Z^S), [r^CR^SV]_C\right\}$发送给$Client$。

$Client$：

1. 收到$H_4$，可得$r^S_2\exp(Z)+O$，$D^S\exp(Z^S)$，$R^SV$，计算$r^S_2\sum_j\exp(Z_{ij})=\sum_j(r^S_2\exp(Z_{ij})+O{ij})$；
2. 计算$(D^S\bigotimes R^S)\exp(Z)\bigotimes V=(D^S\exp(Z^S)\cdot\exp(Z^C))\bigotimes R^SV$，$SoftMax(Z)\bigotimes V\cdot (D^S\bigotimes R^S)/r^S_2=(D^S\bigotimes R^S)\exp(Z)\bigotimes V/r^S_2\sum_j\exp(Z_{ij})$。

## 协议结果：

Client：拥有$Attn_C=SoftMax(Z)\bigotimes V\cdot (D^S\bigotimes R^S)/r^S_2$

Server: 拥有$Attn_S=r_2^S/(D^S\bigotimes R^S)$。
后续计算，Server加密自己的份额发送到客户端，进行element-wise的 ciphertext * plaintext; 得到$[Attn]_S$供客户端使用
***

### Rotate-Free $Ciphertext \bigotimes PlainText$ 矩阵乘

现有矩阵$A$和$B$，其中：
$$
A = 
\begin{pmatrix}
1&2&3&4\\
5&6&7&8\\
9&10&11&12
\end{pmatrix}，
B=
\begin{pmatrix}
a&b&c&d&e\\
f&g&h&i&j\\
k&l&m&n&o\\
p&q&r&s&t
\end{pmatrix}
$$
计算$[A]\cdot B$的方法如下：

1. 将$A$，$B$扩展成如下形状：

$$
\left(\frac AB\right)=
\begin{pmatrix}
1&1&1&1&1&5&5&5&5&5&9&9&9&9&9\\
2&2&2&2&2&6&6&6&6&6&10&10&10&10&10\\
3&3&3&3&3&7&7&7&7&7&11&11&11&11&11\\
4&4&4&4&4&8&8&8&8&8&12&12&12&12&12\\
\hline
a&b&c&d&e&a&b&c&d&e&a&b&c&d&e\\
f&g&h&i&j&f&g&h&i&j&f&g&h&i&j\\
k&l&m&n&o&k&l&m&n&o&k&l&m&n&o\\
p&q&r&s&t&p&q&r&s&t&p&q&r&s&t
\end{pmatrix}_{4\times(3\times5)}
$$

2. 按行相乘再相加，即为结果。





## $Secure ~~LayerNorm$ ($Add$ & $Norm$)

## Init phase :

$Client$: 拥有$[Attn]_S$和$x^C$；

$Server$: 拥有$x^S$,参数$\gamma$和$\beta$


***
过程：


>
>$Client$:
<!-- >1. 生成随机数$h^C_1, h^C_2$,得到$x^Ch^C_1$, $[h^C_2/h^C_1]_C$, $[h^C_2]_C$ 
>2. 计算$[Attn]_S h^C_2 $
>3. 发送集合$H_2$ = $x^Ch^C_1$,   $h^C_2/h^C_1$,   $[h^C_2]_C$， $[Attnh_2]_S$ 
>
>$Server$:
>1. 接收$H_2$
>2. 计算 $x^Sh^C_2$, $Attnh^C_2 + x^Ch^C_1 *  [h^C_2/h^C_1]_C + x^Sh^C_2$ = $[X_{add}h^C_2]_C$
> -->


>1. 生成随机数$h^C$,得到$x^Ch^C$, $[h^C]_C$ 
>2. 计算$[Attn]_S h^C_2 $
>3. 发送集合$H_2$ = $x^Ch^C$, $[h^C]_C$， $[Attnh^C]_S$ 
>
>$Server$:
>1. 接收$H_2$
>2. 计算 $x^Sh^C$, $Attnh^C + x^Ch^C + x^Sh^C$ = $[X_{add}h^C]_C$
>


<!-- >### 方案1(不share初始值)
>
>$Client$:
>1. 加密$[X]_C$, 生成随机数$h^C$
>2. 发送集合$[Xh^C]_C$,$[Attnh^C]_S$ 
>
>$Server$:
>1. 解密$[Attnh^C]_S$ ，得到$Attnh^C_i$, 根据head num，计算$Attnh^C = \sum_{headnum}{Attnh^C_i}$
>2. 计算$Attnh^C$ + $[Xh^C]_C$ ，得到 $[X_{add}h^C]_C$ -->



>3. 生成masked：$g^S$, 得到扰动结果$[X_{add}h^Cg^S]_C$
>4. 发送至Client




> ***关于LayerNorm协议的标注***
>此处有疑问： 关于$(X_{add}-u)g^S / \sqrt{\sigma^2(g^S)^2} = T$ 
> Client给$T$加密， 即 $[T]_C$, 然后S端进行相关计算即可，但问题是$T$是明文计算结果，虽不能分解还原，但使得计算结果暴漏在外，但这种其实很常见，因为参数和输入相辅相成，不可能脱离其中一个单独存在，况且这种中间结果很难推出前驱计算中的任何信息。调研目前的方案中，发现部分方法也有对部分中间数据在明文状态下使用，但是T本身就应该是属于C端得中间输出，因此此处依然保持安全性！


$Client: $
1. 接收$[X_{add}h^Cg^S]_C$, 生成随机数$[k_1^C]_C$, $k_2^C$ ,$[k_2^C/ k_1^C]_C$； 处理$[X_{add}h^Cg^S]_C$，得到$X_{add}g^S$
2. 计算$\sum_{d_k}(X_{add}g^S)$ = $\mu g^S$; 计算$\sum_{d_k}(X_{add}g^S - \mu g^S)^2$ = $\sum_{d_k}\{(X_{add}- \mu )g^S\}^2$ = $(g^S)^2 \sigma^2$
3. 计算$(X_{add} - u)g^S k^C_2 = tmp_1$ , $((g^S)^2 \sigma^2)^{-1/2} \cdot [1/k^C_2]_C = [tmp_2]_C$  == $H_3$, 发送到Server

$Server:$
1. 接收$H_3$, 根据拥有的 $\gamma$ 和 $\beta$计算layerNorm，
2. 计算$[Lanorm]_C$ = $ \gamma \cdot tmp1 \cdot [tmp_2]_C$   + $\beta$ 

<!-- # <mark>there are some ambiguity!</mark> -->

<!-- >### 方案1



$Client:$
1. 接收$[X_{add}h^Cg^S]_C$, 处理得到$X_{add}g^S$
2. 计算$\sum_{d_k}(X_{add}g^S)$ = $\mu g^S$; 计算 $(X_{add}- \mu)g^S$
计算$\sum_{d_k}(X_{add}g^S - \mu g^S)^2$ = $\sum_{d_k}\{(X_{add}- \mu )g^S\}^2$ = $\sum(g^S)^2 \sigma^2 = (g^S)^2 \sum \sigma^2$
3. 计算$\frac{(X_{add}- \mu)g^S}{ g^S\sqrt(\sum\sigma^2)} = \frac{X_{add} - \mu} {\sigma^{-1/2}} = T $ 
4. 加密 $T$， 得到$[T]_C$
5. 发送 $[T]_C$到Server -->


 
 

***


## $Secure  ~FeedForword~~(Linear3 + Gelu + Linear4)$ 

### Gelu函数的等价替代：

- GeLU： $\text{GeLU}(x) = \frac{1}{2}x\left(1 + \tanh\left(\sqrt{\frac{2}{\pi}}\left(x + 0.044715x^3\right)\right)\right) $

- seg5GeLU 
$$ seg5GeLU (x)=\left\{
\begin{array}{rcl}
-\epsilon         &      & {x  <  -5.075}\\
F_1(x)    &      & {-5.075 \leq x  <  -1.414}\\
F_2(x)    &      & {-1.414 \leq 0 < 1.414}\\ 
F_3(x)    &      & {1.414 \leq 0 \leq 5.075}\\
x + \epsilon        &      & {x > 5.075}
\end{array} \right. $$
- where $F_1(x)= -0.568686678 -  0.529288810 * x  - 0.183509590* x^2 - 0.028070202 * x ^ 3  -0.001597741 * x ^ 4$\
$F_2(x)=0.001193207 +  0.5 * x  + 0.385858026* x^2 + .0 * x ^ 3  -0.045101361 * x ^ 4 $\
$F_3(x)= -0.438406187 + 1.340789252 * x - 0.087184212 * x ^ 2 + 0.007334718 * x ^ 3$

因此 ： $seg5Gelu = b_0 * 0 + b_1*F_1(x) + b_2*F_2(x)  + b_3*F_3(x) + b_4 * x  $


## Init phase :
$Client$: 拥有$W_{F1}^C, B_{F1}^C$，和$W_{F2}^C, B_{F2}^C$

$Server$: 拥有$W_{F1}^S, B_{F1}^S$，和$W_{F2}^S, B_{F2}^S$, 以及上一层结果$[Lanorm]_C$

过程：



$Server$

1. 构建$[x^C]_C = [Lanorm]_C - g^S, 则x^S= g^S$，
3. 发送$[x^C]_C$ 给$Client$

$Client$
1. 接收$[x^C]_C$，得到$x^C$, 并生成随机数$v^C$
2. 计算$v^Cx^C , v^Cx^C \bigotimes W_{F1}^C, v^C \bigotimes W_{F1}^C, [v^C]_C,  B_{F1}^C \cdot v^C$
3. 发送$H_4 = \left\{v^Cx^C , v^Cx^C \bigotimes W_{F1}^C, v^C \bigotimes W_{F1}^C, [1/v^C]_C, B_{F1}^C \cdot v^C \right\}$给$Server$

$Server$
1. 收到$H_4$
2. 计算$[x\cdot W^{F1}]_C= (v^Cx^C \bigotimes W_{F1}^C + v^Cx^C \bigotimes W_{F1}^S + v^C W_{F1}^C \bigotimes x^S) \bigotimes [1/v^C]_C + x^S \bigotimes  W_{F1}^S  $
3. 计算$[B_{F1}]_C = [B_{F1}^C]_C +  B_{F1}^S $   
4. 计算$[FFN^1]_C = [x\cdot W^{F1}]_C + [B_{F1}]_C $


2. (按元素)计算 $[S(0)]_C = [FFN^1]_C + 5.075$, $(x > -5.075)$\\\
       $[S(1)]_C  = [FFN^1]_C + 1.414$；$(x > -1.414)$\\\
       $[S(2)]_C  = [FFN^1]_C - 1.414$；$(x > 1.414)$\\\
       $[S(3)]_C  = [FFN^1]_C - 5.075$, $(x > 5.075)$ 
3. 生成$\frac{1}{ Shr(i)^S}$,记录符号：$Server(sgn)$； 计算$[Shr(i)^C]_C= [S(i)]_C \bigotimes \frac{1}{Shr(i)^S}$

3. 发送$[Shr(i)^C]_C$ 到$Client$.

 $Client$

1. 接受并解密$[Shr(i)^C]_C$,得到$Shr(i)^C$
2. 计算起符号$Client(sgn)$，
3. 发送$Client(sgn)$至$Server$

$Server$

1. 拥有$Server(sgn)$,
2. 得到$Client(sgn)$， 正号标记为1， 负号标记为0 ，进行 xnor 计算，得到$sgn([S(i)]_C)$
3. 确定区间：\
   计算$[S(i)]_C = sgn([S(i)]_C) * 0.5 $， \
   计算 $b(0) = 0.5 - [S(0)]_C$ <--->  $b(0) =1 ~~ {(x < - 5.075)}$,\
   计算 $b(1) = [S(0)]_C - [S(1)]_C$ <--->  $b(1) =1 ~~ {(-5.075 < x < -1.414)}$\
   计算 $b(2) = [S(1)]_C - [S(2)]_C$ <--->  $b(2) =1 ~~ {(-1.414 < x < 1.414)}$,\
   计算 $b(3) = [S(2)]_C - [S(3)]_C$ <--->  $b(3) =1 ~~ {( 1.414 < x < 5.075)}$,\
   计算 $b(4) = 0.5 + [S(2)]_C$ <--->   $b(4) =1 ~~ {(x > 5.075)}$,
 
4. 确定完区间，根据$[FFN^1]_C$,计算seg5Gelu：  构建$[FFN^1]_C \bigotimes 1/r^S,以及[r^S]_S$

$Client$
（复用$FFN^1$,增加一轮通信）

1. 接受$[FFN^1 \bigotimes 1/r^S]_C$,以及$[r^S]_S$
2. 计算 $ FFN^1 \bigotimes 1/r^S  \bigotimes  [r^S]_S $, 得到 $[FFN^1]_S$
3. 构建 $[FFN^1 \bigotimes 1/r^C]_S$，及$[r^C]_C$， 发送到$Server$

$Server$(根据区间获取对应seg5GeLU结果)
1. 接受$[FFN^1\bigotimes 1/r^C]_S ,以及[r^C]_C, [(r^C)^2]_C, [(r^C)^3]_C, [(r^C)^4]_C$
2. 得到 $ FFN^1 \bigotimes 1/r^C $
3. 计算 $(FFN^1 \bigotimes 1/r^C)^2、([FFN^1] \bigotimes 1/r^C)^3、([FFN^1] \bigotimes 1/r^C)^4 $
4. 计算$(FFN^1 \bigotimes 1/r^C)^i \bigotimes  [(r^C)^i]_C $, 可得到 $x^i = [(FFN^1)^i]_C$
5. 计算$seg5Gelu = b_0 * 0 + b_1*F_1(x) + b_2*F_2(x)  + b_3*F_3(x) + b_4 * x  $, 得到结果 $[seg5gelu]_C$


6. 生成随机数$t^S$, 构建$[x^C]_C = [seg5gelu]_C -t^S, x^S = t^S$,
7. 发送至$[x^C]_C$到$Client$.



$Client$
1. 接收$[x^C]_C$，得到$x^C$
2. 重复$FFN^1$计算过程，可在$Server$端得到$[FFN^2]_C$


$Server$
得到$[FFN^2]_C$, 构建$[FFN^2]_C \bigotimes 1/v^S, [v^S]_S$


----

$Client$
1. 接受$[FFN^2]_C \bigotimes 1/v^S, [v^S]_S$
2. 处理$FFN^2 \bigotimes 1/v^S$, 并计算：$FFN^2 \bigotimes 1/v^S \bigotimes [v^S]_S  =  [FFN^2]_S $


## $Secure ~~LayerNorm$ ($Add$ & $Norm$)
$Add\&Norm$: 

$Client$
1. 计算$[FFN^2]_S + x = [x_{add}]_S$ ; 复用$secure LayerNorm$，

$Server$
1. 得到$[Bert-Onelayer]_C$ 


最终$Cleint$ $Server$分别拥有结果$y^C$ 和 $y^S$。




Conversion from SS to HE

he = neg_mod(signed_val(ss-input[], ell), (int64_t)plain_mod);

Conversion from HE to SS


get_p_sub:





