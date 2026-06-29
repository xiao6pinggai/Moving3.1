# 三点运动一致性稀疏卷积模块规格书

## 1. 模块名称

**Triplet Motion Consistency Sparse Convolution**

简称：**TMC-SConv**。

该模块用于稀疏时空点云特征增强。模块不改变稀疏点数量和点坐标，仅更新点特征。

---

## 2. 输入与输出

输入稀疏点特征：

$$
F\in\mathbb{R}^{N\times C}
$$

输入稀疏点坐标：

$$
P\in\mathbb{R}^{N\times 4}
$$

其中：

$$
p_i=(b_i,t_i,x_i,y_i)
$$

空间坐标记为：

$$
s_i=(x_i,y_i)
$$

输入形式：

```text
coords : [N, 4]    # (b, t, x, y)
feats  : [N, C]
```

输出形式：

```text
out_coords : [N, 4]    # 坐标保持不变
out_feats  : [N, C]    # 特征更新
```

---

## 3. 基本参数

```text
k       : 每个时间方向的邻居数量
d_t     : 时域 dilation，默认 d_t = 1
C       : 输入/输出通道数
C_h     : FFN 隐藏通道数
```

其中：

```text
上一帧搜索时间 = t_i - d_t
下一帧搜索时间 = t_i + d_t
```

默认：

$$
d_t=1
$$

---

## 4. 前后帧邻居提取

对每个点 $i$，只在同一个 batch 内搜索邻居。

上一帧邻居集合定义为：

$$
\mathcal{N}^{-}(i)
=
KNN_k\left(
    s_i,
    \{s_j\mid b_j=b_i,\ t_j=t_i-d_t\}
\right)
$$

下一帧邻居集合定义为：

$$
\mathcal{N}^{+}(i)
=
KNN_k\left(
    s_i,
    \{s_h\mid b_h=b_i,\ t_h=t_i+d_t\}
\right)
$$

得到邻居索引：

```text
idx_prev : [N, k]
idx_next : [N, k]
```

若某个点在目标帧内的有效邻居数量小于 $k$，使用 padding 补齐，并生成有效性 mask：

```text
mask_prev : [N, k]
mask_next : [N, k]
```

其中有效邻居位置为 1，无效 padding 位置为 0。

三点组合 mask 定义为：

$$
M_{i,a,b}=M^{-}_{i,a}\cdot M^{+}_{i,b}
$$

形状为：

```text
mask_pair : [N, k, k]
```

---

## 5. 特征与坐标 Gather

根据邻居索引提取特征：

```text
f_cur  : [N, C]
f_prev : [N, k, C]
f_next : [N, k, C]
```

提取空间坐标：

```text
s_cur  : [N, 2]
s_prev : [N, k, 2]
s_next : [N, k, 2]
```

其中：

$$
f_{prev}[i,a]=f_{j_a},\quad j_a\in\mathcal{N}^{-}(i)
$$

$$
f_{next}[i,b]=f_{h_b},\quad h_b\in\mathcal{N}^{+}(i)
$$

---

## 6. 构造三点组合

对每个中心点 $i$，从上一帧取一个邻居 $j_a$，从下一帧取一个邻居 $h_b$，与当前点 $i$ 组成三点单元：

$$
(j_a,\ i,\ h_b)
$$

其中：

$$
a=1,\dots,k
$$

$$
b=1,\dots,k
$$

每个中心点共有：

$$
k^2
$$

个三点组合。

---

## 7. 计算相对运动量

上一帧到当前帧的位移定义为：

$$
s^{-}_{i,a}=s_i-s_{j_a}
$$

当前帧到下一帧的位移定义为：

$$
s^{+}_{i,b}=s_{h_b}-s_i
$$

两段位移差定义为：

$$
a_{i,a,b}=s^{+}_{i,b}-s^{-}_{i,a}
$$

展开为：

$$
a_{i,a,b}=s_{h_b}-2s_i+s_{j_a}
$$

其中 $a_{i,a,b}$ 表示短时运动一致性残差。如果目标短时近似线性运动，则：

$$
a_{i,a,b}\approx 0
$$

张量形状：

```text
s_minus : [N, k, 2]
s_plus  : [N, k, 2]
```

通过 broadcast 得到：

```text
s_minus_pair : [N, k, k, 2]
s_plus_pair  : [N, k, k, 2]
a_pair       : [N, k, k, 2]
```

---

## 8. 相对位置编码

将三类相对运动量拼接：

$$
z_{i,a,b}
=
\left[
    s^{-}_{i,a},\
    s^{+}_{i,b},\
    a_{i,a,b}
\right]
$$

因此：

$$
z_{i,a,b}\in\mathbb{R}^{6}
$$

张量形状：

```text
motion_input : [N, k, k, 6]
```

使用位置 MLP 计算相对位置编码分数：

$$
r_{i,a,b}=MLP_{pos}(z_{i,a,b})
$$

其中：

$$
r_{i,a,b}\in\mathbb{R}
$$

张量形状：

```text
r_pos : [N, k, k, 1]
```

为了用于特征调制，将其变换为标量调制权重：

$$
g_{i,a,b}=2\cdot\sigma(r_{i,a,b})
$$

其中：

$$
g_{i,a,b}\in(0,2)
$$

若 $MLP_{pos}$ 最后一层 bias 初始化为 0，则初始时：

$$
g_{i,a,b}\approx 1
$$

对无效三点组合进行 mask：

$$
g_{i,a,b}=g_{i,a,b}\cdot M_{i,a,b}
$$

最终形状：

```text
g_pos : [N, k, k, 1]
```

---

## 9. 三点特征卷积计算

定义三个共享线性映射：

$$
W_{-}:\mathbb{R}^{C}\rightarrow\mathbb{R}^{C}
$$

$$
W_{0}:\mathbb{R}^{C}\rightarrow\mathbb{R}^{C}
$$

$$
W_{+}:\mathbb{R}^{C}\rightarrow\mathbb{R}^{C}
$$

分别对应上一帧点、当前点、下一帧点。

先计算：

$$
u^{-}_{i,a}=W_{-}f_{j_a}
$$

$$
u^{0}_{i}=W_{0}f_i
$$

$$
u^{+}_{i,b}=W_{+}f_{h_b}
$$

张量形状：

```text
u_prev : [N, k, C]
u_cur  : [N, C]
u_next : [N, k, C]
```

通过 broadcast 构造三点卷积特征：

$$
u_{i,a,b}
=
u^{-}_{i,a}+
u^{0}_{i}+
u^{+}_{i,b}
$$

形状为：

```text
u_triplet : [N, k, k, C]
```

---

## 10. FFN 特征增强

对每个三点组合的卷积特征使用共享 FFN：

$$
m_{i,a,b}=FFN(u_{i,a,b})
$$

基础版 FFN 定义为：

$$
FFN(x)=W_2\delta(W_1x)
$$

其中 $\delta$ 为 ReLU。若使用隐藏通道 $C_h$，则：

$$
W_1:\mathbb{R}^{C}\rightarrow\mathbb{R}^{C_h},\quad
W_2:\mathbb{R}^{C_h}\rightarrow\mathbb{R}^{C}
$$

形状保持不变：

```text
m_triplet : [N, k, k, C]
```

---

## 11. 相对位置调制

使用相对位置调制权重对三点特征进行逐组合调制：

$$
\hat{m}_{i,a,b}=g_{i,a,b}\cdot m_{i,a,b}
$$

其中 $g_{i,a,b}$ 为标量，会 broadcast 到通道维度。

张量形状：

```text
m_mod : [N, k, k, C]
```

---

## 12. $k^2$ 三点特征聚合

对每个中心点 $i$，将 $k^2$ 个调制后三点特征求和并归一化：

$$
y_i
=
\frac{1}{k^2}
\sum_{a=1}^{k}
\sum_{b=1}^{k}
\hat{m}_{i,a,b}
$$

张量形状：

```text
y : [N, C]
```

无效 padding 三点组合已经在调制权重 $g_{i,a,b}$ 中被置零。基础版固定除以 $k^2$，保持不同点之间的输出尺度一致。

如果希望只对有效三点组合求均值，可以替换为：

$$
y_i
=
\frac{1}{\max(1,\sum_{a,b}M_{i,a,b})}
\sum_{a=1}^{k}
\sum_{b=1}^{k}
\hat{m}_{i,a,b}
$$

基础版默认使用固定 $k^2$ 归一化。

---

## 13. BN + ReLU

对聚合结果进行归一化和激活：

$$
\tilde{y}_i=ReLU(BN(y_i))
$$

张量形状：

```text
y_norm : [N, C]
```

其中 BN 使用 `BatchNorm1d(C)`，输入形状为：

```text
[N, C]
```

---

## 14. 残差输出

最终输出特征为：

$$
f_i^{out}=f_i+\tilde{y}_i
$$

整体输出：

```text
out_feats  : [N, C]
out_coords : [N, 4]
```

坐标保持不变。

---

## 15. 完整计算流程汇总

```text
输入:
coords: [N, 4]
feats : [N, C]

1. 根据 (b, t-d_t) 和 (b, t+d_t) 并行搜索前后帧 KNN
   idx_prev: [N, k]
   idx_next: [N, k]

2. 处理邻居不足 k 的情况
   mask_prev: [N, k]
   mask_next: [N, k]
   mask_pair: [N, k, k]

3. Gather 特征和坐标
   f_prev: [N, k, C]
   f_cur : [N, C]
   f_next: [N, k, C]
   s_prev: [N, k, 2]
   s_cur : [N, 2]
   s_next: [N, k, 2]

4. 计算相对运动量
   s_minus = s_cur - s_prev
   s_plus  = s_next - s_cur
   a       = s_plus - s_minus

5. 拼接相对运动输入
   motion_input = [s_minus, s_plus, a]
   motion_input: [N, k, k, 6]

6. MLP_pos 生成相对位置调制
   r_pos: [N, k, k, 1]
   g_pos: [N, k, k, 1]

7. 三点特征卷积
   u_triplet = Linear_prev(f_prev)
             + Linear_cur(f_cur)
             + Linear_next(f_next)
   u_triplet: [N, k, k, C]

8. FFN 提取三点特征
   m_triplet: [N, k, k, C]

9. 相对位置调制
   m_mod = g_pos * m_triplet
   m_mod: [N, k, k, C]

10. 聚合
   y = sum(m_mod over k,k) / k^2
   y: [N, C]

11. BN + ReLU
   y_norm = ReLU(BN(y))

12. 残差连接
   out_feats = feats + y_norm
```

---

## 16. 复杂度

由于该模块不改变体素点数和坐标，同级结构中可以缓存：

```text
idx_prev, idx_next, mask_prev, mask_next
```

若邻居索引已缓存，主要计算包括：

1. 三路线性映射：

$$
O(N(2k+1)C^2)
$$

2. 相对位置编码：

$$
O(Nk^2C_{pos})
$$

其中 $C_{pos}$ 表示位置 MLP 的隐藏计算规模。

3. 三点 FFN：

$$
O(Nk^2CC_h)
$$

4. 调制与聚合：

$$
O(Nk^2C)
$$

主要中间张量显存为：

```text
[N, k, k, C]
```

对应：

```text
u_triplet / m_triplet / m_mod
```

若 $C_h\approx C$，主计算量通常由三点 FFN 决定。

---

## 17. 模块特点

```text
1. 不改变稀疏点数量；
2. 不改变稀疏点坐标；
3. 支持时域 dilation；
4. 显式建模 t-d_t, t, t+d_t 三点短时运动关系；
5. 使用 s_minus, s_plus, a 建模短时线性运动一致性；
6. 不需要对邻居排序；
7. 通过相对位置调制实现运动选择；
8. 聚合方式为调制后的均值聚合，结构简单且并行友好。
```
