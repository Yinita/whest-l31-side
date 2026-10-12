# H17 Gaussian 二阶 capped 核心去重

固定上游 lean commit 1a4083f89184d6b1cc55d2e55a5a561d2ae50a8d，lean_k3_aug.py:651 的 pK→K 和1262的 cA 核心组装为来源。V29:1473仅保留 K4/K22 diagonal，另以拟合 lambda*Coff 代替非对角核心。

对互异 a,b,c，令 p=E f′、q=E f″、m=E f，中心平方的导数为 A=2m(1−p)、B=2p−2mq−2p²。Gaussian 二阶 cumulant K(a,a,b,c) 为：

B_a p_b p_c C_ab C_ac + A_a q_b p_c C_ab C_bc + A_a p_b q_c C_ac C_bc。

独立合同从三变量 moment 的协方差边展开构造中心平方，再减 variance(a)*Cov(b,c) 和2Cov(a,b)Cov(a,c)，保留总边阶数≤2。不是引用生产公式作为期望值。n4/7/11最大误差2.665e-15。

cA=6/(n+4)，矩阵 diagonal为零；不修改已存在K4/K22 diagonal。核心投影使用当前post-ReLU C_off（不是输入pre covariance）：R=cA*Kcap−<cA*Kcap,Cpost_off>/<Cpost_off,Cpost_off>*Cpost_off。因此新增R与V29已有scalar metric方向正交。transport W R Wᵀ，diag加入dG，offdiag加入WK431；没有把投影系数再加入拟合lambda。

限制：此正交化去除指定表示空间的重叠，不证明拟合lambda与真实Gaussian/core物理贡献一致。非Gaussian源、三阶source cap抵消、三阶以上covariance图均未恢复。v30.047-r04是完整Gaussian二阶机制实验，不是完整四阶core闭合。全部出生层重置memoryless residual，与V29再生策略一致，首次单网快筛后才决定扩展。

证据：ablation_v30_047_r03_centering_projection_contract.json；候选estimators/ablation_v30_047_r04/estimator.py。分数以r04_mini1原始报告为准。
