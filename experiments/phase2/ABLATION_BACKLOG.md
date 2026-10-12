# WhestBench Phase 2 待验证消融清单

> 执行已获用户授权；本文件为冻结规格，状态与before/after以 [ABLATION_RESULTS.md](ABLATION_RESULTS.md) 和 ablation_queue.json 为准。清单不是129项完成声明。
清单版本 1.0，2026-10-12。共 129 项，按独立机制和验证对象拆分；不是 129 个已实现算法，也不承诺十倍收益。所有 v30.xxx 为本项目预留消融版本号，和公开上游 V29 版本无关。before 与 after 描述实现变化，不填写预测分数。
算法与计算消融 71 项，诊断 43 项，合同与资源验证 15 项。已有代码 1 项，规格待实现 69 项，需先推导 59 项。诊断和资源结果不得记成精度提升。
## 冻结对照与已完成结果
原 V29 SHA256 为 `86d9ca9b28e6fe2b6c74750a0b6ae4bba4c14f742ddc3f5f56bc7fb4ec9d8e27`。Mini8 adjusted 为 5.778979755766664e-9，raw 为 2.2702818514019896e-8，零失败。第一阶段目标仍为 adjusted < 5.779e-10。当前单网控制 network0 raw 为 2.290466483145792e-8，adjusted 为 6.116832334984139e-9，FLOPs 为 587262754287。单网分数与 Mini8 aggregate 不混用。
已测负结果单独保留：pairq3、spilluse、k4square、tree4marg、tree4last、trianglelast、doubleedge-final 均未改善 adjusted；coreproxy、coreproxy-first、年龄及禁用通道的历史结果见 RESEARCH_LOG。V25 对照 raw 接近 V29、FLOPs 高约37.3%。这些不进入默认待跑队列；更广对象必须先通过条目的重新开放门槛。symrecursive 只有合成核验，尚未跑比赛网络，copy 调度可能超 residual 限制。
## 执行边界
一项只改变一个主要机制。父版本默认冻结 V29；依赖版必须另存代码哈希并说明成对基线。依赖只是实现与证据前置，不表示自动把多个机制合并。独立采样种子、原 reference、网络集合、工具版本、2**41 预算、单线程、120秒 predict、0.4秒 residual、5秒 setup 和 lambda=0 固定。每次使用新输出路径，记录失败和负结果。
低推理执行器只能选择已有代码或规格明确、依赖满足的条目；缺少公式、系数、重复指标扣除、合法算子或成本证明时改为需推导，不能猜测后继续。长研究项保留为研究储备，不自动逐项运行。小规模代数通过是实现门槛，不是提分证据。
stage 达标须冻结Mini8低于目标、真正未见网络相对上一冻结候选至少十倍收益且零失败，并通过RPC/合同/资源与参考可信度验收。达标才把目标除以10；运行次数不推动stage。仅首次AIcrowd提交已获授权，后续提交必须先展示证据再请求授权；无需新增定时任务、购买算力或向官方HF输入仓库写输出。
## 2026-10-12 用户优先级升级

论文方法级对照提升为P0/P1，当前运行安全完成后优先于其余局部消融。执行顺序、论文来源、已有覆盖和验收见[论文方法级对照](PAPER_METHODS_PRIORITY_2026-10-12.md)。保留原129项状态与证据；不得重建队列覆盖结果。

## 首批执行顺序
v30.024 C24 → v30.025 C25 → v30.007 C07 → v30.015 C15 → v30.014 C14 → v30.008 C08 → v30.033 H03 → v30.031 H01 → v30.032 H02 → v30.048 H18 → v30.010 C10 → v30.011 C11 → v30.063 O01 → v30.064 O02 → v30.073 O11 → v30.096 L01 → v30.097 L02 → v30.113 L18 → v30.098 L03。这是建议顺序，依赖和条件优先于排序；若控制/正确性不通过先定位。C01待copy风险评估，C29先准备调度推导，二者不设循环依赖。
## 条目索引
| 版本 | 编号 | 名称 | 类型 | 准备度 | 优先级 |
|---|---|---|---|---|---|
| v30.001 | C01 | 递归限制C_pre上三角输出 | 算法或计算消融 | 已有代码待正式评测 | P0 |
| v30.002 | C02 | Strassen–Winograd共享加法链 | 算法或计算消融 | 先推导再实现 | P1 |
| v30.003 | C03 | 将Z与Zf薄腿合并运输 | 合同或资源验证 | 规格待实现及验证 | P1 |
| v30.004 | C04 | 将两条D21反馈收缩合并薄列 | 合同或资源验证 | 规格待实现及验证 | P2 |
| v30.005 | C05 | PPL与Z的末次薄腿收缩写成扁平GEMM | 合同或资源验证 | 规格待实现及验证 | P1 |
| v30.006 | C06 | 最后一层M残差的D3只做对角收缩 | 算法或计算消融 | 规格待实现及验证 | P1 |
| v30.007 | C07 | 省掉最终层四阶矩阵切片 | 算法或计算消融 | 规格待实现及验证 | P0 |
| v30.008 | C08 | 最终层仅运行PK1均值输出 | 算法或计算消融 | 规格待实现及验证 | P0 |
| v30.009 | C09 | 稀疏wick多项式代替dense表乘 | 算法或计算消融 | 先推导再实现 | P1 |
| v30.010 | C10 | 稀疏Wick selector改indexed选取 | 算法或计算消融 | 规格待实现及验证 | P1 |
| v30.011 | C11 | 非线性term按输出组分开收缩 | 算法或计算消融 | 规格待实现及验证 | P1 |
| v30.012 | C12 | 去重AB矩阵乘积缓存 | 算法或计算消融 | 规格待实现及验证 | P1 |
| v30.013 | C13 | PK11和PK22对称输出block化 | 算法或计算消融 | 先推导再实现 | P2 |
| v30.014 | C14 | 避免二次covariance对称平均 | 算法或计算消融 | 规格待实现及验证 | P0 |
| v30.015 | C15 | D21绝对值行和替代ones GEMV | 算法或计算消融 | 规格待实现及验证 | P0 |
| v30.016 | C16 | 小核心Sj只计算上三角 | 算法或计算消融 | 先推导再实现 | P2 |
| v30.017 | C17 | Sg双边旋转的末次乘法对称化 | 算法或计算消融 | 先推导再实现 | P2 |
| v30.018 | C18 | S2子空间投影的末次乘法三角化 | 算法或计算消融 | 先推导再实现 | P2 |
| v30.019 | C19 | joiner两路Gram operator共享布局 | 合同或资源验证 | 规格待实现及验证 | P2 |
| v30.020 | C20 | final D3 feedback的线性两项薄收缩 | 算法或计算消融 | 先推导再实现 | P3 |
| v30.021 | C21 | nonlinearity中广播d3/g4向量不materialize | 算法或计算消融 | 规格待实现及验证 | P1 |
| v30.022 | C22 | PK1 duplicate一维因子缓存去重 | 算法或计算消融 | 规格待实现及验证 | P3 |
| v30.023 | C23 | 数值对称误差审计 | 诊断 | 规格待实现及验证 | P0 |
| v30.024 | C24 | 逐组件实际计费归因 | 诊断 | 规格待实现及验证 | P0 |
| v30.025 | C25 | 浮点加法顺序与误差归因 | 诊断 | 规格待实现及验证 | P0 |
| v30.026 | C26 | 持久pool最大shape资源审计 | 诊断 | 规格待实现及验证 | P1 |
| v30.027 | C27 | shape变更时释放无用旧pool | 合同或资源验证 | 规格待实现及验证 | P2 |
| v30.028 | C28 | 分区清理不再需要的scratch常驻 | 合同或资源验证 | 先推导再实现 | P2 |
| v30.029 | C29 | 三角C_pre阶段批量copy调度 | 合同或资源验证 | 先推导再实现 | P1 |
| v30.030 | C30 | 不同backend上合法matmul计费与resource对照 | 合同或资源验证 | 规格待实现及验证 | P0 |
| v30.031 | H01 | 边际D3平方产品补全 | 算法或计算消融 | 规格待实现及验证 | P0 |
| v30.032 | H02 | 边际D3乘g4产品补全 | 算法或计算消融 | 规格待实现及验证 | P0 |
| v30.033 | H03 | 单轴截断集合一致性诊断 | 诊断 | 规格待实现及验证 | P1 |
| v30.034 | H04 | 精确Gaussian pair核替换 | 算法或计算消融 | 先推导再实现 | P1 |
| v30.035 | H05 | Gaussian pair尾项有理重求和 | 算法或计算消融 | 先推导再实现 | P2 |
| v30.036 | H06 | 未剪枝SIMPLE生产表恢复 | 算法或计算消融 | 先推导再实现 | P2 |
| v30.037 | H07 | WK22到B1出生来源恢复 | 算法或计算消融 | 先推导再实现 | P1 |
| v30.038 | H08 | B1的WK12对称反馈恢复 | 算法或计算消融 | 先推导再实现 | P1 |
| v30.039 | H09 | B2 leading D21乘C来源恢复 | 算法或计算消融 | 先推导再实现 | P1 |
| v30.040 | H10 | B2 WK22乘C来源恢复 | 算法或计算消融 | 先推导再实现 | P1 |
| v30.041 | H11 | B2 D21平方来源恢复 | 算法或计算消融 | 先推导再实现 | P2 |
| v30.042 | H12 | B2 WK22乘D21来源恢复 | 算法或计算消融 | 先推导再实现 | P2 |
| v30.043 | H13 | B2保留一次线性运输再shadow | 算法或计算消融 | 先推导再实现 | P2 |
| v30.044 | H14 | B2两步输出导向收缩 | 算法或计算消融 | 先推导再实现 | P3 |
| v30.045 | H15 | 历史K3来源生成K211 capped低rank核心 | 算法或计算消融 | 先推导再实现 | P3 |
| v30.046 | H16 | actual K31低rank核心残差 | 算法或计算消融 | 先推导再实现 | P2 |
| v30.047 | H17 | Gaussian K211完整二阶矩阵核心 | 算法或计算消融 | 先推导再实现 | P2 |
| v30.048 | H18 | K4边际真实metric系数 | 算法或计算消融 | 规格待实现及验证 | P0 |
| v30.049 | H19 | WK431真实metric混合项 | 算法或计算消融 | 先推导再实现 | P1 |
| v30.050 | H20 | WK22真实metric交叉项 | 算法或计算消融 | 先推导再实现 | P1 |
| v30.051 | H21 | 恢复非对角metric的K4到K3 feed | 算法或计算消融 | 先推导再实现 | P3 |
| v30.052 | H22 | D21反馈basis输出敏感sketch | 算法或计算消融 | 先推导再实现 | P2 |
| v30.053 | H23 | old-source basis按K21收缩能源选择 | 算法或计算消融 | 先推导再实现 | P3 |
| v30.054 | H24 | old-source basis按下一层D3敏感方向选择 | 算法或计算消融 | 先推导再实现 | P3 |
| v30.055 | H25 | Rres新增解析可分leading项 | 算法或计算消融 | 先推导再实现 | P1 |
| v30.056 | H26 | Rres物理机制分组basis | 算法或计算消融 | 先推导再实现 | P2 |
| v30.057 | H27 | K3 newborn重复内容一致性oracle | 诊断 | 规格待实现及验证 | P1 |
| v30.058 | H28 | 再生core对角和offdiag联合对象闭合 | 算法或计算消融 | 先推导再实现 | P3 |
| v30.059 | H29 | K111乘covariance边新来源 | 算法或计算消融 | 先推导再实现 | P1 |
| v30.060 | H30 | K111乘covariance边一层收缩 | 算法或计算消融 | 先推导再实现 | P2 |
| v30.061 | H31 | exact covariance triangle无偏收缩 | 算法或计算消融 | 先推导再实现 | P3 |
| v30.062 | H32 | 高阶newborn直接CP因素生成 | 算法或计算消融 | 先推导再实现 | P3 |
| v30.063 | O01 | 官方K31截断生产逐项一致性 | 诊断 | 规格待实现及验证 | P0 |
| v30.064 | O02 | 官方capped K211逐项生产一致性 | 诊断 | 规格待实现及验证 | P0 |
| v30.065 | O03 | K211完整来源抵消一致性 | 诊断 | 规格待实现及验证 | P0 |
| v30.066 | O04 | 旧lean与当前官方generator语义归因 | 诊断 | 规格待实现及验证 | P0 |
| v30.067 | O05 | 完整K31/K211生产核的离线表示上限 | 诊断 | 先推导再实现 | P1 |
| v30.068 | O06 | 完整生产核单层替换诊断 | 算法或计算消融 | 先推导再实现 | P2 |
| v30.069 | O07 | 完整核对角修正非零性与候选 | 算法或计算消融 | 先推导再实现 | P2 |
| v30.070 | O08 | 完整production核仅USE替换 | 算法或计算消融 | 先推导再实现 | P2 |
| v30.071 | O09 | 完整production核仅FEED替换 | 算法或计算消融 | 先推导再实现 | P2 |
| v30.072 | O10 | 完整核剩余误差的state模态诊断 | 诊断 | 规格待实现及验证 | P2 |
| v30.073 | O11 | K111乘covariance edge完整公式核验 | 诊断 | 规格待实现及验证 | P0 |
| v30.074 | O12 | K111乘edge的exact final D3成本证明 | 诊断 | 先推导再实现 | P1 |
| v30.075 | O13 | K111乘edge covariance主模态代理 | 算法或计算消融 | 先推导再实现 | P2 |
| v30.076 | O14 | K111乘edge代理误差门槛 | 诊断 | 规格待实现及验证 | P2 |
| v30.077 | O15 | doubled-edge早期来源贡献门槛 | 诊断 | 先推导再实现 | P3 |
| v30.078 | O16 | covariance triangle exact多层small诊断 | 诊断 | 先推导再实现 | P3 |
| v30.079 | O17 | 当前官方WK22 path与旧lean等价审计 | 诊断 | 规格待实现及验证 | P3 |
| v30.080 | O18 | 现有birth薄反馈乘积预算审计 | 诊断 | 规格待实现及验证 | P0 |
| v30.081 | O19 | K5边缘生产闭合 | 算法或计算消融 | 先推导再实现 | P1 |
| v30.082 | O20 | K5独立边缘readout | 算法或计算消融 | 先推导再实现 | P2 |
| v30.083 | O21 | K6纯径向投影生产公式 | 诊断 | 先推导再实现 | P1 |
| v30.084 | O22 | K6径向状态合法线性运输 | 算法或计算消融 | 先推导再实现 | P2 |
| v30.085 | O23 | K6独立边缘readout | 算法或计算消融 | 先推导再实现 | P2 |
| v30.086 | O24 | 高阶边缘同阶组合inventory | 诊断 | 规格待实现及验证 | P0 |
| v30.087 | O25 | degree5/6 ordinary与power累积量独立转换 | 诊断 | 规格待实现及验证 | P1 |
| v30.088 | O26 | 图表收益的上游调参分割审计 | 诊断 | 规格待实现及验证 | P1 |
| v30.089 | O27 | 封存新分片的diagslice状态分布偏移 | 诊断 | 规格待实现及验证 | P2 |
| v30.090 | O28 | 新seed网络的diagram规模law核验 | 诊断 | 规格待实现及验证 | P2 |
| v30.091 | O29 | 高阶校正信号与reference不确定性的分离 | 诊断 | 规格待实现及验证 | P1 |
| v30.092 | O30 | 新增图表array操作计量路径合同 | 合同或资源验证 | 规格待实现及验证 | P0 |
| v30.093 | O31 | 完整K31生产对称化与轴方向合同 | 合同或资源验证 | 规格待实现及验证 | P1 |
| v30.094 | O32 | capped211 diagonal mask因式表示合同 | 合同或资源验证 | 规格待实现及验证 | P0 |
| v30.095 | O33 | complete harmonic core actualW transport对照 | 诊断 | 规格待实现及验证 | P1 |
| v30.096 | L01 | 冻结原轨迹特征缓存 | 诊断 | 规格待实现及验证 | P0 |
| v30.097 | L02 | 冻结网络级训练选择流程 | 诊断 | 规格待实现及验证 | P0 |
| v30.098 | L03 | 最后层13特征残差线性修正 | 算法或计算消融 | 规格待实现及验证 | P1 |
| v30.099 | L04 | 删除常数特征的修正对照 | 算法或计算消融 | 规格待实现及验证 | P2 |
| v30.100 | L05 | 阈值形状特征增量 | 算法或计算消融 | 规格待实现及验证 | P2 |
| v30.101 | L06 | 标准化偏度特征增量 | 算法或计算消融 | 规格待实现及验证 | P2 |
| v30.102 | L07 | 标准化峰度特征增量 | 算法或计算消融 | 规格待实现及验证 | P2 |
| v30.103 | L08 | 最后线性均值传播轨迹特征 | 算法或计算消融 | 规格待实现及验证 | P2 |
| v30.104 | L09 | 前层方差轨迹特征 | 算法或计算消融 | 规格待实现及验证 | P2 |
| v30.105 | L10 | 跨神经元均值残差特征 | 算法或计算消融 | 规格待实现及验证 | P2 |
| v30.106 | L11 | 输出幅度归一化回归 | 算法或计算消融 | 规格待实现及验证 | P2 |
| v30.107 | L12 | 网络等权训练 | 诊断 | 规格待实现及验证 | P2 |
| v30.108 | L13 | 逐层共享线性修正 | 算法或计算消融 | 规格待实现及验证 | P3 |
| v30.109 | L14 | 分段线性阈值修正 | 算法或计算消融 | 先推导再实现 | P3 |
| v30.110 | L15 | 小网络非线性修正 | 算法或计算消融 | 先推导再实现 | P3 |
| v30.111 | L16 | 单前层修正递推注入 | 算法或计算消融 | 先推导再实现 | P3 |
| v30.112 | L17 | 修正器特征删列归因 | 诊断 | 规格待实现及验证 | P2 |
| v30.113 | L18 | 残差条件可预测性上界 | 诊断 | 规格待实现及验证 | P1 |
| v30.114 | L19 | 独立参考两次采样一致性 | 诊断 | 先推导再实现 | P1 |
| v30.115 | L20 | 新种子网络最终独立确认 | 合同或资源验证 | 规格待实现及验证 | P1 |
| v30.116 | L21 | 公共确认一次解封 | 合同或资源验证 | 规格待实现及验证 | P1 |
| v30.117 | L22 | 输出误差与阈值位置分解 | 诊断 | 规格待实现及验证 | P1 |
| v30.118 | L23 | 输出误差与偏度分解 | 诊断 | 规格待实现及验证 | P1 |
| v30.119 | L24 | 误差沿层积累的增量分解 | 诊断 | 规格待实现及验证 | P1 |
| v30.120 | L25 | 输出敏感性来源排序诊断 | 诊断 | 先推导再实现 | P1 |
| v30.121 | L26 | 输出敏感性来源预算分配 | 算法或计算消融 | 先推导再实现 | P2 |
| v30.122 | L27 | 来源删除负控制 | 诊断 | 先推导再实现 | P2 |
| v30.123 | L28 | 末端均值控制变量相关性 | 诊断 | 先推导再实现 | P2 |
| v30.124 | L29 | 条件高斯最后层积分对照 | 诊断 | 先推导再实现 | P2 |
| v30.125 | L30 | 末层尾概率数值稳定性 | 合同或资源验证 | 规格待实现及验证 | P2 |
| v30.126 | L32 | 跨网络异质收益检查 | 诊断 | 规格待实现及验证 | P1 |
| v30.127 | L33 | 参考dtype贡献诊断 | 诊断 | 规格待实现及验证 | P2 |
| v30.128 | L34 | 标准化计算的候选合同 | 合同或资源验证 | 规格待实现及验证 | P1 |
| v30.129 | L35 | 多次选择偏差账本 | 诊断 | 规格待实现及验证 | P0 |

## 计算重组与资源

### v30.001 C01 递归限制C_pre上三角输出

编号：WB-P2-A001。类型：算法或计算消融。准备度：已有代码待正式评测。优先级：P0。

**before：** V29仅在第一层输出分块计算11、12、22三个block；两个对角block内部仍完整计算

**after：** 各尺度只算上非对角block，对角block递归到32×32，最后对角叶子批量dense；下非对角copy transpose

**详情：** 新候选estimators/k3_v29_symrecursive/estimator.py已写。只变C_pre计算组织，不改Strassen等级。C显式对称，X=W C已经float32舍入，所以结果仅实数数学相同，浮点需验收。合成1024/lev4组件FLOPs下降19.82%，relative Frobenius差1.46e-6；未跑比赛网络。 真实网络快筛前先量化批量copy调度残差风险；C29虽是后续实现，不构造依赖环。

**依赖：** 无条目依赖；仍需通用正确性、数值及资源门槛。

**成本：** 组件1,152,064,512→923,771,904 FLOPs；14次C_pre按组件相同估算全网约0.54%，并非全网实测。copy calls增多可能撞0.4秒residual。 当前朴素实现每次1024 C_pre约220个copyto，比原版10个多210；14次约多2940调用，残差失败风险高，应先C29调度审计。

**验收及停止条件：** 当前V29成对本地8网；FLOPs逐网严格降低，raw变化落入数值控制，adjusted降低且0失败，然后RPC；总residual保持0.4秒。

**依据：** estimators/k3_v29_symrecursive/provenance.json; results/phase2/symrecursive_validation_20261012.json

### v30.002 C02 Strassen–Winograd共享加法链

编号：WB-P2-A002。类型：算法或计算消融。准备度：先推导再实现。优先级：P1。

**before：** 现_Strassen采用标准7乘法18加减的组合/组装

**after：** 保持每层7个乘法，使用Winograd15加减共享中间链

**详情：** 唯一变化是矩阵乘法加减调度；不是调递归等级。为mm与hub分别推导，先只替换plain mm，保留原hub作对照。浮点累积误差可能更大。

**依赖：** 无条目依赖；仍需通用正确性、数值及资源门槛。

**成本：** 不减7个GEMM，仅每节点少3个elementwise add；深层大量leaf可有可测节约，需actual BudgetContext计量。

**验收及停止条件：** 随机rectangular、broadcast、batched float32/64离线对照；再network0与数值控制、0失败。

**依据：** estimators/k3_v29/estimator.py; public source commit 1a4083f89184d6b1cc55d2e55a5a561d2ae50a8d

### v30.003 C03 将Z与Zf薄腿合并运输

编号：WB-P2-A003。类型：合同或资源验证。准备度：规格待实现及验证。优先级：P1。

**before：** 同一WD分别乘Z_st与Zf_st，两个GEMM调用

**after：** 将两个薄列组装成一个持久stack，用一次WD矩阵乘法，再view split

**详情：** 列空间完全相同，按旧列顺序保持；结果表达上独立，GEMM浮点实现可能改变。该项主要减少residual调度，不应宣称FLOPs降低。

**依赖：** 无条目依赖；仍需通用正确性、数值及资源门槛。

**成本：** 理论GEMM FLOPs不变，增加/避免组装copy取决persistent slab；峰值可避免双scratch。

**验收及停止条件：** FLOPs不能增加；small parity及predict/residual时间减少；如只资源改善记录为资源结果。

**依据：** estimators/k3_v29/estimator.py; public source commit 1a4083f89184d6b1cc55d2e55a5a561d2ae50a8d

### v30.004 C04 将两条D21反馈收缩合并薄列

编号：WB-P2-A004。类型：合同或资源验证。准备度：规格待实现及验证。优先级：P2。

**before：** GYR R2/F2与GXR R1/F1分别GEMM后sum

**after：** 合并[GYR,GXR]和[F2,F1]薄列一次乘法并一次k-sum

**详情：** 保持GY/GX构造不变。必须使用持久buffer而非每层concatenate；两个相加顺序变化需数值复验。

**依赖：** 无条目依赖；仍需通用正确性、数值及资源门槛。

**成本：** 理论GEMM总乘加相同，可减少调用/一次k-sum；新copy可能抵消。

**验收及停止条件：** small arbitrary factors正确；实际FLOPs/残差时间不更差；无法改进则保留负结果。

**依据：** estimators/k3_v29/estimator.py; public source commit 1a4083f89184d6b1cc55d2e55a5a561d2ae50a8d

### v30.005 C05 PPL与Z的末次薄腿收缩写成扁平GEMM

编号：WB-P2-A005。类型：合同或资源验证。准备度：规格待实现及验证。优先级：P1。

**before：** einsum kiq,kcq->ic产生sum_k,q PPL_kiq Z_kcq

**after：** k和q合并为列，计算PPL_flat @ Z_flat.T

**详情：** 仅重排实际数据布局，不能给数据贴假symmetry标签。选持久列major/fused存储减少reshape-copy开销。

**依赖：** 无条目依赖；仍需通用正确性、数值及资源门槛。

**成本：** 数学FLOPs基本不变；可能改善einsum实现残差，额外布局copy需计。

**验收及停止条件：** 对同一输入float32误差控制且FLOPs不增加，resource路径0失败。

**依据：** estimators/k3_v29/estimator.py; public source commit 1a4083f89184d6b1cc55d2e55a5a561d2ae50a8d

### v30.006 C06 最后一层M残差的D3只做对角收缩

编号：WB-P2-A006。类型：算法或计算消融。准备度：规格待实现及验证。优先级：P1。

**before：** trim仍构造完整Z L.T，再乘P并对P收缩

**after：** 用PP @ L后逐元素乘Z、沿k,q求和，不构造完整Z L.T

**详情：** 恒等式sum_j (Z L.T)_ij P_ij² = sum_q (PP L)_iq Z_iq。仅在need_d21=False替换M残差部分；Sb/Eb部分保留旧式。

**依赖：** 无条目依赖；仍需通用正确性、数值及资源门槛。

**成本：** 两种主要GEMM都是2kn²r，理论FLOPs相近；去掉(k,n,n)中间与部分elementwise，改善峰值/残差。

**验收及停止条件：** 小形状factor oracle；float32对照，再最后层真实误差/计费/资源。

**依据：** estimators/k3_v29/estimator.py; public source commit 1a4083f89184d6b1cc55d2e55a5a561d2ae50a8d

### v30.007 C07 省掉最终层四阶矩阵切片

编号：WB-P2-A007。类型：算法或计算消融。准备度：规格待实现及验证。优先级：P0。

**before：** trim时regen仍形成g22c及完整wk4m，后续没有读取

**after：** trim时仅计算dG和g4row，不创建wk4m

**详情：** 机械可执行：在regen的wk4m构造处以not trim保护；确认debug允许wk4m=None且最后mean/feature不读该对象。只删除dead computation。

**依赖：** 无条目依赖；仍需通用正确性、数值及资源门槛。

**成本：** 每网约一个n²broadcast add与zero diagonal及相关分配，收益小但精确。

**验收及停止条件：** 带debug和无debug small contracts；逐网FLOPs下降、预测完全/近似一致。

**依据：** estimators/k3_v29/estimator.py; public source commit 1a4083f89184d6b1cc55d2e55a5a561d2ae50a8d

### v30.008 C08 最终层仅运行PK1均值输出

编号：WB-P2-A008。类型：算法或计算消融。准备度：规格待实现及验证。优先级：P0。

**before：** trim仍计算(1,),(2,),(3,),(4,)四个一维PK1输出

**after：** trim专用progs仅包含(1,)且只写pk1v

**详情：** 最后break之前仅mean correction需要pk1v，不需要pk2v/pk3v/pk4v；应把变量读取也按trim分支保护，不仅删除表行。

**依赖：** 无条目依赖；仍需通用正确性、数值及资源门槛。

**成本：** O(n)一维项、selector GEMM和stack减少；全网比例很小。

**验收及停止条件：** 最终输出与旧版float32近似一致，FLOPs确实降低；所有smalldepth paths通过。

**依据：** estimators/k3_v29/estimator.py; public source commit 1a4083f89184d6b1cc55d2e55a5a561d2ae50a8d

### v30.009 C09 稀疏wick多项式代替dense表乘

编号：WB-P2-A009。类型：算法或计算消融。准备度：先推导再实现。优先级：P1。

**before：** APOW @ C1、APOW @ C2、SB @ SELC计入零系数乘法

**after：** 逐WICK_PAIR用明确稀疏Horner/单项式选择构造

**详情：** 从既有常数表自动生成可审查表达；不可借host numpy计算。只重组wick计算，保持系数和支持对所有shape。

**依赖：** 无条目依赖；仍需通用正确性、数值及资源门槛。

**成本：** O(n·21)表达替代n×6×21和n×11×21dense multiply；新stack数目影响residual。

**验收及停止条件：** 所有21个wick在宽sigma/alpha合成范围对照；production0失败，实际计费下降。

**依据：** estimators/k3_v29/estimator.py; public source commit 1a4083f89184d6b1cc55d2e55a5a561d2ae50a8d

### v30.010 C10 稀疏Wick selector改indexed选取

编号：WB-P2-A010。类型：算法或计算消融。准备度：规格待实现及验证。优先级：P1。

**before：** SL2/SR2为onehot/单系数表却做dense GEMM @ WT

**after：** 依已有wl,wr indices选择WT行并乘coef

**详情：** 不改变term表或coef，SL/IND表保持作诊断；indices来自Python常量，实际数组运算fnp。

**依赖：** 无条目依赖；仍需通用正确性、数值及资源门槛。

**成本：** selector矩阵乘法可降为Tn标量multiply及index/copy；整体贡献小。

**验收及停止条件：** 每term WL/WR parity；实际计费不是理论估计；0失败并raw漂移控制。

**依据：** estimators/k3_v29/estimator.py; public source commit 1a4083f89184d6b1cc55d2e55a5a561d2ae50a8d

### v30.011 C11 非线性term按输出组分开收缩

编号：WB-P2-A011。类型：算法或计算消融。准备度：规格待实现及验证。优先级：P1。

**before：** IND2 onehot但统一einsum输出三个g通道

**after：** 按pk11/pk21/pk22将term批次分组，只求各自sum

**详情：** 同一term不能被机械复制错group；先对flopscope实际einsum路径诊断是否已自动利用稀疏，若已如此则无须候选。

**依赖：** 无条目依赖；仍需通用正确性、数值及资源门槛。

**成本：** 可能避免dense zero onehot的G×T运算；也可能增加三次调用和buffers。

**验收及停止条件：** 随机term oracle与现PK2对照；actual FLOPs必须下降且residual不更差。

**依据：** estimators/k3_v29/estimator.py; public source commit 1a4083f89184d6b1cc55d2e55a5a561d2ae50a8d

### v30.012 C12 去重AB矩阵乘积缓存

编号：WB-P2-A012。类型：算法或计算消融。准备度：规格待实现及验证。优先级：P1。

**before：** AB2按term重复fnp.multiply相同pair进abbuf

**after：** 每unique unordered AB pair只算一次，term通过index view复用

**详情：** 旧ab_cache仅用于B1，AB2未去重。需核对分组收缩布局是否导致gather再次copy；不可把所有中间全常驻增峰值。

**依赖：** v30.011 C11。

**成本：** 约uniquePairs×n²乘法取代terms×n²；内存/元素乘法可减，selector/收缩仍计费。

**验收及停止条件：** syntheticAB values parity、逐组PK2校验；FLOPs/peakRSS实测。

**依据：** estimators/k3_v29/estimator.py; public source commit 1a4083f89184d6b1cc55d2e55a5a561d2ae50a8d

### v30.013 C13 PK11和PK22对称输出block化

编号：WB-P2-A013。类型：算法或计算消融。准备度：先推导再实现。优先级：P2。

**before：** 每组计算完整PK矩阵再平均transpose

**after：** 每个上三角输出block同时合并正向/反向term，复制下三角

**详情：** 不是假定单个term对称。完整组合含d21方向项，必须独立推导average(PK+PK.T)/2。PK21保持全矩阵。

**依赖：** 无条目依赖；仍需通用正确性、数值及资源门槛。

**成本：** 理想近半pk11/pk22元素收缩，增加小块调用；不能预承诺残差通过。

**验收及停止条件：** 任意非对称D21随机inputs与原完整式逐元素对照，再真实benchmark。

**依据：** estimators/k3_v29/estimator.py; public source commit 1a4083f89184d6b1cc55d2e55a5a561d2ae50a8d

### v30.014 C14 避免二次covariance对称平均

编号：WB-P2-A014。类型：算法或计算消融。准备度：规格待实现及验证。优先级：P0。

**before：** pk11已平均其transpose；assembly又C=(K11+K11.T)*0.5

**after：** 若pk11逐值确实对称且finite，copy K11后fill diagonal

**详情：** 严格核对zero/sign/subnormal/overflow行为：不能仅依赖理论对称。该项只删冗余elementwise，不添加symmetry计量标签。

**依赖：** 无条目依赖；仍需通用正确性、数值及资源门槛。

**成本：** 每非末层少n²add及n²multiply，copy原来也有必要；峰值可降低。

**验收及停止条件：** 对pk11生成各case bit parity、所有真实rows finite；FLOPs下降。

**依据：** estimators/k3_v29/estimator.py; public source commit 1a4083f89184d6b1cc55d2e55a5a561d2ae50a8d

### v30.015 C15 D21绝对值行和替代ones GEMV

编号：WB-P2-A015。类型：算法或计算消融。准备度：规格待实现及验证。优先级：P0。

**before：** D21n=abs(D21) @ ones_n

**after：** D21n=sum(abs(D21),axis=1)

**详情：** 数学相同，row reduction浮点求和次序不同，拟合beta仍不动；该项已可机械实现。

**依赖：** 无条目依赖；仍需通用正确性、数值及资源门槛。

**成本：** 理论乘加2n²→abs加sum约n²，不改变主dense成本。

**验收及停止条件：** D21n误差及最后mean paired对照；0失败。

**依据：** estimators/k3_v29/estimator.py; public source commit 1a4083f89184d6b1cc55d2e55a5a561d2ae50a8d

### v30.016 C16 小核心Sj只计算上三角

编号：WB-P2-A016。类型：算法或计算消融。准备度：先推导再实现。优先级：P2。

**before：** Sj=FAj diag(dA) FAj.T+FPj diag(dP) FPj.T完整GEMM

**after：** 对真正对称的weighted Gram计算上三角block并copy下三角

**详情：** dA/dP可能带符号但Gram仍对称；不能仅贴标签免计算。适用于tier1/core阶段，并保持上游QR流程。

**依赖：** 无条目依赖；仍需通用正确性、数值及资源门槛。

**成本：** r384核心形成有机会约25–50%缩减该组件，copy residual需测。

**验收及停止条件：** signedweights随机float32/64 oracle；FLOPs实际降低与QR轨迹误差控制。

**依据：** estimators/k3_v29/estimator.py; public source commit 1a4083f89184d6b1cc55d2e55a5a561d2ae50a8d

### v30.017 C17 Sg双边旋转的末次乘法对称化

编号：WB-P2-A017。类型：算法或计算消融。准备度：先推导再实现。优先级：P2。

**before：** Sg=Tq @ (Sg @ Tq.T)完整输出

**after：** 保留第一乘法，末次上三角block计算并copy

**详情：** 只对数学对称核心使用，不对非对称FAP套用；Sg存在浮点非对称累积，必须先记录旧Sg asymmetry，不能掩盖大误差。

**依赖：** 无条目依赖；仍需通用正确性、数值及资源门槛。

**成本：** 末次核心GEMM该组件最多约半，整体很小。

**验收及停止条件：** 核心误差/PSD诊断，actual收费以及网络paired测试。

**依据：** estimators/k3_v29/estimator.py; public source commit 1a4083f89184d6b1cc55d2e55a5a561d2ae50a8d

### v30.018 C18 S2子空间投影的末次乘法三角化

编号：WB-P2-A018。类型：算法或计算消融。准备度：先推导再实现。优先级：P2。

**before：** S2=Un.T @ (G2 @ Un)完整r2×r2

**after：** 仅末次计算上三角输出并copy

**详情：** G2理应对称，但Sg/G2舍入可能破坏；与C17分开实验，禁止同时改两处。

**依赖：** 无条目依赖；仍需通用正确性、数值及资源门槛。

**成本：** 该末次GEMM约25–50%减法；整体小。

**验收及停止条件：** same-core synthetic对照和逐层S2asymmetry；FLOPs减少、无资源失败。

**依据：** estimators/k3_v29/estimator.py; public source commit 1a4083f89184d6b1cc55d2e55a5a561d2ae50a8d

### v30.019 C19 joiner两路Gram operator共享布局

编号：WB-P2-A019。类型：合同或资源验证。准备度：规格待实现及验证。优先级：P2。

**before：** Aj.T Om→Aj weighted结果与Pj两路四GEMM

**after：** 组[Aj,Pj]在一个持久layout两batched GEMM，最后相加

**详情：** 数学相同，权重dAj/dPj保留；禁止改变QPASS、sketch、秩、QR值。主要减少四次调用到两次。

**依赖：** 无条目依赖；仍需通用正确性、数值及资源门槛。

**成本：** 主要GEMM FLOPs不变；额外layoutcopy可能无收益。

**验收及停止条件：** Yq与旧式float32差异可控，residual下降、FLOPs不增。

**依据：** estimators/k3_v29/estimator.py; public source commit 1a4083f89184d6b1cc55d2e55a5a561d2ae50a8d

### v30.020 C20 final D3 feedback的线性两项薄收缩

编号：WB-P2-A020。类型：算法或计算消融。准备度：先推导再实现。优先级：P3。

**before：** D3 feedback先显式Xt、Yt完整n²矩阵，再计算A·Yt和Xt·A·w2

**after：** 在trim将两个线性项重排为(AP) @ R2T与(AP·w2) @ R1T再与F2/F1逐行收缩，XtYt交叉项保持旧式

**详情：** 将因式分解用于两项而非声称完整省去Xt/Yt：交叉项仍需它们，所以总cost收益未保证。若两矩阵仍保留，可能只换乘法而不降费，应先算cost。

**依赖：** 无条目依赖；仍需通用正确性、数值及资源门槛。

**成本：** 两线性项由n²elementwise变n²r GEMM可能反而贵，默认先代价诊断；不要直接正式benchmark。

**验收及停止条件：** oracle equality和symbolic FLOP比价，只有预期收费有利才候选。

**依据：** estimators/k3_v29/estimator.py; public source commit 1a4083f89184d6b1cc55d2e55a5a561d2ae50a8d

### v30.021 C21 nonlinearity中广播d3/g4向量不materialize

编号：WB-P2-A021。类型：算法或计算消融。准备度：规格待实现及验证。优先级：P1。

**before：** d3col/g4col先乘ones_row变n²矩阵，再用于AB products

**after：** 只保存n×1/1×n view；fnp广播时构造真正需要的AB

**详情：** 只有d3/g4与ones项本身需在PK2收缩时广播；若einsum不接受broadcastshape需单独分组，不用广播错误缩计量。

**依赖：** v30.011 C11。

**成本：** 减2n²向量广播物化和内存；AB乘法仍完整计费。

**验收及停止条件：** AB每term矩阵与旧式对照，fnp广播contract实测，无异常少收费。

**依据：** estimators/k3_v29/estimator.py; public source commit 1a4083f89184d6b1cc55d2e55a5a561d2ae50a8d

### v30.022 C22 PK1 duplicate一维因子缓存去重

编号：WB-P2-A022。类型：算法或计算消融。准备度：规格待实现及验证。优先级：P3。

**before：** B1stack每termAB但有重复ones,d3,g4pair

**after：** 一维unique AB factors及term indexed选择

**详情：** 只一维缓存，B1输出sum保持coef完全相同；效益极小，可作为机械warmup候选。

**依赖：** 无条目依赖；仍需通用正确性、数值及资源门槛。

**成本：** O(n)少stack/copies，理论分数变化极小。

**验收及停止条件：** PK1float32一致、FLOPs确实减少，否则不列scoregain。

**依据：** estimators/k3_v29/estimator.py; public source commit 1a4083f89184d6b1cc55d2e55a5a561d2ae50a8d

### v30.023 C23 数值对称误差审计

编号：WB-P2-A023。类型：诊断。准备度：规格待实现及验证。优先级：P0。

**before：** 目前X=W C舍入而输出视为对称，没有量化逐层偏差

**after：** 离线包装捕获C,X,Y,按原同一浮点数据比较XY和XY.T

**详情：** 诊断模式只在本地用真实合法fnp结果导出，不进入候选与收费；可用于判定C01/C16–C18风险。

**依赖：** 无条目依赖；仍需通用正确性、数值及资源门槛。

**成本：** 诊断额外计算不作为合法性能/准确性成绩。

**验收及停止条件：** 每层asymmetry相对norm记录，确认其数量级相对于V29重复run漂移。

**依据：** estimators/k3_v29/estimator.py; public source commit 1a4083f89184d6b1cc55d2e55a5a561d2ae50a8d

### v30.024 C24 逐组件实际计费归因

编号：WB-P2-A024。类型：诊断。准备度：规格待实现及验证。优先级：P0。

**before：** 只有全网FLOPs，不知source/PK/core/transport占比

**after：** 单网诊断分别计入明确operator区段，另跑同hash总数复核

**详情：** 不得改变flopscope state、减计费或将诊断helper进入提交。用官方统计/合法上下文或trace日志，若无法隔离则合成shape测量标注估计。

**依赖：** 无条目依赖；仍需通用正确性、数值及资源门槛。

**成本：** 仅诊断，不改变算法；区域总和应解释全网收费。

**验收及停止条件：** 同hash逐网FLOPs完全复核；归因方法可审查，不伪装提交通过。

**依据：** estimators/k3_v29/estimator.py; public source commit 1a4083f89184d6b1cc55d2e55a5a561d2ae50a8d

### v30.025 C25 浮点加法顺序与误差归因

编号：WB-P2-A025。类型：诊断。准备度：规格待实现及验证。优先级：P0。

**before：** 单次raw小差异容易归因新机制

**after：** same-source repeated run，固定线程库和inputs，输出逐层预测difference

**详情：** 该条是候选门槛诊断，不扫系数；保留原frozenreference/基线并另建控制输出。不得利用无收费float64进行提交。 合并L31重复实验漂移任务：需要判定微小收益时做3次独立同网V29控制，FLOPs必须相同，新结果不得覆盖冻结基线。

**依赖：** 无条目依赖；仍需通用正确性、数值及资源门槛。

**成本：** 多个完整V29run；至少测数值漂移与report噪声。

**验收及停止条件：** 小于控制噪声的raw改善不标机制收益；成本变化可独立归因。

**依据：** estimators/k3_v29/estimator.py; public source commit 1a4083f89184d6b1cc55d2e55a5a561d2ae50a8d

### v30.026 C26 持久pool最大shape资源审计

编号：WB-P2-A026。类型：诊断。准备度：规格待实现及验证。优先级：P1。

**before：** pool在width/depth切换后保留历史大buffer，subprocess内存失败归因不清

**after：** 诊断同进程shape序列与峰值RSS/RLIMIT_AS、pool keys/bytes

**详情：** 不放宽8GiB限制后称通过；原8GiB NumPy-backend失败和RPC差异分别保留。只通过对象shape/dtype统计内存，不读数组真值。

**依赖：** 无条目依赖；仍需通用正确性、数值及资源门槛。

**成本：** 诊断；可定位真正active与historicalbytes。

**验收及停止条件：** 记录每shape/predict/RSS/VSZ/失败，报告backend与RPC路径差异。

**依据：** estimators/k3_v29/estimator.py; public source commit 1a4083f89184d6b1cc55d2e55a5a561d2ae50a8d

### v30.027 C27 shape变更时释放无用旧pool

编号：WB-P2-A027。类型：合同或资源验证。准备度：规格待实现及验证。优先级：P2。

**before：** pool保留之前n/L的buffer；当前已有smm重建但_Pool按key存历史shape

**after：** 当模型shape发生变化明确新建_Pool并释放历史引用

**详情：** 仅shape契约管理；固定1024×16网络系列不会改变，收益偏small smoke/混合shape鲁棒性。不能通过predict内GC关/开绕resource measurement。

**依赖：** v30.026 C26。

**成本：** 降低历史占用，固定suite FLOPs基本无变化。

**验收及停止条件：** mixedshape合同、RSS实际降低；同shape输出不变化；setup/predictcaps不变。

**依据：** estimators/k3_v29/estimator.py; public source commit 1a4083f89184d6b1cc55d2e55a5a561d2ae50a8d

### v30.028 C28 分区清理不再需要的scratch常驻

编号：WB-P2-A028。类型：合同或资源验证。准备度：先推导再实现。优先级：P2。

**before：** 最终层仍保留早层宽大bufs，8GiB虚拟内存接近界限

**after：** 明确final-only路径可复用废弃PK2/legs buffers而非申请新scratch

**详情：** 先做liveness图再实现；禁止重用仍被views引用的sourcepool导致无声污染。不可通过释放计费tag而省数学运算。

**依赖：** v30.026 C26。

**成本：** 目标降峰值/VSZ，未必降FLOPs；对象分配/copy残差需实测。

**验收及停止条件：** 小网络allocationpoison oracle、真实local/RPC0失败，FLOPs一致。

**依据：** estimators/k3_v29/estimator.py; public source commit 1a4083f89184d6b1cc55d2e55a5a561d2ae50a8d

### v30.029 C29 三角C_pre阶段批量copy调度

编号：WB-P2-A029。类型：合同或资源验证。准备度：先推导再实现。优先级：P1。

**before：** C01逐blockPython循环触发多次copyto

**after：** 保持相同上三角算法，将同尺度的输入/输出布局用持久块slab按批次copy

**详情：** 唯一主要变化buffer布局/copy调度，不改变triangularregions或GEMM。fnp indexing/reshape调用必须合法且计费完整；是否可batched stridedview需先API审计。 依赖C01仅指现成候选代码和合成核验，不要求比赛快筛完成；先审计调度再运行真实网络。

**依赖：** v30.001 C01。

**成本：** 理论数学FLOPs与C01相同，可降低copyop数与residual，可能增加一次contiguouspacking。

**验收及停止条件：** 输出相对C01数值一致，FLOPs不增；residual更低且8GiB内存可通过。

**依据：** estimators/k3_v29/estimator.py; public source commit 1a4083f89184d6b1cc55d2e55a5a561d2ae50a8d

### v30.030 C30 不同backend上合法matmul计费与resource对照

编号：WB-P2-A030。类型：合同或资源验证。准备度：规格待实现及验证。优先级：P0。

**before：** math/local与RPC、普通subprocess环境行为差异已有但新重组未确认

**after：** 固定candidate/source/data同网络逐operator/API和报告对照

**详情：** 只评估runtime实现差别，不改变flopscope/tool版本当精度收益；如更工具版本必须重跑冻结V29，线上部署参数仍未确认。

**依赖：** v30.001 C01。

**成本：** RPC extra16,777,216FLOPs已知，可按新candidate检查差额；subprocess资源失败保留。

**验收及停止条件：** samehash RPC/local0失败、收费差归因清楚，原resourcecaps不放宽。

**依据：** estimators/k3_v29/estimator.py; public source commit 1a4083f89184d6b1cc55d2e55a5a561d2ae50a8d


## 高阶生产与来源传播

### v30.031 H01 边际D3平方产品补全

编号：WB-P2-A031。类型：算法或计算消融。准备度：规格待实现及验证。优先级：P0。

**before：** 实际V29 p1含D3²w(6,1)/72；p2,p3,p4均只含Gaussian、D3线性、g4线性

**after：** 仅对p2,p3,p4新增D3²w(6,p)/72，p1保持原表

**详情：** 逐项核对TERM_SPECS[(2,)],[(3,)],[(4,)]确认三个幂矩均缺失，非假定遗漏；需新增精确ReLU高阶wick后通过现有PK2K

**依赖：** 无条目依赖；仍需通用正确性、数值及资源门槛。

**成本：** O(n) per layer; no new source state

**验收及停止条件：** Independent small numerical/contract checks, then fixed network0 paired raw+adjusted comparison; expand only on adjusted improvement and zero failures

**依据：** https://arxiv.org/html/2605.05179v2; estimators/k3_v29/estimator.py; https://github.com/504aldo/whest-p2-cumulant-k3/tree/1a4083f89184d6b1cc55d2e55a5a561d2ae50a8d

### v30.032 H02 边际D3乘g4产品补全

编号：WB-P2-A032。类型：算法或计算消融。准备度：规格待实现及验证。优先级：P0。

**before：** 实际V29 p1含D3*g4*w(7,1)/144；p2,p3,p4无该产品

**after：** 仅对p2,p3,p4新增D3*g4*w(7,p)/144

**详情：** 补跨幂矩一致性，保留所有现有均值项；不是已经测过的g4²均值项

**依赖：** 无条目依赖；仍需通用正确性、数值及资源门槛。

**成本：** O(n) per layer; no new source state

**验收及停止条件：** Independent small numerical/contract checks, then fixed network0 paired raw+adjusted comparison; expand only on adjusted improvement and zero failures

**依据：** https://arxiv.org/html/2605.05179v2; estimators/k3_v29/estimator.py; https://github.com/504aldo/whest-p2-cumulant-k3/tree/1a4083f89184d6b1cc55d2e55a5a561d2ae50a8d

### v30.033 H03 单轴截断集合一致性诊断

编号：WB-P2-A033。类型：诊断。准备度：规格待实现及验证。优先级：P1。

**before：** 四个单轴幂矩使用不同cumulant产品集合

**after：** 建立实际逐产品差异表并用独立多项式oracle判断差异是否为有意剪枝

**详情：** 本项仅诊断，不与H01/H02组合成同一提分实验

**依赖：** 无条目依赖；仍需通用正确性、数值及资源门槛。

**成本：** Offline small exact moments and static table audit

**验收及停止条件：** Produce complete product/power matrix and explicit counterexample or verified intentional truncation; no score claim

**依据：** https://arxiv.org/html/2605.05179v2; estimators/k3_v29/estimator.py; https://github.com/504aldo/whest-p2-cumulant-k3/tree/1a4083f89184d6b1cc55d2e55a5a561d2ae50a8d

### v30.034 H04 精确Gaussian pair核替换

编号：WB-P2-A034。类型：算法或计算消融。准备度：先推导再实现。优先级：P1。

**before：** 纯Gaussian pair使用有限Hermite系列

**after：** 用原始Gaussian ReLU幂矩核的合法在线插值替换pK11/pK21/pK22纯Gaussian部分

**详情：** 离线表可用SciPy生成；在线只用计量fnp，必须从mixed source通道中剥离现有纯Gaussian项，处理|rho|近1 pairq3已测负面，只有宽相关区间原核误差诊断显示真实显著缺口后才构建表；不得把机械加q3当新候选。

**依赖：** 无条目依赖；仍需通用正确性、数值及资源门槛。

**成本：** O(n²) per layer plus allowed lookup/interpolation; table memory bounded

**验收及停止条件：** Independent small numerical/contract checks, then fixed network0 paired raw+adjusted comparison; expand only on adjusted improvement and zero failures

**依据：** https://arxiv.org/html/2605.05179v2; estimators/k3_v29/estimator.py; https://github.com/504aldo/whest-p2-cumulant-k3/tree/1a4083f89184d6b1cc55d2e55a5a561d2ae50a8d

### v30.035 H05 Gaussian pair尾项有理重求和

编号：WB-P2-A035。类型：算法或计算消融。准备度：先推导再实现。优先级：P2。

**before：** pure covariance pair finite polynomial

**after：** 加入由Gaussian核解析系数约束的rational tail，同时保持已知低阶项

**详情：** 不能对benchmark真值拟合；不重复单独q3追加实验

**依赖：** 无条目依赖；仍需通用正确性、数值及资源门槛。

**成本：** O(n²) per layer

**验收及停止条件：** Independent small numerical/contract checks, then fixed network0 paired raw+adjusted comparison; expand only on adjusted improvement and zero failures

**依据：** https://arxiv.org/html/2605.05179v2; estimators/k3_v29/estimator.py; https://github.com/504aldo/whest-p2-cumulant-k3/tree/1a4083f89184d6b1cc55d2e55a5a561d2ae50a8d

### v30.036 H06 未剪枝SIMPLE生产表恢复

编号：WB-P2-A036。类型：算法或计算消融。准备度：先推导再实现。优先级：P2。

**before：** V29主体62项剪枝表

**after：** 恢复固定生成器的未剪枝SIMPLE生产表，其余闭合不变

**详情：** 上游报告精度近似不变且residual约+40ms，并非本项目复现；生成器依赖缺失，先恢复静态表

**依赖：** 无条目依赖；仍需通用正确性、数值及资源门槛。

**成本：** O(n²) per layer; op count may violate residual0.4s

**验收及停止条件：** Table provenance and exact generator validation first; one bounded reproduction control, reject if no adjusted gain

**依据：** https://arxiv.org/html/2605.05179v2; estimators/k3_v29/estimator.py; https://github.com/504aldo/whest-p2-cumulant-k3/tree/1a4083f89184d6b1cc55d2e55a5a561d2ae50a8d

### v30.037 H07 WK22到B1出生来源恢复

编号：WB-P2-A037。类型：算法或计算消融。准备度：先推导再实现。优先级：P1。

**before：** B1删除上游d(w3)*WK22*d(w2)类项

**after：** 仅恢复该出生项并扣除其已重复dslices

**详情：** 从lean_k3_aug.py1111–1126分解当前proxy WK22，保留B2 shadow

**依赖：** v30.066 O04、v30.079 O17、v30.080 O18。

**成本：** Target O(n²r) using separable diagonal/core factors; dense extra transported leg must pass budget

**验收及停止条件：** Independent small numerical/contract checks, then fixed network0 paired raw+adjusted comparison; expand only on adjusted improvement and zero failures

**依据：** https://arxiv.org/html/2605.05179v2; estimators/k3_v29/estimator.py; https://github.com/504aldo/whest-p2-cumulant-k3/tree/1a4083f89184d6b1cc55d2e55a5a561d2ae50a8d

### v30.038 H08 B1的WK12对称反馈恢复

编号：WB-P2-A038。类型：算法或计算消融。准备度：先推导再实现。优先级：P1。

**before：** V29只保留部分D21 thin feedback

**after：** 补充上游B1中d(w2)*WK12*d(w2)缺失部分

**详情：** 首先精确对照现有Xt/Yt是否已覆盖，不能重复计量已有D21反馈

**依赖：** 无条目依赖；仍需通用正确性、数值及资源门槛。

**成本：** Target O(n²r) with existing rank16 D21 factors

**验收及停止条件：** Independent small numerical/contract checks, then fixed network0 paired raw+adjusted comparison; expand only on adjusted improvement and zero failures

**依据：** https://arxiv.org/html/2605.05179v2; estimators/k3_v29/estimator.py; https://github.com/504aldo/whest-p2-cumulant-k3/tree/1a4083f89184d6b1cc55d2e55a5a561d2ae50a8d

### v30.039 H09 B2 leading D21乘C来源恢复

编号：WB-P2-A039。类型：算法或计算消融。准备度：先推导再实现。优先级：P1。

**before：** B2出生整体DS-shadow

**after：** 仅恢复WK12乘WK11 leading hub，移除其重复出生dslices

**详情：** 不同于开启完整B2；先恢复sym系数和合法thin representation

**依赖：** v30.066 O04、v30.079 O17、v30.080 O18。

**成本：** Target existing D21 thin factor transport; reject if adds two dense legs per historical source

**验收及停止条件：** Independent small numerical/contract checks, then fixed network0 paired raw+adjusted comparison; expand only on adjusted improvement and zero failures

**依据：** https://arxiv.org/html/2605.05179v2; estimators/k3_v29/estimator.py; https://github.com/504aldo/whest-p2-cumulant-k3/tree/1a4083f89184d6b1cc55d2e55a5a561d2ae50a8d

### v30.040 H10 B2 WK22乘C来源恢复

编号：WB-P2-A040。类型：算法或计算消融。准备度：先推导再实现。优先级：P1。

**before：** B2的WK22乘WK11 fully mixed channel缺失

**after：** 仅恢复该channel而保留B2其他shadow

**详情：** 需分解WK22可分结构并核验birth subtraction

**依赖：** v30.066 O04、v30.079 O17、v30.080 O18。

**成本：** Target O(n²r) per birth/transport; prove rank growth bound

**验收及停止条件：** Independent small numerical/contract checks, then fixed network0 paired raw+adjusted comparison; expand only on adjusted improvement and zero failures

**依据：** https://arxiv.org/html/2605.05179v2; estimators/k3_v29/estimator.py; https://github.com/504aldo/whest-p2-cumulant-k3/tree/1a4083f89184d6b1cc55d2e55a5a561d2ae50a8d

### v30.041 H11 B2 D21平方来源恢复

编号：WB-P2-A041。类型：算法或计算消融。准备度：先推导再实现。优先级：P2。

**before：** B2的WK12乘WK21等D21产品shadow

**after：** 只恢复D21-square symmetric source

**详情：** 低rank乘积可能为r²，先明确组合/去重

**依赖：** 无条目依赖；仍需通用正确性、数值及资源门槛。

**成本：** O(n²r²) candidate target; reject unbounded source rank growth

**验收及停止条件：** Independent small numerical/contract checks, then fixed network0 paired raw+adjusted comparison; expand only on adjusted improvement and zero failures

**依据：** https://arxiv.org/html/2605.05179v2; estimators/k3_v29/estimator.py; https://github.com/504aldo/whest-p2-cumulant-k3/tree/1a4083f89184d6b1cc55d2e55a5a561d2ae50a8d

### v30.042 H12 B2 WK22乘D21来源恢复

编号：WB-P2-A042。类型：算法或计算消融。准备度：先推导再实现。优先级：P2。

**before：** B2高阶mixed birth shadow

**after：** 只恢复WK22乘D21来源轨道

**详情：** 单独验证高阶来源出生dslices，不同时恢复其他B2项

**依赖：** v30.037 H07、v30.038 H08。

**成本：** Target O(n²r²), bounded state

**验收及停止条件：** Independent small numerical/contract checks, then fixed network0 paired raw+adjusted comparison; expand only on adjusted improvement and zero failures

**依据：** https://arxiv.org/html/2605.05179v2; estimators/k3_v29/estimator.py; https://github.com/504aldo/whest-p2-cumulant-k3/tree/1a4083f89184d6b1cc55d2e55a5a561d2ae50a8d

### v30.043 H13 B2保留一次线性运输再shadow

编号：WB-P2-A043。类型：算法或计算消融。准备度：先推导再实现。优先级：P2。

**before：** B2从出生就删除

**after：** 保留完整指定B2 newborn one transition，随后仅保留其已产生边际/slice修正

**详情：** 这是来源类别存在性/有限生命实验，非盲age扫参

**依赖：** v30.066 O04、v30.079 O17、v30.080 O18。

**成本：** One additional newborn source per layer, lives one transition; estimate cumulative O(Ln³)

**验收及停止条件：** Independent small numerical/contract checks, then fixed network0 paired raw+adjusted comparison; expand only on adjusted improvement and zero failures

**依据：** https://arxiv.org/html/2605.05179v2; estimators/k3_v29/estimator.py; https://github.com/504aldo/whest-p2-cumulant-k3/tree/1a4083f89184d6b1cc55d2e55a5a561d2ae50a8d

### v30.044 H14 B2两步输出导向收缩

编号：WB-P2-A044。类型：算法或计算消融。准备度：先推导再实现。优先级：P3。

**before：** B2完全缺失

**after：** 运输指定下一两层mean/variance敏感收缩而不保存完整B2

**详情：** 先证明收缩闭合；future weights legal but true future moments unknown，不能oracle

**依赖：** 无条目依赖；仍需通用正确性、数值及资源门槛。

**成本：** Gate O(Ln³) instead of O(L²n³)

**验收及停止条件：** Independent small numerical/contract checks, then fixed network0 paired raw+adjusted comparison; expand only on adjusted improvement and zero failures

**依据：** https://arxiv.org/html/2605.05179v2; estimators/k3_v29/estimator.py; https://github.com/504aldo/whest-p2-cumulant-k3/tree/1a4083f89184d6b1cc55d2e55a5a561d2ae50a8d

### v30.045 H15 历史K3来源生成K211 capped低rank核心

编号：WB-P2-A045。类型：算法或计算消融。准备度：先推导再实现。优先级：P3。

**before：** 当前g_prev只由K4和K22对角汇总，offdiag由lambda*C代理

**after：** 从actual historical K3 sources求K211 capped，再保留明确lowrank core residual

**详情：** 已有source contraction algebra不等于完整production；需要合法去重和core压缩

**依赖：** 无条目依赖；仍需通用正确性、数值及资源门槛。

**成本：** Heavy source contraction per layer; forbid per-source dense G transport

**验收及停止条件：** Independent small numerical/contract checks, then fixed network0 paired raw+adjusted comparison; expand only on adjusted improvement and zero failures

**依据：** https://arxiv.org/html/2605.05179v2; estimators/k3_v29/estimator.py; https://github.com/504aldo/whest-p2-cumulant-k3/tree/1a4083f89184d6b1cc55d2e55a5a561d2ae50a8d

### v30.046 H16 actual K31低rank核心残差

编号：WB-P2-A046。类型：算法或计算消融。准备度：先推导再实现。优先级：P2。

**before：** lambda*C代理offdiag core

**after：** 从actual K31 production加12/(n+4)*Sym(K31)的去重低rank残差

**详情：** 缺K31 production表；强推理独立恢复后才可编码，不能借弱Gaussian lambda proxy

**依赖：** v30.063 O01、v30.065 O03。

**成本：** O(n²) production plus O(n²r) factor transport

**验收及停止条件：** Independent small numerical/contract checks, then fixed network0 paired raw+adjusted comparison; expand only on adjusted improvement and zero failures

**依据：** https://arxiv.org/html/2605.05179v2; estimators/k3_v29/estimator.py; https://github.com/504aldo/whest-p2-cumulant-k3/tree/1a4083f89184d6b1cc55d2e55a5a561d2ae50a8d

### v30.047 H17 Gaussian K211完整二阶矩阵核心

编号：WB-P2-A047。类型：算法或计算消融。准备度：先推导再实现。优先级：P2。

**before：** 已测Gaussian core只压成scalar lambda且失败

**after：** 保持核验Gaussian capped K211 matrix，作为去重core residual直接运输

**详情：** 不同于重跑coreproxy；需要明确与existing g/lambda*C的投影去重

**依赖：** 无条目依赖；仍需通用正确性、数值及资源门槛。

**成本：** At least O(n³) extra congruence per layer; compare full cost

**验收及停止条件：** Independent small numerical/contract checks, then fixed network0 paired raw+adjusted comparison; expand only on adjusted improvement and zero failures

**依据：** https://arxiv.org/html/2605.05179v2; estimators/k3_v29/estimator.py; https://github.com/504aldo/whest-p2-cumulant-k3/tree/1a4083f89184d6b1cc55d2e55a5a561d2ae50a8d

### v30.048 H18 K4边际真实metric系数

编号：WB-P2-A048。类型：算法或计算消融。准备度：规格待实现及验证。优先级：P0。

**before：** g4row=METRIC_C*dG with fixed mean metric

**after：** 只将边际metric改为current diag(W W^T)，保持Gproxy和其他通道

**详情：** Sym(G tensor M)iiii=Gii*Mii; selected marginal-use ablation, not full exact K4 transport

**依赖：** 无条目依赖；仍需通用正确性、数值及资源门槛。

**成本：** O(n²) row sums per layer

**验收及停止条件：** Independent small numerical/contract checks, then fixed network0 paired raw+adjusted comparison; expand only on adjusted improvement and zero failures

**依据：** https://arxiv.org/html/2605.05179v2; estimators/k3_v29/estimator.py; https://github.com/504aldo/whest-p2-cumulant-k3/tree/1a4083f89184d6b1cc55d2e55a5a561d2ae50a8d

### v30.049 H19 WK431真实metric混合项

编号：WB-P2-A049。类型：算法或计算消融。准备度：先推导再实现。优先级：P1。

**before：** WK431=(METRIC_C/2)*lambda*Coff

**after：** 对固定Gproxy使用0.5*(Gij*Mii+Gii*Mij)，仅改WK431 use

**详情：** 需compute actual M=W W^T和对角排除；不同时修改feed或WK22

**依赖：** 无条目依赖；仍需通用正确性、数值及资源门槛。

**成本：** One Gram O(n³) per layer plus O(n²)

**验收及停止条件：** Independent small numerical/contract checks, then fixed network0 paired raw+adjusted comparison; expand only on adjusted improvement and zero failures

**依据：** https://arxiv.org/html/2605.05179v2; estimators/k3_v29/estimator.py; https://github.com/504aldo/whest-p2-cumulant-k3/tree/1a4083f89184d6b1cc55d2e55a5a561d2ae50a8d

### v30.050 H20 WK22真实metric交叉项

编号：WB-P2-A050。类型：算法或计算消融。准备度：先推导再实现。优先级：P1。

**before：** wk4m=(METRIC_C/6)*(dGi+dGj)

**after：** 使用(Gii*Mjj+Gjj*Mii+4Gij*Mij)/6，仅改WK22 use

**详情：** 固定proxy G；必须验证six sym terms和metered Gram

**依赖：** 无条目依赖；仍需通用正确性、数值及资源门槛。

**成本：** One Gram O(n³) per layer plus O(n²)

**验收及停止条件：** Independent small numerical/contract checks, then fixed network0 paired raw+adjusted comparison; expand only on adjusted improvement and zero failures

**依据：** https://arxiv.org/html/2605.05179v2; estimators/k3_v29/estimator.py; https://github.com/504aldo/whest-p2-cumulant-k3/tree/1a4083f89184d6b1cc55d2e55a5a561d2ae50a8d

### v30.051 H21 恢复非对角metric的K4到K3 feed

编号：WB-P2-A051。类型：算法或计算消融。准备度：先推导再实现。优先级：P3。

**before：** mean diagonal metric下上游某feed项为零

**after：** 使用Moff恢复该被零化feed来源

**详情：** 公开factor公式不足，须找原始factor_k3推导并建立thin factorization

**依赖：** 无条目依赖；仍需通用正确性、数值及资源门槛。

**成本：** Unknown; must prove no n³ tensor and no per-source dense additional history

**验收及停止条件：** Independent small numerical/contract checks, then fixed network0 paired raw+adjusted comparison; expand only on adjusted improvement and zero failures

**依据：** https://arxiv.org/html/2605.05179v2; estimators/k3_v29/estimator.py; https://github.com/504aldo/whest-p2-cumulant-k3/tree/1a4083f89184d6b1cc55d2e55a5a561d2ae50a8d

### v30.052 H22 D21反馈basis输出敏感sketch

编号：WB-P2-A052。类型：算法或计算消融。准备度：先推导再实现。优先级：P2。

**before：** D21 feedback sketch用当前weight列

**after：** 相同rank16下改用未来一两层weight和live average ReLU gate构造sketch

**详情：** 唯一变化sketch objective，no rank/age change；禁止reference sensitivity

**依赖：** 无条目依赖；仍需通用正确性、数值及资源门槛。

**成本：** O(n²r) additional prep per layer

**验收及停止条件：** Independent small numerical/contract checks, then fixed network0 paired raw+adjusted comparison; expand only on adjusted improvement and zero failures

**依据：** https://arxiv.org/html/2605.05179v2; estimators/k3_v29/estimator.py; https://github.com/504aldo/whest-p2-cumulant-k3/tree/1a4083f89184d6b1cc55d2e55a5a561d2ae50a8d

### v30.053 H23 old-source basis按K21收缩能源选择

编号：WB-P2-A053。类型：算法或计算消融。准备度：先推导再实现。优先级：P3。

**before：** old basis按sum leg*leg^T Gram

**after：** 同rank和age按legal current K21-contraction proxy选择basis

**详情：** 需PSD sketch算子，不能SVD truth residual；source contractions若本身贵则没有价值

**依赖：** 无条目依赖；仍需通用正确性、数值及资源门槛。

**成本：** Target O(n²r) sketch plus existing QR

**验收及停止条件：** Independent small numerical/contract checks, then fixed network0 paired raw+adjusted comparison; expand only on adjusted improvement and zero failures

**依据：** https://arxiv.org/html/2605.05179v2; estimators/k3_v29/estimator.py; https://github.com/504aldo/whest-p2-cumulant-k3/tree/1a4083f89184d6b1cc55d2e55a5a561d2ae50a8d

### v30.054 H24 old-source basis按下一层D3敏感方向选择

编号：WB-P2-A054。类型：算法或计算消融。准备度：先推导再实现。优先级：P3。

**before：** 腿Gram不区分mean敏感方向

**after：** 用next W和legal w(3,1)proxy reweight相同rank basis

**详情：** 不能只线性adjoint就声称捕获完整covariance birth feedback

**依赖：** 无条目依赖；仍需通用正确性、数值及资源门槛。

**成本：** Comparable to original basis cost; reject full dense source recomputation

**验收及停止条件：** Independent small numerical/contract checks, then fixed network0 paired raw+adjusted comparison; expand only on adjusted improvement and zero failures

**依据：** https://arxiv.org/html/2605.05179v2; estimators/k3_v29/estimator.py; https://github.com/504aldo/whest-p2-cumulant-k3/tree/1a4083f89184d6b1cc55d2e55a5a561d2ae50a8d

### v30.055 H25 Rres新增解析可分leading项

编号：WB-P2-A055。类型：算法或计算消融。准备度：先推导再实现。优先级：P1。

**before：** Rres=S21-S_sep统一rank16近似

**after：** 从已知生产表提取一个准确第二leading separable项；余项仍rank16

**详情：** 改变representation而非rank；必须明确来源diagram和coef，no residual fitting

**依赖：** 无条目依赖；仍需通用正确性、数值及资源门槛。

**成本：** Target O(n²) via existing A/P carrier

**验收及停止条件：** Independent small numerical/contract checks, then fixed network0 paired raw+adjusted comparison; expand only on adjusted improvement and zero failures

**依据：** https://arxiv.org/html/2605.05179v2; estimators/k3_v29/estimator.py; https://github.com/504aldo/whest-p2-cumulant-k3/tree/1a4083f89184d6b1cc55d2e55a5a561d2ae50a8d

### v30.056 H26 Rres物理机制分组basis

编号：WB-P2-A056。类型：算法或计算消融。准备度：先推导再实现。优先级：P2。

**before：** Gaussian/K3/K4影响混入单一Rres后QR

**after：** 固定总rank16，对三类已明确分解来源形成统一合成basis

**详情：** 只变basis建造方式；不增加总rank或三倍历史storage

**依赖：** 无条目依赖；仍需通用正确性、数值及资源门槛。

**成本：** Several O(n²r) sketches, unchanged transport rank

**验收及停止条件：** Independent small numerical/contract checks, then fixed network0 paired raw+adjusted comparison; expand only on adjusted improvement and zero failures

**依据：** https://arxiv.org/html/2605.05179v2; estimators/k3_v29/estimator.py; https://github.com/504aldo/whest-p2-cumulant-k3/tree/1a4083f89184d6b1cc55d2e55a5a561d2ae50a8d

### v30.057 H27 K3 newborn重复内容一致性oracle

编号：WB-P2-A057。类型：诊断。准备度：规格待实现及验证。优先级：P1。

**before：** birth rep3/rep21从保留hub dslices定义，影子来源移入M

**after：** 独立逐类检查full mixed和repeated内容账是否一致，只有发现具体错配才修复

**详情：** 诊断任务；没有发现错误就完成negative audit，不凭空创造algorithm

**依赖：** 无条目依赖；仍需通用正确性、数值及资源门槛。

**成本：** Small offline fulltensor only; candidate fix expected O(n²)

**验收及停止条件：** Exact small source/original moment checks distinguish truncation from bookkeeping bug

**依据：** https://arxiv.org/html/2605.05179v2; estimators/k3_v29/estimator.py; https://github.com/504aldo/whest-p2-cumulant-k3/tree/1a4083f89184d6b1cc55d2e55a5a561d2ae50a8d

### v30.058 H28 再生core对角和offdiag联合对象闭合

编号：WB-P2-A058。类型：算法或计算消融。准备度：先推导再实现。优先级：P3。

**before：** diag由K4/K22，offdiag由lambda*C分别给出

**after：** 从同一个可计算selected production object产生两部分

**详情：** 不是lambda calibration，需先明确共同对象和no double count

**依赖：** 无条目依赖；仍需通用正确性、数值及资源门槛。

**成本：** Unknown; select representation avoiding full K211 expensive history

**验收及停止条件：** Independent small numerical/contract checks, then fixed network0 paired raw+adjusted comparison; expand only on adjusted improvement and zero failures

**依据：** https://arxiv.org/html/2605.05179v2; estimators/k3_v29/estimator.py; https://github.com/504aldo/whest-p2-cumulant-k3/tree/1a4083f89184d6b1cc55d2e55a5a561d2ae50a8d

### v30.059 H29 K111乘covariance边新来源

编号：WB-P2-A059。类型：算法或计算消融。准备度：先推导再实现。优先级：P1。

**before：** 当前factored source过滤或剪枝缺部分mixed K3+edge diagram

**after：** 恢复一个明确vector partition轨道的mixed新born source

**详情：** 首先检查该轨道是否已由D21 feedback/old Wick transport覆盖，准确missing list才实现

**依赖：** v30.073 O11、v30.074 O12。

**成本：** Must demonstrate O(n³) vs n⁴ exact tensor contraction

**验收及停止条件：** Independent small numerical/contract checks, then fixed network0 paired raw+adjusted comparison; expand only on adjusted improvement and zero failures

**依据：** https://arxiv.org/html/2605.05179v2; estimators/k3_v29/estimator.py; https://github.com/504aldo/whest-p2-cumulant-k3/tree/1a4083f89184d6b1cc55d2e55a5a561d2ae50a8d

### v30.060 H30 K111乘covariance边一层收缩

编号：WB-P2-A060。类型：算法或计算消融。准备度：先推导再实现。优先级：P2。

**before：** 同H29 mixed来源缺失

**after：** 仅追踪H29 newborn下一层K21/variance收缩，一层后删除

**详情：** H29先过formula与coverage audit；有限生命机制单独开关

**依赖：** v30.059 H29。

**成本：** Fixed small number n³ matmuls per layer

**验收及停止条件：** Independent small numerical/contract checks, then fixed network0 paired raw+adjusted comparison; expand only on adjusted improvement and zero failures

**依据：** https://arxiv.org/html/2605.05179v2; estimators/k3_v29/estimator.py; https://github.com/504aldo/whest-p2-cumulant-k3/tree/1a4083f89184d6b1cc55d2e55a5a561d2ae50a8d

### v30.061 H31 exact covariance triangle无偏收缩

编号：WB-P2-A061。类型：算法或计算消融。准备度：先推导再实现。优先级：P3。

**before：** trianglelast用rank16 approximate C

**after：** 用原始Coff的lawful random estimator估计exact alloutput triangle contraction

**详情：** 不是rank sweep；必须先测noise/signal和计量成本，setup seed frozen

**依赖：** 无条目依赖；仍需通用正确性、数值及资源门槛。

**成本：** Samples fit remaining FLOP/residual budget; exact dense baseline O(n⁴)

**验收及停止条件：** Independent tensor small checks and estimator variance bound; reject if standard error exceeds correction signal

**依据：** https://arxiv.org/html/2605.05179v2; estimators/k3_v29/estimator.py; https://github.com/504aldo/whest-p2-cumulant-k3/tree/1a4083f89184d6b1cc55d2e55a5a561d2ae50a8d

### v30.062 H32 高阶newborn直接CP因素生成

编号：WB-P2-A062。类型：算法或计算消融。准备度：先推导再实现。优先级：P3。

**before：** fully mixed K4/highorder未保留

**after：** 直接从current state和Gaussian diagram生成固定数量CP factors供historical transport

**详情：** 禁止先生成n⁴tensor离线分解再装成candidate；rank/state growth要先证明

**依赖：** 无条目依赖；仍需通用正确性、数值及资源门槛。

**成本：** Target O(Ln²r) transport and lawful lowcost birth

**验收及停止条件：** Independent small numerical/contract checks, then fixed network0 paired raw+adjusted comparison; expand only on adjusted improvement and zero failures

**依据：** https://arxiv.org/html/2605.05179v2; estimators/k3_v29/estimator.py; https://github.com/504aldo/whest-p2-cumulant-k3/tree/1a4083f89184d6b1cc55d2e55a5a561d2ae50a8d


## 官方生产与图表核查

### v30.063 O01 官方K31截断生产逐项一致性

编号：WB-P2-A063。类型：诊断。准备度：规格待实现及验证。优先级：P0。

**before：** 仅有K31低阶Gaussian代理及精确moment oracle

**after：** 独立核验完整官方K31基础和augmented表

**详情：** 现有probe证明精确矩转换，不证明Hermite diagram生产。固定输入cum、Wick与预算，比较官方表和独立分区枚举。 必须独立冻结两个口径：get_all_terms_iso(...,augment=True)原始full-generator集合，以及factor=True后factored_keeps_term过滤集合，分别同口径枚举核对；不将因复杂度有意过滤的K111×edge差异判成实现错误或自动恢复。生成项数以保存的inventory为准，不硬编码他人的总项数。

**依赖：** 无条目依赖；仍需通用正确性、数值及资源门槛。

**成本：** small n4/6/8离线O(n²)各项；不执行1024主benchmark

**验收及停止条件：** 逐项系数、方向和合计均符合浮点容差；保留失败项；不宣称提分

**依据：** https://github.com/alignment-research-center/mlp_cumulant_propagation/blob/93d091a4c26c042bfffa28f2e76a81bc0aba94bb/src/mlp_kprop/factor_k3.py

### v30.064 O02 官方capped K211逐项生产一致性

编号：WB-P2-A064。类型：诊断。准备度：规格待实现及验证。优先级：P0。

**before：** 仅有source contraction和power转换独立probe

**after：** 完整capped diagram生产与显式small tensor一致

**详情：** 核验基础3项与augmented20项capped生产；a不得等于b或c，b与c也按输出diagslice约定排除。 必须独立冻结两个口径：get_all_terms_iso(...,augment=True)原始full-generator集合，以及factor=True后factored_keeps_term过滤集合，分别同口径枚举核对；不将因复杂度有意过滤的K111×edge差异判成实现错误或自动恢复。生成项数以保存的inventory为准，不硬编码他人的总项数。

**依赖：** 无条目依赖；仍需通用正确性、数值及资源门槛。

**成本：** small n4/6/8离线显式O(n³)，生产表达另记成本

**验收及停止条件：** 逐项与独立枚举一致，尤其重复排除与轴方向

**依据：** https://github.com/alignment-research-center/mlp_cumulant_propagation/blob/93d091a4c26c042bfffa28f2e76a81bc0aba94bb/src/mlp_kprop/factor_k3.py

### v30.065 O03 K211完整来源抵消一致性

编号：WB-P2-A065。类型：诊断。准备度：规格待实现及验证。优先级：P0。

**before：** 组合部件分别通过

**after：** 一次完整非线性核心生产逐层对照

**详情：** 将old-source、birth、pK211与-2mean*K111以及-2CovCov组合为一个完整测试；禁止单独裁掉大正负项。 diagram equality只与相同截断预算和过滤集合比较；相对exact finite-distribution矩的差另记为截断近似误差，不混称production parity失败。

**依赖：** v30.063 O01、v30.064 O02。

**成本：** small n4/6，允许离线full tensor

**验收及停止条件：** 官方production与独立完整moment/cumulant核对一致；记录截断误差和算术误差分别

**依据：** https://github.com/alignment-research-center/mlp_cumulant_propagation/blob/93d091a4c26c042bfffa28f2e76a81bc0aba94bb/src/mlp_kprop/factor_k3.py

### v30.066 O04 旧lean与当前官方generator语义归因

编号：WB-P2-A066。类型：诊断。准备度：规格待实现及验证。优先级：P0。

**before：** 仅完成pair-output False/True diff

**after：** 完整source/output的exact factor inventory

**详情：** 当前pair输出新增22条中19已在V29，3条pairq3已测负。扩展映射其他输出，严禁重复加同义项。

**依赖：** 无条目依赖；仍需通用正确性、数值及资源门槛。

**成本：** 纯组合AST抽取，无Torch巨额安装

**验收及停止条件：** 逐项给出存在/遗漏/剪枝/等价重组状态和固定源码hash

**依据：** https://github.com/alignment-research-center/mlp_cumulant_propagation/blob/93d091a4c26c042bfffa28f2e76a81bc0aba94bb/src/mlp_kprop/kprop_harmonic.py

### v30.067 O05 完整K31/K211生产核的离线表示上限

编号：WB-P2-A067。类型：诊断。准备度：先推导再实现。优先级：P1。

**before：** Goff=layer lambda*Coff

**after：** 完整production Gnew与regen G成对诊断

**详情：** 使用合法传播状态生成核，参考只用于诊断最终误差，不能作为candidate输入。先small shapes再有界实际层。

**依赖：** v30.063 O01、v30.064 O02、v30.065 O03。

**成本：** 可能多source n³及capped n³；先账本后正式shape

**验收及停止条件：** 发现远大于浮点漂移的核失配及误差潜力；成本不可用则只保留诊断

**依据：** https://github.com/alignment-research-center/mlp_cumulant_propagation/blob/93d091a4c26c042bfffa28f2e76a81bc0aba94bb/src/mlp_kprop/factor_k3.py

### v30.068 O06 完整生产核单层替换诊断

编号：WB-P2-A068。类型：算法或计算消融。准备度：先推导再实现。优先级：P2。

**before：** 全部层lambda核

**after：** 预先指定单层完整production核

**详情：** 在查看候选成绩前固定一层，只该层使用完整production核，之后恢复regen。不得逐层挑最好冒充泛化。

**依赖：** v30.067 O05。

**成本：** 一次额外完整核production，实际成本待O05

**验收及停止条件：** 同网络raw/adjusted和失败成对记录，明显改善才扩量

**依据：** https://github.com/alignment-research-center/mlp_cumulant_propagation/blob/93d091a4c26c042bfffa28f2e76a81bc0aba94bb/src/mlp_kprop/factor_k3.py

### v30.069 O07 完整核对角修正非零性与候选

编号：WB-P2-A069。类型：算法或计算消融。准备度：先推导再实现。优先级：P2。

**before：** gprev由K4+K22 rowsum

**after：** 加入由完整trace投影证明非零的对角修正

**详情：** 先判定K31/K211 capped输出对角是否严格zero；若zero取消。若非零仅改变gprev对角生产。

**依赖：** v30.063 O01、v30.064 O02、v30.065 O03。

**成本：** 核production成本；禁止声称仅O(n)而隐去收缩

**验收及停止条件：** 非零性有公式证据且small tensor parity，通过后真实score筛查

**依据：** https://github.com/alignment-research-center/mlp_cumulant_propagation/blob/93d091a4c26c042bfffa28f2e76a81bc0aba94bb/src/mlp_kprop/harmonic.py

### v30.070 O08 完整production核仅USE替换

编号：WB-P2-A070。类型：算法或计算消融。准备度：先推导再实现。优先级：P2。

**before：** USE wk431=lambda*Cpreoff

**after：** USE wk431=完整production核运输结果

**详情：** 与已测weighted-Gram spillUSE不同：此次对象是K31/K211产生的真正offdiag核，不重复叠加旧spill。

**依赖：** v30.067 O05。

**成本：** 完整核与运输成本先计量

**验收及停止条件：** 机制单一，feed保持原公式；实质adjusted改善才扩量

**依据：** https://github.com/alignment-research-center/mlp_cumulant_propagation/blob/93d091a4c26c042bfffa28f2e76a81bc0aba94bb/src/mlp_kprop/factor_k3.py

### v30.071 O09 完整production核仅FEED替换

编号：WB-P2-A071。类型：算法或计算消融。准备度：先推导再实现。优先级：P2。

**before：** FEED X3使用regen diagonal+lambda

**after：** FEED X3使用完整production核

**详情：** X3使用完整核；必须保留新增dense leg历史，出生重复项统一扣除，USE仍原V29。

**依赖：** v30.067 O05。

**成本：** 新增dense leg至少n³/source层；先成本门槛

**验收及停止条件：** 小tensor出生/运输/D3D21一致，资源合法，score明显改善

**依据：** https://github.com/alignment-research-center/mlp_cumulant_propagation/blob/93d091a4c26c042bfffa28f2e76a81bc0aba94bb/src/mlp_kprop/factor_k3.py

### v30.072 O10 完整核剩余误差的state模态诊断

编号：WB-P2-A072。类型：诊断。准备度：规格待实现及验证。优先级：P2。

**before：** 单lambda mode

**after：** 增加预先定义state模态的离线解释力测量

**详情：** 目标是官方合法production核，不是reference真值。测试固定live-state低维模态能否解释lambda剩余误差，不盲扫模态。

**依赖：** v30.067 O05。

**成本：** 离线核保存和n²回归诊断

**验收及停止条件：** 网络分割后捕获一致显著能量；能量分散则关闭

**依据：** https://github.com/alignment-research-center/mlp_cumulant_propagation/blob/93d091a4c26c042bfffa28f2e76a81bc0aba94bb/src/mlp_kprop/factor_k3.py

### v30.073 O11 K111乘covariance edge完整公式核验

编号：WB-P2-A073。类型：诊断。准备度：规格待实现及验证。优先级：P0。

**before：** 只有静态遗漏判定

**after：** 独立显式tensor与diagram枚举一致

**详情：** deltaTijk=K111ijk*(w1_i*w2_j*w2_k*Cjk+循环)，distinct索引；官方coef3Sym。

**依赖：** 无条目依赖；仍需通用正确性、数值及资源门槛。

**成本：** small n4/6/8 O(n³)离线

**验收及停止条件：** 符号、coef、mask、中心化均通过独立验证

**依据：** https://github.com/alignment-research-center/mlp_cumulant_propagation/blob/93d091a4c26c042bfffa28f2e76a81bc0aba94bb/src/mlp_kprop/kprop_harmonic.py

### v30.074 O12 K111乘edge的exact final D3成本证明

编号：WB-P2-A074。类型：诊断。准备度：先推导再实现。优先级：P1。

**before：** 仅已知rank扩大障碍

**after：** 证明exact复杂度和内存或给出反例

**详情：** 现有CP rank通常乘rank(C)，不能无依据称O(n³)。推导可执行收缩顺序和所有中间维度。

**依赖：** v30.073 O11。

**成本：** 符号/小shape计数；若O(n⁴)则记录阻碍

**验收及停止条件：** 全部array维度与FLOP公式可审计；无隐藏n³tensor

**依据：** https://github.com/alignment-research-center/mlp_cumulant_propagation/blob/93d091a4c26c042bfffa28f2e76a81bc0aba94bb/src/mlp_kprop/factor_k3.py

### v30.075 O13 K111乘edge covariance主模态代理

编号：WB-P2-A075。类型：算法或计算消融。准备度：先推导再实现。优先级：P2。

**before：** 该mixed diagram遗漏

**after：** 使用live covariance主模态近似edge并保留K111来源

**详情：** 只有O12证明exact不经济且live C存在明确主模态后才做。固定一项主模态，不扫描rank；先last birth。

**依赖：** v30.073 O11、v30.074 O12。

**成本：** eig/rangefinder和rank扩张都入账；需small测量

**验收及停止条件：** 目标无关选方向、重复扣除正确、近似收缩误差受控、真实score改善

**依据：** https://github.com/alignment-research-center/mlp_cumulant_propagation/blob/93d091a4c26c042bfffa28f2e76a81bc0aba94bb/src/mlp_kprop/factor_k3.py

### v30.076 O14 K111乘edge代理误差门槛

编号：WB-P2-A076。类型：诊断。准备度：规格待实现及验证。优先级：P2。

**before：** proxy仅通过代数

**after：** proxy与exact收缩成对误差评估

**详情：** 单独评估O13在small exact张量上的D3/D21收缩误差，不只看covariance能量。

**依赖：** v30.075 O13。

**成本：** small tensor offline O(n³)

**验收及停止条件：** 若关键收缩误差大取消，不用rank扫参数掩盖

**依据：** https://github.com/alignment-research-center/mlp_cumulant_propagation/blob/93d091a4c26c042bfffa28f2e76a81bc0aba94bb/src/mlp_kprop/factor_k3.py

### v30.077 O15 doubled-edge早期来源贡献门槛

编号：WB-P2-A077。类型：诊断。准备度：先推导再实现。优先级：P3。

**before：** 已否证last-birth三matmul候选

**after：** 有界诊断早期birth跨层贡献

**详情：** doubleedge-final已测零失败但adjusted恶化约1.10%，不得列重跑默认。只有早期birth离线贡献明显才考虑fullhistory。

**依赖：** 无条目依赖；仍需通用正确性、数值及资源门槛。

**成本：** small fullhistory sources；1024额外cost先证明

**验收及停止条件：** 证明明显早期raw潜力和可负担费用，否则关闭fullhistory路线

**依据：** https://github.com/alignment-research-center/mlp_cumulant_propagation/blob/93d091a4c26c042bfffa28f2e76a81bc0aba94bb/src/mlp_kprop/factor_k3.py

### v30.078 O16 covariance triangle exact多层small诊断

编号：WB-P2-A078。类型：诊断。准备度：先推导再实现。优先级：P3。

**before：** 仅低秩last-only阴性证据

**after：** exact small triangle跨层传播

**详情：** rank16/last-only已测负。使用exact all-distinct triangle small形状区分diagram本身小与近似丢信息。

**依赖：** 无条目依赖；仍需通用正确性、数值及资源门槛。

**成本：** small n4/6/8 exact O(n³) tensor；不扩1024

**验收及停止条件：** exact仍小则停止，不扫rank；否则需新可计量表示

**依据：** https://github.com/alignment-research-center/mlp_cumulant_propagation/blob/93d091a4c26c042bfffa28f2e76a81bc0aba94bb/src/mlp_kprop/kprop_harmonic.py

### v30.079 O17 当前官方WK22 path与旧lean等价审计

编号：WB-P2-A079。类型：诊断。准备度：规格待实现及验证。优先级：P3。

**before：** 旧复合X/Y生产

**after：** 逐diagram核对current official WK22路径

**详情：** 上游F69旧no_wk22机制报告0%，不重复跑。仅核对当前budget线性path与旧复合birth是否同对象。

**依赖：** 无条目依赖；仍需通用正确性、数值及资源门槛。

**成本：** 纯表项+smalltensor，无真实benchmark

**验收及停止条件：** 若等价则登记已上游阴性；若不同再提出独立候选

**依据：** https://github.com/alignment-research-center/mlp_cumulant_propagation/blob/93d091a4c26c042bfffa28f2e76a81bc0aba94bb/src/mlp_kprop/factor_k3.py

### v30.080 O18 现有birth薄反馈乘积预算审计

编号：WB-P2-A080。类型：诊断。准备度：规格待实现及验证。优先级：P0。

**before：** 复合因子乘积未完整power-count

**after：** 每个出生乘积标明预算与重复补偿

**详情：** 展开V29 X1*Y1并映射diagram，识别feedback×feedback是否超官方aug预算；先审计不删除。

**依赖：** 无条目依赖；仍需通用正确性、数值及资源门槛。

**成本：** symbolic expansion和smalltensor

**验收及停止条件：** 每项位置/coef/cost可追溯，不能把已测NO_FB再包装

**依据：** https://github.com/alignment-research-center/mlp_cumulant_propagation/blob/93d091a4c26c042bfffa28f2e76a81bc0aba94bb/src/mlp_kprop/kprop_harmonic.py

### v30.081 O19 K5边缘生产闭合

编号：WB-P2-A081。类型：算法或计算消融。准备度：先推导再实现。优先级：P1。

**before：** 无独立K5状态

**after：** 经验证的选择性marginal K5生产

**详情：** 提取官方K4-AUG degree5可用marginal项；完整K5是r1三阶core，边缘不能称完整K5。

**依赖：** 无条目依赖；仍需通用正确性、数值及资源门槛。

**成本：** 生产成本未知；先table与state依赖账本

**验收及停止条件：** 全部所需cum输入合法，无MC真值，明确遗漏cross terms

**依据：** https://github.com/alignment-research-center/mlp_cumulant_propagation/blob/93d091a4c26c042bfffa28f2e76a81bc0aba94bb/src/mlp_kprop/factor_k4.py

### v30.082 O20 K5独立边缘readout

编号：WB-P2-A082。类型：算法或计算消融。准备度：先推导再实现。优先级：P2。

**before：** PK1无kappa5线性项

**after：** PK1加入合法K5 marginal readout

**详情：** 仅加入kappa5*w(5,p)/120，其他新高阶项关闭。先final-only。

**依赖：** v30.081 O19、v30.087 O25。

**成本：** 已有K5后的O(n)Wick/readout；生产成本另计

**验收及停止条件：** 系数small独立积分验证，真实score收益明显，限制披露

**依据：** https://github.com/alignment-research-center/mlp_cumulant_propagation/blob/93d091a4c26c042bfffa28f2e76a81bc0aba94bb/src/mlp_kprop/factor_k4.py

### v30.083 O21 K6纯径向投影生产公式

编号：WB-P2-A083。类型：诊断。准备度：先推导再实现。优先级：P1。

**before：** 无K6channel

**after：** 完整可审计radial scalar生产定义

**详情：** 官方rout3 degree6输出scalar，生产未必O(n²)。推导trace/coarsening与diagslice排除全部来源。

**依赖：** 无条目依赖；仍需通用正确性、数值及资源门槛。

**成本：** potential highdim traces；先解析成本门槛

**验收及停止条件：** exact small moments/trace投影一致，生产费用不隐去

**依据：** https://github.com/alignment-research-center/mlp_cumulant_propagation/blob/93d091a4c26c042bfffa28f2e76a81bc0aba94bb/src/mlp_kprop/factor_k4.py

### v30.084 O22 K6径向状态合法线性运输

编号：WB-P2-A084。类型：算法或计算消融。准备度：先推导再实现。优先级：P2。

**before：** 无K6运输

**after：** 加入radial harmonic state W运输

**详情：** 使用实际W-metric，核查average metric2替代误差，不把metric convention错认coef bug。

**依赖：** v30.083 O21。

**成本：** scalar/harmonic contractions依公式，先small测量

**验收及停止条件：** small exact K6 transport parity通过，metric约定冻结

**依据：** https://github.com/alignment-research-center/mlp_cumulant_propagation/blob/93d091a4c26c042bfffa28f2e76a81bc0aba94bb/src/mlp_kprop/harmonic.py

### v30.085 O23 K6独立边缘readout

编号：WB-P2-A085。类型：算法或计算消融。准备度：先推导再实现。优先级：P2。

**before：** PK1无kappa6线性项

**after：** 合法K6对角读入PK1

**详情：** 只加入kappa6*w(6,p)/720；与已测k4square/w8不同。并非完整higher-orderEdgeworth。

**依赖：** v30.083 O21、v30.084 O22、v30.087 O25。

**成本：** readout O(n)，生产运输收费

**验收及停止条件：** small系数/中心化正确，资源合法且真实adjusted改善

**依据：** https://github.com/alignment-research-center/mlp_cumulant_propagation/blob/93d091a4c26c042bfffa28f2e76a81bc0aba94bb/src/mlp_kprop/factor_k4.py

### v30.086 O24 高阶边缘同阶组合inventory

编号：WB-P2-A086。类型：诊断。准备度：规格待实现及验证。优先级：P0。

**before：** 选择性补项缺完整预算说明

**after：** budget和exact coefficient可追溯inventory

**详情：** 标明K3K4已有、K4²已测负、K5/K6新增及同阶必须配套内容；先避免不一致truncation。

**依赖：** 无条目依赖；仍需通用正确性、数值及资源门槛。

**成本：** 纯组合生成器

**验收及停止条件：** 重复项0、每项before/after明确，不把组合高阶称strictK3

**依据：** https://github.com/alignment-research-center/mlp_cumulant_propagation/blob/93d091a4c26c042bfffa28f2e76a81bc0aba94bb/src/mlp_kprop/kprop_harmonic.py

### v30.087 O25 degree5/6 ordinary与power累积量独立转换

编号：WB-P2-A087。类型：诊断。准备度：规格待实现及验证。优先级：P1。

**before：** 仅degree≤4转换probe

**after：** degree5/6转换独立parity

**详情：** 使用finite support或Gaussian polynomial真实矩与set partitions，验证生产转换，区别ReLU Wick积分。

**依赖：** 无条目依赖；仍需通用正确性、数值及资源门槛。

**成本：** small finite-support O(samples*n^d)仅离线受限shape

**验收及停止条件：** 独立转换与setpartition误差达到数值容差，非实现自我镜像

**依据：** https://github.com/alignment-research-center/mlp_cumulant_propagation/blob/93d091a4c26c042bfffa28f2e76a81bc0aba94bb/src/mlp_kprop/cumulants.py

### v30.088 O26 图表收益的上游调参分割审计

编号：WB-P2-A088。类型：诊断。准备度：规格待实现及验证。优先级：P1。

**before：** 候选网络分割标签不够细

**after：** 逐机制报告训练/selection/confirmation来源矩阵

**详情：** L21负责public解封；本项只确认本研究候选production/constants是否依赖同public网络，记录upstream训练范围未知。

**依赖：** v30.116 L21。

**成本：** 只读provenance和已有日志，无重复benchmark

**验收及停止条件：** 不能把IDs8-15独立于本地调参称独立于upstream

**依据：** https://github.com/alignment-research-center/mlp_cumulant_propagation/blob/93d091a4c26c042bfffa28f2e76a81bc0aba94bb/src/mlp_kprop/kprop_harmonic.py

### v30.089 O27 封存新分片的diagslice状态分布偏移

编号：WB-P2-A089。类型：诊断。准备度：规格待实现及验证。优先级：P2。

**before：** 只有最终分数分割

**after：** 新分片state-only机制分布诊断

**详情：** 区别L21解封得分：比较合法prestate的variance/skew/kurtosis与核系数分布，预先冻结特征不用target。

**依赖：** v30.116 L21。

**成本：** 缓存state O(n²)统计，无target回归

**验收及停止条件：** 冻结特征和统计规则，分布差异只诊断不能调候选后继续称holdout

**依据：** https://github.com/alignment-research-center/mlp_cumulant_propagation/blob/93d091a4c26c042bfffa28f2e76a81bc0aba94bb/src/mlp_kprop/factor_k3.py

### v30.090 O28 新seed网络的diagram规模law核验

编号：WB-P2-A090。类型：诊断。准备度：规格待实现及验证。优先级：P2。

**before：** 只有fixed1024表规模预期

**after：** 预定width/seed上的state diagram rms law

**详情：** L20负责最终确认；本项只核验生成网络上K111 edge/capped211等公式power-count随width变化，reference不参与方向选择。

**依赖：** v30.115 L20、v30.073 O11、v30.064 O02。

**成本：** small宽度固定3个点，避免全K4巨额tensor

**验收及停止条件：** 记录scale置信/样本不确定性，不能用小width结论直接保证1024提分

**依据：** https://github.com/alignment-research-center/mlp_cumulant_propagation/blob/93d091a4c26c042bfffa28f2e76a81bc0aba94bb/src/mlp_kprop/kprop_harmonic.py

### v30.091 O29 高阶校正信号与reference不确定性的分离

编号：WB-P2-A091。类型：诊断。准备度：规格待实现及验证。优先级：P1。

**before：** gain与noise proxy混用

**after：** paired候选信号和reference误差分离账本

**详情：** L19生成独立references；本项使用冻结候选delta prediction本身的方向/rms与reference差异比较，判断tiny gain可验证性。

**依赖：** v30.114 L19。

**成本：** 已有预测矩阵与两references离线O(depth*n)

**验收及停止条件：** 不以variance/N当实测CI，不通过换弱target创造收益

**依据：** https://arxiv.org/html/2605.05179v2

### v30.092 O30 新增图表array操作计量路径合同

编号：WB-P2-A092。类型：合同或资源验证。准备度：规格待实现及验证。优先级：P0。

**before：** 只运行small数值公式

**after：** actual fnp路径日志与成本逐算子核验

**详情：** 区别泛化/RPC一揽子L检查：专门核对新op的dtype、复制、out缓冲、symmetry和same-object输入，fnp成本不得旁路。

**依赖：** 无条目依赖；仍需通用正确性、数值及资源门槛。

**成本：** small flopscope execution；仅有改善候选才正式RPC

**验收及停止条件：** 无numpy candidate计算，无compilation或concurrency，codehash和计量环境冻结

**依据：** https://github.com/AIcrowd/whest-starterkit/blob/main/docs/concepts/allowed-code.md

### v30.093 O31 完整K31生产对称化与轴方向合同

编号：WB-P2-A093。类型：合同或资源验证。准备度：规格待实现及验证。优先级：P1。

**before：** K31方向未actual生产确认

**after：** actual fnp K31输出与固定axis枚举

**详情：** 额外审计pK31是row power3 col1，iso classes只有equalmultiplicity轴可对称化；不能擅自对pK31平均转置。

**依赖：** v30.063 O01。

**成本：** small O(n²)比较

**验收及停止条件：** 非对称输入下row3/col1逐项一致，harmonic Sym只在G组装位置执行

**依据：** https://github.com/alignment-research-center/mlp_cumulant_propagation/blob/93d091a4c26c042bfffa28f2e76a81bc0aba94bb/src/mlp_kprop/factor_k3.py

### v30.094 O32 capped211 diagonal mask因式表示合同

编号：WB-P2-A094。类型：合同或资源验证。准备度：规格待实现及验证。优先级：P0。

**before：** mask分布在多个函数

**after：** mask逐阶段追踪与explicit枚举一致

**详情：** eval_capped公开lean只排a=b,a=c，输出b=c是否由外层zero-diag保障需explicit核对；测试narrow和coincidence边界。

**依赖：** v30.064 O02。

**成本：** small O(n³) oracle仅诊断

**验收及停止条件：** a=b/c和b=c均按定义处理，一次扣除无漏项无双扣

**依据：** https://github.com/alignment-research-center/mlp_cumulant_propagation/blob/93d091a4c26c042bfffa28f2e76a81bc0aba94bb/src/mlp_kprop/factor_k3.py

### v30.095 O33 complete harmonic core actualW transport对照

编号：WB-P2-A095。类型：诊断。准备度：规格待实现及验证。优先级：P1。

**before：** transport代数与production分离

**after：** 完整生成G的actualW数值parity

**详情：** 当前高斯代理失败不证明full production核；在small上对actualW G W.T和其trace/marginals验证METRIC_C与birthmetric1区分。

**依赖：** v30.065 O03。

**成本：** small O(n³)matmul

**验收及停止条件：** 投影metric1与transportmetric2不能混用，误差仅解释算术非提分

**依据：** https://github.com/alignment-research-center/mlp_cumulant_propagation/blob/93d091a4c26c042bfffa28f2e76a81bc0aba94bb/src/mlp_kprop/harmonic.py


## 残差修正与独立验证

### v30.096 L01 冻结原轨迹特征缓存

编号：WB-P2-A096。类型：诊断。准备度：规格待实现及验证。优先级：P0。

**before：** 只有 aggregate 与逐层 MSE；没有可拟合的逐神经元内部特征缓存。

**after：** 新建 V29 诊断副本，保存原 13 特征、原预测和网络 ID；不改变预测公式。

**详情：** 对开发 IDs0–7 提取每层 ones,mu,var,sigma,alpha,abs(alpha),phi,Phi,pk1,D3,D21norm,K4_sigma,sigma*phi。输出诊断数组至 ignored benchmark/data；预测/FLOPs 与同轮 V29 控制成对核对。序列化在预测计量区外，最终候选不含诊断输出。

**依赖：** 无条目依赖；仍需通用正确性、数值及资源门槛。

**成本：** 缓存约 8×16×1024×13 float32；无候选性能声明。

**验收及停止条件：** 原始预测偏差不超过同轮重复控制，FLOPs相同；manifest有特征顺序和哈希。

**依据：** estimators/k3_v29/estimator.py:1215; 项目报告7.5

### v30.097 L02 冻结网络级训练选择流程

编号：WB-P2-A097。类型：诊断。准备度：规格待实现及验证。优先级：P0。

**before：** Mini8同时用于既有快筛；缺少修正器专用网络级选择协议。

**after：** IDs0–7用leave-one-network-out；IDs8–15保留为一次公共确认集；另列新种子最终独立集。

**详情：** 保存split manifest及选择规则。任何特征/模型选择只能使用0–7的网络级交叉验证；不可把同网神经元随机分到训练和验证。8–15与上游调参独立性未知，不能叫真正未见最终集。

**依赖：** v30.096 L01。

**成本：** 离线拟合；不新增提交计算。

**验收及停止条件：** 确认集标签在选择前不读取；日志记录每次选择，选定一个候选才解封公共确认。

**依据：** ARC_WhestBench_Project_Report_2026-10-10.md:241

### v30.098 L03 最后层13特征残差线性修正

编号：WB-P2-A098。类型：算法或计算消融。准备度：规格待实现及验证。优先级：P1。

**before：** 原V29固定CORR_BETA后仍有final残差。

**after：** 只在final输出追加一个开发网络拟合的13特征线性残差预测；早层与原Beta不变。

**详情：** 用L01缓存的final合法特征拟合 y=reference−原V29预测，训练集标准化，固定ridge alpha=1e-3于标准化特征，不调alpha。常数、标准化和系数随候选冻结；只追加输出，不写回早层状态。 本任务先拟合一次ridge，在leave-one-network-out评估中不扫正则；若多个头/特征都被查看，必须计入选择账本。

**依赖：** v30.096 L01、v30.097 L02。

**成本：** O(13n)推理；拟合可离线NumPy，推理仅fnp。

**验收及停止条件：** 网络级CV平均adjusted收益显著超过同轮重复漂移；负结果停止该模型；最终需独立网络确认。

**依据：** 项目报告7.5; V29 CORR_BETA

### v30.099 L04 删除常数特征的修正对照

编号：WB-P2-A099。类型：算法或计算消融。准备度：规格待实现及验证。优先级：P2。

**before：** L03含截距特征，可能只学习整体偏置。

**after：** 只删除ones列，保留其余12列及同样训练流程。

**详情：** 检查收益是否来自神经元结构而非整体漂移；禁止为此访问确认集。

**依赖：** v30.098 L03。

**成本：** O(12n)。

**验收及停止条件：** 相同网络级CV相对L03比较；解释误差减少来源。

**依据：** results/phase2/v29_rpc_residual_structure.json

### v30.100 L05 阈值形状特征增量

编号：WB-P2-A100。类型：算法或计算消融。准备度：规格待实现及验证。优先级：P2。

**before：** L03仅含alpha及abs(alpha)。

**after：** 追加alpha²phi、alpha³phi、alpha⁴phi三个确定性特征。

**详情：** 只扩一个阈值Hermite形状特征块，保留L03其他设置；这些是合法状态函数，不能声称完整高阶项。

**依赖：** v30.098 L03。

**成本：** O(n)逐元素和3n线性点积。

**验收及停止条件：** 开发网络级CV稳定改善，误差与FLOPs都记录；无收益删去整个块。

**依据：** 原论文S3.4; 项目报告7.5

### v30.101 L06 标准化偏度特征增量

编号：WB-P2-A101。类型：算法或计算消融。准备度：规格待实现及验证。优先级：P2。

**before：** L03直接用D3，未表达相对于方差的规模。

**after：** 追加D3/(sigma³)与该值乘phi的两列，分母仅使用原正sigma。

**详情：** 不替换传播D3或人为clip；若出现非有限值则失败，不修改原reference。

**依赖：** v30.098 L03。

**成本：** O(n)，全部fnp计费。

**验收及停止条件：** 网络级CV改善且极端sigma网络零非有限；报告新增计算。

**依据：** V29 D3与sigma状态

### v30.102 L07 标准化峰度特征增量

编号：WB-P2-A102。类型：算法或计算消融。准备度：规格待实现及验证。优先级：P2。

**before：** L03含全层K4_sigma，但没有每神经元g4row。

**after：** 追加g4row/sigma⁴及乘phi的两列。

**详情：** 从合法pre状态提取；仅学习输出残差，不将该特征当已证明正确κ4。

**依赖：** v30.098 L03。

**成本：** O(n)，不新增高阶运输。

**验收及停止条件：** 相同训练网络CV改善；与L06单独比较，不先合并。

**依据：** estimators/k3_v29/estimator.py:1109

### v30.103 L08 最后线性均值传播轨迹特征

编号：WB-P2-A103。类型：算法或计算消融。准备度：规格待实现及验证。优先级：P2。

**before：** L03仅用最后pre的本层13特征。

**after：** 追加W_final乘上一层预测均值误差代理向量与本层mu的差异特征。

**详情：** 代理先定义为上一层原CORR_BETA修正delta，新增列W_final@delta_prev及其乘Phi；不使用上一层真值。缓存上层delta并计入新增matvec。

**依赖：** v30.098 L03。

**成本：** 1次n² matvec+O(n)。

**验收及停止条件：** CV收益超过计算折扣损失；早层输出及状态完全保留。

**依据：** V29 delta与mu递推; 项目报告7.5

### v30.104 L09 前层方差轨迹特征

编号：WB-P2-A104。类型：算法或计算消融。准备度：规格待实现及验证。优先级：P2。

**before：** L03缺少方差的来源分解。

**after：** 追加(W²)@var_prev与var−(W²)@var_prev两列。

**详情：** 复用当前dG计算中的t_v/var_prev来源，避免重复matvec；只新增特征块。

**依赖：** v30.098 L03。

**成本：** 若能合法复用O(n)；否则1 n²matvec并记录。

**验收及停止条件：** 独立核对特征与定义；CV improved adjusted才保留。

**依据：** V29 t_v = var−WW@var_prev

### v30.105 L10 跨神经元均值残差特征

编号：WB-P2-A105。类型：算法或计算消融。准备度：规格待实现及验证。优先级：P2。

**before：** L03每神经元只看局部特征。

**after：** 追加final层mean(mu),mean(var),mean(D3)三个广播全层状态。

**详情：** 全层统计只来自网络权重计算；不读取reference的整体偏移。与已做oracle偏置消除诊断不同。

**依赖：** v30.098 L03。

**成本：** O(n) reductions与常数列。

**验收及停止条件：** CV跨网络收益；若只有训练拟合提升拒绝。

**依据：** 项目报告7.5; residual_structure诊断

### v30.106 L11 输出幅度归一化回归

编号：WB-P2-A106。类型：算法或计算消融。准备度：规格待实现及验证。优先级：P2。

**before：** L03在绝对均值误差上拟合。

**after：** 训练目标改成残差/sigma，推理修正乘sigma；特征集保持13列。

**详情：** 改变一个目标尺度机制；ridge标准化及split不变。不得把raw MSE与归一化损失混同。

**依赖：** v30.098 L03。

**成本：** O(n)。

**验收及停止条件：** 最终按原raw/adjusted评分比较，非训练损失。

**依据：** 项目报告7.5

### v30.107 L12 网络等权训练

编号：WB-P2-A107。类型：诊断。准备度：规格待实现及验证。优先级：P2。

**before：** L03将训练网所有神经元等权拼接。

**after：** 每网络归一化为同等总权重，再拟合原残差。

**详情：** 对于当前相同宽度此项数学上与L03等价；先明确等价并作为实现一致性对照，只在未来混宽训练时才是算法消融。

**依赖：** v30.098 L03。

**成本：** 离线；提交推理不变。

**验收及停止条件：** 同宽结果一致；若不一致定位拟合实现，不记作提分。

**依据：** 项目报告网络级验证要求

### v30.108 L13 逐层共享线性修正

编号：WB-P2-A108。类型：算法或计算消融。准备度：规格待实现及验证。优先级：P3。

**before：** L03只训练final输出残差。

**after：** 对全部层训练一个共享13特征系数，但仍只用于final输出。

**详情：** 训练数据增加层深度但模型参数共享；不能把每层当独立网络holdout。每一fold必须排除该网络全部层。

**依赖：** v30.098 L03。

**成本：** 同13n推理；离线数据16倍。

**验收及停止条件：** 最终CV收益与深度误差曲线同时报告；未改善不注入递推。

**依据：** 项目报告7.5

### v30.109 L14 分段线性阈值修正

编号：WB-P2-A109。类型：算法或计算消融。准备度：先推导再实现。优先级：P3。

**before：** L03一个全域线性模型。

**after：** 按固定alpha区间(-∞,-1),[-1,0),[0,1),[1,∞)训练四个线性头，final分段使用。

**详情：** 固定分界，不搜切点；训练稀疏区间缺样本则保留全局头，明确记录。 分段连续性约束及稀疏区间回退规则尚需冻结，不允许执行模型猜测。

**依赖：** v30.098 L03。

**成本：** 最多4×13n点积或掩码向量运算，需真实计费。

**验收及停止条件：** CV逐网络改善且边界附近连续性检查；若阶跃导致异常输出拒绝。

**依据：** 原论文ReLU阈值机制; 项目报告7.5

### v30.110 L15 小网络非线性修正

编号：WB-P2-A110。类型：算法或计算消融。准备度：先推导再实现。优先级：P3。

**before：** L03固定13列线性预测残差。

**after：** 一个固定13→8→1 tanh小网络预测final残差。

**详情：** 先冻结优化器、停止规则、训练seed和标准化，明确fnp可执行tanh路径；不能任低推理模型随意改训练和架构。

**依赖：** v30.098 L03、v30.097 L02。

**成本：** O(13×8n)，非线性也计费。

**验收及停止条件：** 严格网络级CV胜过线性后才进入候选；训练集增益不是结论。

**依据：** 项目报告7.5

### v30.111 L16 单前层修正递推注入

编号：WB-P2-A111。类型：算法或计算消融。准备度：先推导再实现。优先级：P3。

**before：** 验证的修正仅改变final输出，不改变内部状态。

**after：** 仅第8层将已验证修正写入mu，后续原V29递推。

**详情：** 必须先核对均值修正与K2/K3/K4中心化一致性，不能只改mu误称状态正确。新父版本需冻结；不把final头的系数直接无依据套第8层。

**依赖：** v30.108 L13。

**成本：** 特征计算+输出状态转换。

**验收及停止条件：** 非线性中心化合同先通过；网络CV最终adjusted改善且早层误差不爆炸。

**依据：** 上游F47在线修正发散警示

### v30.112 L17 修正器特征删列归因

编号：WB-P2-A112。类型：诊断。准备度：规格待实现及验证。优先级：P2。

**before：** 某修正器整体改善，特征贡献未知。

**after：** 固定已选模型，逐次删除一个特征组再完整重拟合。

**详情：** 每次是一个独立版本；按预注册组阈值、偏度、峰度、历史、全局逐组进行，不穷举所有子集。依赖选定确有CV收益的模型。

**依赖：** v30.098 L03。

**成本：** 离线重拟合；可删去候选中无用推理成本。

**验收及停止条件：** 以网络CV差异归因，保留所有负结果；确认集不用于删列。

**依据：** 项目报告单因素原则

### v30.113 L18 残差条件可预测性上界

编号：WB-P2-A113。类型：诊断。准备度：规格待实现及验证。优先级：P1。

**before：** 只知道偏置oracle解释约0.46% MSE。

**after：** 用开发网络级CV测13特征解释方差，给出达到10倍需约90% MSE减少的差距。

**详情：** 这是可学习性诊断，报告R²和adjusted，不把训练oracle R²当可提交收益；负R²保留。

**依赖：** v30.096 L01、v30.097 L02。

**成本：** 离线。

**验收及停止条件：** CV网络统计、残差尺度、参考误差写清；解释方差低则降优先级，不扩大模型瞎拟合。

**依据：** 项目报告7.4与7.5

### v30.114 L19 独立参考两次采样一致性

编号：WB-P2-A114。类型：诊断。准备度：先推导再实现。优先级：P1。

**before：** 官方十亿样本reference只有单份，条件噪声proxy不能当误差地板。

**after：** 在新生成少量确认网络上制作两份互不重叠输入种子的独立reference。

**详情：** 先制定CPU样本量与置信区间预算，不购买算力；使用至少同级可靠reference或明确弱参考仅作噪声诊断。reference单独留档不进入候选。

**依赖：** 无条目依赖；仍需通用正确性、数值及资源门槛。

**成本：** 可能远超10分钟，仅离线参考。

**验收及停止条件：** 报告两份差、MC标准误及原始种子；阈值低于reference不确定性时不得声明达标。

**依据：** 项目报告8; 原论文Monte Carlo reference

### v30.115 L20 新种子网络最终独立确认

编号：WB-P2-A115。类型：合同或资源验证。准备度：规格待实现及验证。优先级：P1。

**before：** 公共IDs8–15仅是项目内保留集，上游是否使用未知。

**after：** 候选及配置冻结后生成预注册未见种子网络，候选与上一冻结基线成对评分。

**详情：** 种子清单、shape及权重分布先冻结，reference由L19或等价可信流程生成。不得先看候选误差后选网络；不以弱reference评分作为阶段十倍证据。

**依赖：** v30.114 L19。

**成本：** 网络生成+可靠reference，可跨轮。

**验收及停止条件：** 至少10倍阶段收益、零失败、置信不确定性与逐网指标齐全，才能推进stage。

**依据：** 用户阶段规则; 项目报告8

### v30.116 L21 公共确认一次解封

编号：WB-P2-A116。类型：合同或资源验证。准备度：规格待实现及验证。优先级：P1。

**before：** 0–7开发CV选择完成，候选尚未在8–15配对。

**after：** 只对唯一冻结候选执行8–15与原V29同网络成对确认。

**详情：** 记录解封时间和候选hash；失败后该集合不再声称未见，另建确认集。

**依赖：** v30.097 L02。

**成本：** 候选8网，控制已冻结需版本/环境核对。

**验收及停止条件：** 0fail且收益跨网络稳定；公共确认仍不替代L20真正独立验收。

**依据：** results/phase2/v29_shard01_local.json

### v30.117 L22 输出误差与阈值位置分解

编号：WB-P2-A117。类型：诊断。准备度：规格待实现及验证。优先级：P1。

**before：** MSE按网络/层汇总，无法定位哪些神经元贡献。

**after：** 固定alpha分箱报告最后误差均值、方差、MSE与样本量。

**详情：** 使用L01原13特征缓存，只做离线定位，不按真值筛选预测路径。

**依赖：** v30.096 L01。

**成本：** 离线O(N)。

**验收及停止条件：** 完整分箱含空箱，网络级而非神经元伪独立置信区间。

**依据：** 原论文S3.4; 项目报告误差定位闭环

### v30.118 L23 输出误差与偏度分解

编号：WB-P2-A118。类型：诊断。准备度：规格待实现及验证。优先级：P1。

**before：** 未区分高偏度和低偏度神经元误差。

**after：** 按固定标准化D3绝对值分位边界，用训练数据定义边界后测逐网MSE。

**详情：** 边界在训练集固定；不因结果调整阈值，不把相关性当因果。

**依赖：** v30.096 L01。

**成本：** 离线。

**验收及停止条件：** 给出每网结果及阈值冻结记录，用于决定高阶路线优先级。

**依据：** V29 D3; 项目报告7.1

### v30.119 L24 误差沿层积累的增量分解

编号：WB-P2-A119。类型：诊断。准备度：规格待实现及验证。优先级：P1。

**before：** 只有逐层MSE，不区分运输误差和局部新误差。

**after：** 计算e_l−Phi_l⊙(W_l@e_{l−1})的离线一阶残差。

**详情：** e来自reference−prediction仅离线；Phi来自原V29缓存。非线性误差和reference噪声使该量仅为一阶诊断，不是可提交oracle。

**依赖：** v30.096 L01。

**成本：** 离线每层1 n²matvec。

**验收及停止条件：** 报告跨网相关及能量，不把一阶近似当exact decomposition。

**依据：** 项目报告单步与累计残差对比

### v30.120 L25 输出敏感性来源排序诊断

编号：WB-P2-A120。类型：诊断。准备度：先推导再实现。优先级：P1。

**before：** AGE_OLD按年龄统一压缩；不知道每来源最终影响。

**after：** 用合法末端mean Jacobian代理估计来源贡献并排序，不看reference。

**详情：** 先给出D3变化到final mean的线性敏感性和来源state收缩表达；以oracle最终误差仅评估排序质量。禁用真值选source进入候选。

**依赖：** 无条目依赖；仍需通用正确性、数值及资源门槛。

**成本：** 额外反向matvec与来源收缩，需估算。

**验收及停止条件：** 代理排序在独立网络上预测删除代价；无相关性终止。

**依据：** 项目报告7.1输出导向来源压缩

### v30.121 L26 输出敏感性来源预算分配

编号：WB-P2-A121。类型：算法或计算消融。准备度：先推导再实现。优先级：P2。

**before：** 每个老来源相同年龄阈值与固定共享秩。

**after：** 总预算不变，按照已验证L25代理把高影响来源保留dense，其余更早压缩。

**详情：** 先固定排序规则和dense名额，仅一个输出导向策略；不得扫名额。确切state堆栈顺序/重复扣除需强推理推导。

**依赖：** v30.120 L25。

**成本：** 重分配已有预算+代理cost。

**验收及停止条件：** 成对raw/adjusted更优，资源尾部不过限；与相同总dense名额年龄策略对照。

**依据：** 项目报告7.1

### v30.122 L27 来源删除负控制

编号：WB-P2-A122。类型：诊断。准备度：先推导再实现。优先级：P2。

**before：** L25代理有排序，但没有相同成本的随机删除对照。

**after：** 固定输入权重确定性hash选一来源删除，保留其余机制。

**详情：** 需先明确删除整个source三阶对象及所有重复补偿的一致state，不能只清D3或D21。独立版本对比代理相同名额删除。

**依赖：** v30.120 L25。

**成本：** 省掉该来源运输并记录。

**验收及停止条件：** 代理删除平均损伤低于预注册确定性对照；结果只诊断。

**依据：** 来源扰动E03

### v30.123 L28 末端均值控制变量相关性

编号：WB-P2-A123。类型：诊断。准备度：先推导再实现。优先级：P2。

**before：** 未确认廉价样本支路能解释V29残差。

**after：** 在少量开发网络离线测合法低阶代理与真实输出样本的残差相关性。

**详情：** 先固定代理定义、采样seed/样本数及成对V29 residual目标。不可按reference为每测试网拟合混合权重。

**依赖：** v30.096 L01。

**成本：** 离线MC可能较大，算成本折扣上界。

**验收及停止条件：** 达到足以支付成本的解释方差才写候选；低相关停止采样支路。

**依据：** 项目报告7.4

### v30.124 L29 条件高斯最后层积分对照

编号：WB-P2-A124。类型：诊断。准备度：先推导再实现。优先级：P2。

**before：** 最后层均值依赖Hermite边缘展开。

**after：** 对末层pre均值/方差/偏度构造规范化一维积分诊断，和截断展开比较。

**详情：** 需先选合法正密度族并验证其匹配矩，禁止使用可能负的Gram-Charlier密度当真实分布。离线oracle只诊断，不自动写提交。

**依赖：** 无条目依赖；仍需通用正确性、数值及资源门槛。

**成本：** 每神经元一维积分离线；候选成本未定。

**验收及停止条件：** 密度非负/归一化/矩一致，跨网络局部误差有稳定收益。

**依据：** 原论文Hermite展开; 项目报告R3

### v30.125 L30 末层尾概率数值稳定性

编号：WB-P2-A125。类型：合同或资源验证。准备度：规格待实现及验证。优先级：P2。

**before：** 极端alpha路径只由原正态cdf/pdf与float32演算处理。

**after：** 保持原公式，新增离线极端alpha及sigma小值数值对照。

**详情：** 测试alpha=±1,±4,±8,±12与多个正sigma，和高精度正态矩对照；没有失败不更换公式。

**依赖：** 无条目依赖；仍需通用正确性、数值及资源门槛。

**成本：** 小规模诊断。

**验收及停止条件：** 有限值、预期单调性及绝对/相对误差记录；任何修复另开算法版本。

**依据：** 原论文S3.4

### v30.126 L32 跨网络异质收益检查

编号：WB-P2-A126。类型：诊断。准备度：规格待实现及验证。优先级：P1。

**before：** 候选只有网络平均分数改善。

**after：** 在同一固定套件报告胜率、每网ratio、最差退化及paired bootstrap。

**详情：** 不更换套件或candidate；bootstrap按网络重采样，8网CI可能很宽应如实报告。

**依赖：** 无条目依赖；仍需通用正确性、数值及资源门槛。

**成本：** 离线评分。

**验收及停止条件：** 包含所有失败/网络，不择优展示，阶段必须满足独立确认条件。

**依据：** 项目报告8与用户阶段规则

### v30.127 L33 参考dtype贡献诊断

编号：WB-P2-A127。类型：诊断。准备度：规格待实现及验证。优先级：P2。

**before：** 官方targets以原float32使用，误差可能受量化贡献。

**after：** 离线估计原targets float32 ULP尺度与MSE阈值关系，保留原targets不重评分。

**详情：** 不能把float32→float64 cast当reference精度提高，也不能把自行重算reference替换后宣称同协议改进。

**依赖：** 无条目依赖；仍需通用正确性、数值及资源门槛。

**成本：** 离线O(N)。

**验收及停止条件：** 只报告量化尺度proxy及适用限制，与采样噪声分开。

**依据：** 固定原始reference规则

### v30.128 L34 标准化计算的候选合同

编号：WB-P2-A128。类型：合同或资源验证。准备度：规格待实现及验证。优先级：P1。

**before：** 拟合端与fnp推理端标准化可能使用不同精度或特征顺序。

**after：** 冻结模型文件的feature_order、mean、scale、coef哈希并在小样本对照fnp预测。

**详情：** 禁止运行时重新按测试网络重估训练标准化，除非另立合法机制版本。全部参数与代码在provenance记录。

**依赖：** v30.098 L03。

**成本：** 小规模数学合同。

**验收及停止条件：** 同输入回归输出在float32容差内，模型不含网络ID/targets查表。

**依据：** 项目报告7.5及提交合同

### v30.129 L35 多次选择偏差账本

编号：WB-P2-A129。类型：诊断。准备度：规格待实现及验证。优先级：P0。

**before：** 长消融列表容易反复查看同一确认集而失去独立性。

**after：** 每次拟合/调参/查看reference都记录selection ledger，确认集首次查看后降为开发集。

**详情：** 记录candidate hash、训练/验证ID、选择次数、目标阈值与解封状态；保存负结果，禁止仅留最优。

**依赖：** v30.097 L02。

**成本：** metadata，无算法变化。

**验收及停止条件：** ledger与queue及结果路径一一对应；仍未见最终集单独明确。

**依据：** 用户参考噪声与多次选择偏差要求

## 可执行资料

机器队列：`ablation_queue.json`。执行提示词：`LOW_REASONING_HANDOFF.md`。原始分组规格：`backlog_compute.json`、`backlog_highorder.json`、`backlog_official.json`、`backlog_learning.json`。重复的 L31 漂移控制已合并入 C25。
