# 按 is_promising 排序的执行优先级

134项＝原129项＋5项论文方法。is_promising 是研究判断，不是已验证收益；false不表示数学否决。原status/result/evidence均保留。先满足依赖、公式和成本门槛。

| 顺序 | is_promising | 执行优先级 | key | 版本 | 名称 | 状态 | 判断依据 |
|---:|---|---|---|---|---|---|---|
| 1 | true | P0 | M01 | paper.M01.r01 | 论文—官方代码—本地映射 | partial_mapping | 结构性机制或论文完整对照；尚未证明收益 |
| 2 | true | P0 | M03 | paper.M03.r01 | Wu2026阶数及表示方法对照 | partial_diagnostic | 结构性机制或论文完整对照；尚未证明收益 |
| 3 | true | P0 | O03 | v30.065 | K211完整来源抵消一致性 | partial_diagnostic | 结构性机制或论文完整对照；尚未证明收益 |
| 4 | true | P0 | O05 | v30.067 | 完整K31/K211生产核的离线表示上限 | partial_diagnostic | 结构性机制或论文完整对照；尚未证明收益 |
| 5 | true | P0 | O06 | v30.068 | 完整生产核单层替换诊断 | partial_diagnostic | 结构性机制或论文完整对照；尚未证明收益 |
| 6 | true | P0 | H15 | v30.045 | 历史K3来源生成K211 capped低rank核心 | partial_derivation | 结构性机制或论文完整对照；尚未证明收益 |
| 7 | true | P0 | H16 | v30.046 | actual K31低rank核心残差 | partial_derivation | 结构性机制或论文完整对照；尚未证明收益 |
| 8 | true | P0 | H21 | v30.051 | 恢复非对角metric的K4到K3 feed | partial_derivation | 结构性机制或论文完整对照；尚未证明收益 |
| 9 | true | P0 | L19 | v30.114 | 独立参考两次采样一致性 | partial_derivation | 结构性机制或论文完整对照；尚未证明收益 |
| 10 | true | P0 | O29 | v30.091 | 高阶校正信号与reference不确定性的分离 | partial_diagnostic | 结构性机制或论文完整对照；尚未证明收益 |
| 11 | true | P0 | L20 | v30.115 | 新种子网络最终独立确认 | partial_diagnostic | 结构性机制或论文完整对照；尚未证明收益 |
| 12 | true | P1 | M04 | paper.M04.r01 | 解析代理和残差采样控制变量 | partial_negative | 结构性机制或论文完整对照；尚未证明收益 |
| 13 | true | P1 | M05 | paper.M05.r01 | Monchot2023条件或混合传播 | partial_diagnostic | 结构性机制或论文完整对照；尚未证明收益 |
| 14 | true | P1 | M02 | paper.M02.r01 | Wright2024完整精确高斯协方差 | partial_diagnostic | 结构性机制或论文完整对照；尚未证明收益 |
| 15 | true | P1 | H28 | v30.058 | 再生core对角和offdiag联合对象闭合 | partial_derivation | 结构性机制或论文完整对照；尚未证明收益 |
| 16 | true | P1 | H14 | v30.044 | B2两步输出导向收缩 | partial_derivation | 结构性机制或论文完整对照；尚未证明收益 |
| 17 | true | P1 | H23 | v30.053 | old-source basis按K21收缩能源选择 | partial_derivation | 结构性机制或论文完整对照；尚未证明收益 |
| 18 | true | P1 | L25 | v30.120 | 输出敏感性来源排序诊断 | partial_derivation | 结构性机制或论文完整对照；尚未证明收益 |
| 19 | true | P1 | L26 | v30.121 | 输出敏感性来源预算分配 | partial_derivation | 结构性机制或论文完整对照；尚未证明收益 |
| 20 | true | P1 | H25 | v30.055 | Rres新增解析可分leading项 | partial_derivation | 结构性机制或论文完整对照；尚未证明收益 |
| 21 | true | P1 | H26 | v30.056 | Rres物理机制分组basis | partial_derivation | 结构性机制或论文完整对照；尚未证明收益 |
| 22 | true | P1 | H29 | v30.059 | K111乘covariance边新来源 | partial_derivation | 结构性机制或论文完整对照；尚未证明收益 |
| 23 | true | P1 | H30 | v30.060 | K111乘covariance边一层收缩 | partial_derivation | 结构性机制或论文完整对照；尚未证明收益 |
| 24 | true | P1 | H32 | v30.062 | 高阶newborn直接CP因素生成 | partial_derivation | 结构性机制或论文完整对照；尚未证明收益 |
| 25 | false | P2 | H04 | v30.034 | 精确Gaussian pair核替换 | partial_derivation | 推导储备；当前证据不足，排在结构性任务之后 |
| 26 | false | P2 | H05 | v30.035 | Gaussian pair尾项有理重求和 | partial_derivation | 推导储备；当前证据不足，排在结构性任务之后 |
| 27 | false | P2 | H07 | v30.037 | WK22到B1出生来源恢复 | partial_derivation | 推导储备；当前证据不足，排在结构性任务之后 |
| 28 | false | P2 | H08 | v30.038 | B1的WK12对称反馈恢复 | partial_derivation | 推导储备；当前证据不足，排在结构性任务之后 |
| 29 | false | P2 | H09 | v30.039 | B2 leading D21乘C来源恢复 | partial_derivation | 推导储备；当前证据不足，排在结构性任务之后 |
| 30 | false | P2 | H10 | v30.040 | B2 WK22乘C来源恢复 | partial_derivation | 推导储备；当前证据不足，排在结构性任务之后 |
| 31 | false | P2 | H11 | v30.041 | B2 D21平方来源恢复 | partial_derivation | 推导储备；当前证据不足，排在结构性任务之后 |
| 32 | false | P2 | H12 | v30.042 | B2 WK22乘D21来源恢复 | partial_derivation | 推导储备；当前证据不足，排在结构性任务之后 |
| 33 | false | P2 | H24 | v30.054 | old-source basis按下一层D3敏感方向选择 | partial_derivation | 推导储备；当前证据不足，排在结构性任务之后 |
| 34 | false | P2 | O04 | v30.066 | 旧lean与当前官方generator语义归因 | partial_diagnostic | 推导储备；当前证据不足，排在结构性任务之后 |
| 35 | false | P2 | O07 | v30.069 | 完整核对角修正非零性与候选 | partial_diagnostic | 推导储备；当前证据不足，排在结构性任务之后 |
| 36 | false | P2 | O08 | v30.070 | 完整production核仅USE替换 | partial_diagnostic | 推导储备；当前证据不足，排在结构性任务之后 |
| 37 | false | P2 | O09 | v30.071 | 完整production核仅FEED替换 | partial_diagnostic | 推导储备；当前证据不足，排在结构性任务之后 |
| 38 | false | P2 | O10 | v30.072 | 完整核剩余误差的state模态诊断 | partial_diagnostic | 推导储备；当前证据不足，排在结构性任务之后 |
| 39 | false | P2 | O13 | v30.075 | K111乘edge covariance主模态代理 | partial_derivation | 推导储备；当前证据不足，排在结构性任务之后 |
| 40 | false | P2 | O14 | v30.076 | K111乘edge代理误差门槛 | partial_derivation | 推导储备；当前证据不足，排在结构性任务之后 |
| 41 | false | P2 | O15 | v30.077 | doubled-edge早期来源贡献门槛 | partial_diagnostic | 推导储备；当前证据不足，排在结构性任务之后 |
| 42 | false | P2 | O16 | v30.078 | covariance triangle exact多层small诊断 | partial_diagnostic | 推导储备；当前证据不足，排在结构性任务之后 |
| 43 | false | P2 | O18 | v30.080 | 现有birth薄反馈乘积预算审计 | partial_diagnostic | 推导储备；当前证据不足，排在结构性任务之后 |
| 44 | false | P2 | O19 | v30.081 | K5边缘生产闭合 | partial_derivation | 推导储备；当前证据不足，排在结构性任务之后 |
| 45 | false | P2 | O20 | v30.082 | K5独立边缘readout | partial_derivation | 推导储备；当前证据不足，排在结构性任务之后 |
| 46 | false | P2 | O21 | v30.083 | K6纯径向投影生产公式 | partial_derivation | 推导储备；当前证据不足，排在结构性任务之后 |
| 47 | false | P2 | O22 | v30.084 | K6径向状态合法线性运输 | partial_derivation | 推导储备；当前证据不足，排在结构性任务之后 |
| 48 | false | P2 | O23 | v30.085 | K6独立边缘readout | partial_derivation | 推导储备；当前证据不足，排在结构性任务之后 |
| 49 | false | P2 | O28 | v30.090 | 新seed网络的diagram规模law核验 | partial_diagnostic | 推导储备；当前证据不足，排在结构性任务之后 |
| 50 | false | P2 | O30 | v30.092 | 新增图表array操作计量路径合同 | partial_diagnostic | 推导储备；当前证据不足，排在结构性任务之后 |
| 51 | false | P2 | L01 | v30.096 | 冻结原轨迹特征缓存 | partial_diagnostic | 推导储备；当前证据不足，排在结构性任务之后 |
| 52 | false | P2 | L02 | v30.097 | 冻结网络级训练选择流程 | partial_diagnostic | 推导储备；当前证据不足，排在结构性任务之后 |
| 53 | false | P2 | L16 | v30.111 | 单前层修正递推注入 | partial_derivation | 推导储备；当前证据不足，排在结构性任务之后 |
| 54 | false | P2 | L17 | v30.112 | 修正器特征删列归因 | partial_diagnostic | 推导储备；当前证据不足，排在结构性任务之后 |
| 55 | false | P2 | L27 | v30.122 | 来源删除负控制 | partial_derivation | 推导储备；当前证据不足，排在结构性任务之后 |
| 56 | false | P2 | L28 | v30.123 | 末端均值控制变量相关性 | partial_derivation | 推导储备；当前证据不足，排在结构性任务之后 |
| 57 | false | P3 | C01 | v30.001 | 递归限制C_pre上三角输出 | completed_positive | 已有负结果或已完成；保留证据，需新机制才重开 |
| 58 | false | P3 | C02 | v30.002 | Strassen–Winograd共享加法链 | completed_negative | 已有负结果或已完成；保留证据，需新机制才重开 |
| 59 | false | P3 | C03 | v30.003 | 将Z与Zf薄腿合并运输 | completed_negative | 已有负结果或已完成；保留证据，需新机制才重开 |
| 60 | false | P3 | C04 | v30.004 | 将两条D21反馈收缩合并薄列 | completed_negative | 已有负结果或已完成；保留证据，需新机制才重开 |
| 61 | false | P3 | C05 | v30.005 | PPL与Z的末次薄腿收缩写成扁平GEMM | completed_positive | 已有负结果或已完成；保留证据，需新机制才重开 |
| 62 | false | P3 | C06 | v30.006 | 最后一层M残差的D3只做对角收缩 | completed_positive | 已有负结果或已完成；保留证据，需新机制才重开 |
| 63 | false | P3 | C07 | v30.007 | 省掉最终层四阶矩阵切片 | completed_positive | 已有负结果或已完成；保留证据，需新机制才重开 |
| 64 | false | P3 | C08 | v30.008 | 最终层仅运行PK1均值输出 | completed_positive | 已有负结果或已完成；保留证据，需新机制才重开 |
| 65 | false | P3 | C09 | v30.009 | 稀疏wick多项式代替dense表乘 | completed_positive | 已有负结果或已完成；保留证据，需新机制才重开 |
| 66 | false | P3 | C10 | v30.010 | 稀疏Wick selector改indexed选取 | completed_positive | 已有负结果或已完成；保留证据，需新机制才重开 |
| 67 | false | P3 | C11 | v30.011 | 非线性term按输出组分开收缩 | completed_positive | 已有负结果或已完成；保留证据，需新机制才重开 |
| 68 | false | P3 | C12 | v30.012 | 去重AB矩阵乘积缓存 | completed_negative | 已有负结果或已完成；保留证据，需新机制才重开 |
| 69 | false | P3 | C13 | v30.013 | PK11和PK22对称输出block化 | completed_negative | 已有负结果或已完成；保留证据，需新机制才重开 |
| 70 | false | P3 | C14 | v30.014 | 避免二次covariance对称平均 | completed_positive | 已有负结果或已完成；保留证据，需新机制才重开 |
| 71 | false | P3 | C15 | v30.015 | D21绝对值行和替代ones GEMV | completed_positive | 已有负结果或已完成；保留证据，需新机制才重开 |
| 72 | false | P3 | C16 | v30.016 | 小核心Sj只计算上三角 | completed_positive | 已有负结果或已完成；保留证据，需新机制才重开 |
| 73 | false | P3 | C17 | v30.017 | Sg双边旋转的末次乘法对称化 | completed_positive | 已有负结果或已完成；保留证据，需新机制才重开 |
| 74 | false | P3 | C18 | v30.018 | S2子空间投影的末次乘法三角化 | completed_negative | 已有负结果或已完成；保留证据，需新机制才重开 |
| 75 | false | P3 | C19 | v30.019 | joiner两路Gram operator共享布局 | completed_positive | 已有负结果或已完成；保留证据，需新机制才重开 |
| 76 | false | P3 | C20 | v30.020 | final D3 feedback的线性两项薄收缩 | completed_negative | 已有负结果或已完成；保留证据，需新机制才重开 |
| 77 | false | P3 | C21 | v30.021 | nonlinearity中广播d3/g4向量不materialize | completed_negative | 已有负结果或已完成；保留证据，需新机制才重开 |
| 78 | false | P3 | C22 | v30.022 | PK1 duplicate一维因子缓存去重 | completed_negative | 已有负结果或已完成；保留证据，需新机制才重开 |
| 79 | false | P3 | C23 | v30.023 | 数值对称误差审计 | completed_diagnostic | 已有负结果或已完成；保留证据，需新机制才重开 |
| 80 | false | P3 | C24 | v30.024 | 逐组件实际计费归因 | completed_diagnostic | 已有负结果或已完成；保留证据，需新机制才重开 |
| 81 | false | P3 | C25 | v30.025 | 浮点加法顺序与误差归因 | completed_diagnostic | 已有负结果或已完成；保留证据，需新机制才重开 |
| 82 | false | P3 | C26 | v30.026 | 持久pool最大shape资源审计 | completed_diagnostic | 已有负结果或已完成；保留证据，需新机制才重开 |
| 83 | false | P3 | C27 | v30.027 | shape变更时释放无用旧pool | completed_negative | 已有负结果或已完成；保留证据，需新机制才重开 |
| 84 | false | P3 | C28 | v30.028 | 分区清理不再需要的scratch常驻 | completed_diagnostic | 已有负结果或已完成；保留证据，需新机制才重开 |
| 85 | false | P3 | C29 | v30.029 | 三角C_pre阶段批量copy调度 | partial_negative | 已有负结果或已完成；保留证据，需新机制才重开 |
| 86 | false | P3 | C30 | v30.030 | 不同backend上合法matmul计费与resource对照 | completed_diagnostic | 已有负结果或已完成；保留证据，需新机制才重开 |
| 87 | false | P3 | H01 | v30.031 | 边际D3平方产品补全 | completed_positive | 已有负结果或已完成；保留证据，需新机制才重开 |
| 88 | false | P3 | H02 | v30.032 | 边际D3乘g4产品补全 | completed_positive | 已有负结果或已完成；保留证据，需新机制才重开 |
| 89 | false | P3 | H03 | v30.033 | 单轴截断集合一致性诊断 | completed_diagnostic | 已有负结果或已完成；保留证据，需新机制才重开 |
| 90 | false | P3 | H06 | v30.036 | 未剪枝SIMPLE生产表恢复 | completed_negative | 已有负结果或已完成；保留证据，需新机制才重开 |
| 91 | false | P3 | H13 | v30.043 | B2保留一次线性运输再shadow | completed_negative | 已有负结果或已完成；保留证据，需新机制才重开 |
| 92 | false | P3 | H17 | v30.047 | Gaussian K211完整二阶矩阵核心 | completed_negative | 已有负结果或已完成；保留证据，需新机制才重开 |
| 93 | false | P3 | H18 | v30.048 | K4边际真实metric系数 | completed_positive | 已有负结果或已完成；保留证据，需新机制才重开 |
| 94 | false | P3 | H19 | v30.049 | WK431真实metric混合项 | completed_negative | 已有负结果或已完成；保留证据，需新机制才重开 |
| 95 | false | P3 | H20 | v30.050 | WK22真实metric交叉项 | completed_negative | 已有负结果或已完成；保留证据，需新机制才重开 |
| 96 | false | P3 | H22 | v30.052 | D21反馈basis输出敏感sketch | completed_positive | 已有负结果或已完成；保留证据，需新机制才重开 |
| 97 | false | P3 | H27 | v30.057 | K3 newborn重复内容一致性oracle | completed_diagnostic | 已有负结果或已完成；保留证据，需新机制才重开 |
| 98 | false | P3 | H31 | v30.061 | exact covariance triangle无偏收缩 | completed_negative | 已有负结果或已完成；保留证据，需新机制才重开 |
| 99 | false | P3 | O01 | v30.063 | 官方K31截断生产逐项一致性 | completed_diagnostic | 已有负结果或已完成；保留证据，需新机制才重开 |
| 100 | false | P3 | O02 | v30.064 | 官方capped K211逐项生产一致性 | completed_diagnostic | 已有负结果或已完成；保留证据，需新机制才重开 |
| 101 | false | P3 | O11 | v30.073 | K111乘covariance edge完整公式核验 | completed_diagnostic | 已有负结果或已完成；保留证据，需新机制才重开 |
| 102 | false | P3 | O12 | v30.074 | K111乘edge的exact final D3成本证明 | completed_diagnostic | 已有负结果或已完成；保留证据，需新机制才重开 |
| 103 | false | P3 | O17 | v30.079 | 当前官方WK22 path与旧lean等价审计 | completed_diagnostic | 已有负结果或已完成；保留证据，需新机制才重开 |
| 104 | false | P3 | O24 | v30.086 | 高阶边缘同阶组合inventory | completed_diagnostic | 已有负结果或已完成；保留证据，需新机制才重开 |
| 105 | false | P3 | O25 | v30.087 | degree5/6 ordinary与power累积量独立转换 | completed_diagnostic | 已有负结果或已完成；保留证据，需新机制才重开 |
| 106 | false | P3 | O26 | v30.088 | 图表收益的上游调参分割审计 | completed_diagnostic | 已有负结果或已完成；保留证据，需新机制才重开 |
| 107 | false | P3 | O27 | v30.089 | 封存新分片的diagslice状态分布偏移 | completed_diagnostic | 已有负结果或已完成；保留证据，需新机制才重开 |
| 108 | false | P3 | O31 | v30.093 | 完整K31生产对称化与轴方向合同 | completed_diagnostic | 已有负结果或已完成；保留证据，需新机制才重开 |
| 109 | false | P3 | O32 | v30.094 | capped211 diagonal mask因式表示合同 | completed_diagnostic | 已有负结果或已完成；保留证据，需新机制才重开 |
| 110 | false | P3 | O33 | v30.095 | complete harmonic core actualW transport对照 | completed_diagnostic | 已有负结果或已完成；保留证据，需新机制才重开 |
| 111 | false | P3 | L03 | v30.098 | 最后层13特征残差线性修正 | completed_negative | 已有负结果或已完成；保留证据，需新机制才重开 |
| 112 | false | P3 | L04 | v30.099 | 删除常数特征的修正对照 | partial_negative | 已有负结果或已完成；保留证据，需新机制才重开 |
| 113 | false | P3 | L05 | v30.100 | 阈值形状特征增量 | partial_negative | 已有负结果或已完成；保留证据，需新机制才重开 |
| 114 | false | P3 | L06 | v30.101 | 标准化偏度特征增量 | partial_negative | 已有负结果或已完成；保留证据，需新机制才重开 |
| 115 | false | P3 | L07 | v30.102 | 标准化峰度特征增量 | partial_negative | 已有负结果或已完成；保留证据，需新机制才重开 |
| 116 | false | P3 | L08 | v30.103 | 最后线性均值传播轨迹特征 | partial_negative | 已有负结果或已完成；保留证据，需新机制才重开 |
| 117 | false | P3 | L09 | v30.104 | 前层方差轨迹特征 | partial_negative | 已有负结果或已完成；保留证据，需新机制才重开 |
| 118 | false | P3 | L10 | v30.105 | 跨神经元均值残差特征 | partial_negative | 已有负结果或已完成；保留证据，需新机制才重开 |
| 119 | false | P3 | L11 | v30.106 | 输出幅度归一化回归 | partial_negative | 已有负结果或已完成；保留证据，需新机制才重开 |
| 120 | false | P3 | L12 | v30.107 | 网络等权训练 | partial_negative | 已有负结果或已完成；保留证据，需新机制才重开 |
| 121 | false | P3 | L13 | v30.108 | 逐层共享线性修正 | partial_negative | 已有负结果或已完成；保留证据，需新机制才重开 |
| 122 | false | P3 | L14 | v30.109 | 分段线性阈值修正 | partial_negative | 已有负结果或已完成；保留证据，需新机制才重开 |
| 123 | false | P3 | L15 | v30.110 | 小网络非线性修正 | partial_negative | 已有负结果或已完成；保留证据，需新机制才重开 |
| 124 | false | P3 | L18 | v30.113 | 残差条件可预测性上界 | completed_diagnostic | 已有负结果或已完成；保留证据，需新机制才重开 |
| 125 | false | P3 | L21 | v30.116 | 公共确认一次解封 | completed_diagnostic | 已有负结果或已完成；保留证据，需新机制才重开 |
| 126 | false | P3 | L22 | v30.117 | 输出误差与阈值位置分解 | completed_diagnostic | 已有负结果或已完成；保留证据，需新机制才重开 |
| 127 | false | P3 | L23 | v30.118 | 输出误差与偏度分解 | completed_diagnostic | 已有负结果或已完成；保留证据，需新机制才重开 |
| 128 | false | P3 | L24 | v30.119 | 误差沿层积累的增量分解 | completed_diagnostic | 已有负结果或已完成；保留证据，需新机制才重开 |
| 129 | false | P3 | L29 | v30.124 | 条件高斯最后层积分对照 | partial_negative | 已有负结果或已完成；保留证据，需新机制才重开 |
| 130 | false | P3 | L30 | v30.125 | 末层尾概率数值稳定性 | completed_diagnostic | 已有负结果或已完成；保留证据，需新机制才重开 |
| 131 | false | P3 | L32 | v30.126 | 跨网络异质收益检查 | completed_diagnostic | 已有负结果或已完成；保留证据，需新机制才重开 |
| 132 | false | P3 | L33 | v30.127 | 参考dtype贡献诊断 | completed_diagnostic | 已有负结果或已完成；保留证据，需新机制才重开 |
| 133 | false | P3 | L34 | v30.128 | 标准化计算的候选合同 | completed_diagnostic | 已有负结果或已完成；保留证据，需新机制才重开 |
| 134 | false | P3 | L35 | v30.129 | 多次选择偏差账本 | completed_diagnostic | 已有负结果或已完成；保留证据，需新机制才重开 |
