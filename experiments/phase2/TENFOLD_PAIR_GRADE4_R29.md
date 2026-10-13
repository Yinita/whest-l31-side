# 一致 pair/marginal cost4 闭合：r26–r29

2026-10-13。主要变化仅为在冻结V37bH18的mode1层补齐一整级cost4边缘与二变量pK，并以同一mean重中心化。目标依旧Mini8 adjusted <5.779e-10；本文件没有达标声明，strict候选成绩由根代理收集。

| 版本 | 名称 | before | after | 实测证据 |
|---|---|---|---|---|
| r26 | 官方cost4二变量/边缘生成与独立积分 | 现pair最高cost3；边缘p1有D3²/D3K4，p2..4均缺cost4 | pinned kmax4,dmax4,augmentTrue全部cost4；每pair28labelled/16iso项，每边缘3项 | 3组固定mu/sigma×4pair通道，独立160² Hermite密度积分并减去分级disconnect均值乘积，maxabs4.0e-19 |
| r28 | 同H18实际li14state幅度门 | 自parent完整pair/边缘state | 同矩加一阶grade4后在float64精确重中心化 | ΔK2/K3/K4 RMS5.002e-6/8.964e-6/2.602e-5，ΔK11/21/22 RMS8.900e-6/1.069e-5/1.392e-5（相对.081%/.556%/4.391%），nextvarΔRMS2.516e-5；5.987G FLOPs含真实下层W收缩 |
| r29 | 全链合法生产候选 | 冻结H18 c3dd…ffda2 | mode1全部层新增同cost4 pK，p1去掉已经有的D3²/D3K4避免重复；pair/边缘一起float64中心化后回float32；最后层只均值 | helper n4/8/32 parity与finaltrim误差zero；wholecall n32/64 depth4 finite float32；strict1024待根代理 |

r26与H01/H02/r15不同：后者只有边缘产品补全；与H04/pair12不同：后者只有Gaussian rho级数；与H17不同：后者Gaussian三指标K211core。r29完整二变量含D3·D21、D21²、C²D21、C·K31、K4·D21及K22²等固定图，p1新增只有K4²/1152。仅高一级二变量闭合，未补三/四distinctsupport生产，未解决父K4proxy/source压缩，也未更改Gaussian旧历史增益。mode0保留原路，所以没有Gaussian firstbirthC4改动；这项机制定义为mode1闭合，不能称完整k4算法。

所有候选算术仅hardcode固定tuple与允许flopscope算子；没有导入officialrepo、offlinehelper、MC、truth或reference。实际capture source就是H18，以当前层内部排列保存fullpair；14source仅前32内部rows但保留1024birthcolumns供K4组件重建，同call自己的prediction绑定，不混旧V29cache。capture17npz共48.12MB保留本地不Git。

原native capture55.45秒/480,763,746,423 FLOPs仅诊断instrumented资源，不能当strict验收。独立gate同样不是score，delta幅度和符号不能证明改善。第一次r27 gate字段KeyError与p1现有项去重疏漏未报告结果，失败script/log保留；r28修正后仅实测target-free幅度。wholecandidate两次小shape集成失败（非regen wk431不可用；flopscope immutable assignment）均保存当时source/script/log，最终修为非regen zeroK31和合法stack，当前合同r31通过。父legacy lint37项继承，新增helper/脚本ruff通过范围另列，不能冒称候选整体ruffclean。

冻结candidate SHA73099422205a37b7ce0f41fcb599292a04c30dd0e3d00509ee706ad7e3f9034d。生产来源、MITnotice详见estimators/tenfold_pair_grade4_consistent_r29/SOURCE.md；新报告必须唯一version，strict结果不能被component证据替代。


## strict1 完成与归因（根代理成对实测）

原network0/reference严格RPC：parent raw2.2281179568e-8、adjusted4.8714160494e-9；r29 raw2.2174980074396444e-8（−0.47663%）、adjusted5.025853859795121e-9（+3.17029%），零失败。真实raw小改善被新增计费抵消，十倍机制未获支持；不扩Mini8、不上传、不追系数/rank。保留负adjusted结果。实际wholecall前后center独立直接代数合同r32已通过：n32/64两个mode1中间层K1..4/K11/K21/K22 float32与float64公式cast逐位一致、mode0 firstprediction与parent逐位一致；r32 executed脚本lint变量命名问题保留，r33仅修lint重跑，候选未修改。

下一路线是准确保存早Gaussian出生的K22及其四阶历史，解决pureD4候选遗漏K22出生造成的早层偏差。与K4代理分工追踪真正四阶矩的表示/输运成本；不能将r29小raw收益当已解决K4闭合或十倍路线。
