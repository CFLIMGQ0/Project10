# PC-CMKA-DDKAC 仓库审计与实施计划

本文是修改代码前的 Phase 0 审计记录。审计对象是当前工作区，而不是原始
MRePath 仓库。新方法暂定配置名为 `pc_cmka_ddkac`，不会覆盖已有的
`original`、`dd_kac`、MLP、GCN 或其他 KAN 变体。

## 1. 当前基因组数据流

1. `main.py` 创建 `SurvivalDatasetFactory`，随后按公开 split CSV 调用
   `return_splits` 获取训练集和验证集。
2. `SurvivalDatasetFactory._setup_omics_data` 读取当前队列的
   `rna_clean.csv`。当前实际只有 RNA，没有 CNV/SNV。
3. `_setup_mcat` 读取 `datasets_csv/metadata/signatures.csv`，将每列基因与
   RNA 列取交集并排序，得到六个功能组。COREAD 与 STAD 当前 RNA 都是
   4,999 维，交集后的六组维数相同：53、173、216、252、387、200。
4. `_get_split_from_df` 先处理训练折并拟合 RNA scaler，再把同一个 scaler
   用到验证折。`SurvivalDataset.__getitem__` 用 `case_id` 取唯一 RNA 行，
   按六组基因列表切出六个张量。
5. `_collate_MCAT` 将六组张量分别拼成批次；`_unpack_data` 将它们传到 GPU；
   `_process_data_and_forward` 以 `x_omic1` 至 `x_omic6` 传给 `MRePath`。
6. `MRePath.forward` 调用已有 SNN 或可选 genomic encoder，接口统一为
   `[batch, 6, d]`；随后可选 `GeneGraphAggregator`、模态加权、IFA、两模态
   token 均值池化和离散时间生存头。

六组名称是：Tumor Suppressor Genes、Oncogenes、Protein Kinases、Cell
Differentiation Markers、Transcription Factors、Cytokines and Growth Factors。
功能组之间允许基因重叠；组内顺序是按基因名排序，不被当作空间邻接。

## 2. 当前 DD-KAC 的真实实现

当前 `DDKACEncoder` 对六组分别实例化一个 `DDKACPathway`，每组输入
`[batch, n_g]`，输出一个 d 维 token，最终堆叠为 `[batch, 6, d]`。

- 值域分支：8 个固定中心的 Gaussian/RBF 基函数，加一个可学习线性倍率，
  再通过 `Linear(n_g, d)` 投影。
- 图来源：每折进入 `_train_val` 后，`_build_fold_gene_graphs` 只使用该折
  训练 RNA；每组独立计算基因间绝对 Pearson 相关，每个节点保留 top-8，
  对称化并加单位对角。图在该折内固定，不随患者改变。
- 拉普拉斯：构造 `A` 的对称归一化，`L=I-D^-1/2 A D^-1/2`；实际用于
  Chebyshev 的 `scaled_laplacian=L-I=-D^-1/2 A D^-1/2`，等价于假设
  `lambda_max=2` 的缩放。
- 结构分支：先对每位患者的组内表达中心化，再计算 0 至 `order` 阶
  Chebyshev 递推，默认 `order=2`。
- 患者频率路由：从 `[均值、标准差、近零比例、图能量、高负荷比例]`
  预测 `order+1` 个 softmax 权重。因此路由按患者变化，但图本身不变。
- 双分支门控：同一组统计量预测一个 sigmoid gate，在值域投影与结构投影
  之间插值，并叠加输入的线性残差，最后 LayerNorm。
- 正则：六组值域/结构输出的 `1-cosine_similarity` 均值乘 `1e-4`，通过
  `model.auxiliary_loss` 加到生存损失。

所以当前 DD-KAC 不是 PC-CMKA-DDKAC：它没有固定先验边支撑、参考度
`D0`、患者矩目标、低秩边字典、可微逆求解器、信赖域、不确定性反向配对、
Krylov 安全约束、共享路由双视图、无负样本 SSL 或可辨识性正则。

## 3. 病理与融合路径

病理侧从缓存图文件取 ResNet50 patch 特征及拓扑/特征超边，经过
`pathomics_fc` 和选定的 SHGNN/HGNN/GCN/GAT/MLP。动态模态加权在高阶
病理传播前由编码后的病理 token 与六个基因 token 计算；随后两个模态乘
可用性和权重，进入 `AlignFusion`。默认 IFA 顺序是基因自注意力、病理指导
基因、基因指导病理。两侧分别均值池化、拼接，再送入 MLP 和 4 区间 hazard
logit 头。

PC-CMKA-DDKAC 的第一阶段只替换基因 encoder，保持上述病理、加权、IFA、
生存头和 split 成员不变。

## 4. 数据泄漏、维度、数值和复现风险

| 项目 | 结论 | 风险/处置 |
|---|---|---|
| 基因图跨折泄漏 | 当前未发现 | 图由 `train_split` 构建；新先验、字典初始化、谱上界和统计量也必须在每折模型创建前只用训练折。 |
| RNA scaler 泄漏 | 当前未发现直接验证折拟合 | 但实现将整个训练矩阵展平成单列，只拟合一个全局 MinMax，并把原始零值强制还原；不是逐基因 scaler，需显式记录。 |
| 生存分箱泄漏 | **存在** | 数据工厂在拆折前，用全队列未删失病例计算 `qcut` 边界。严格实验需增加“每折仅训练病例拟合分箱”路径。 |
| 邻接维度 | 当前六组匹配 | 图与 `omic_names` 同序构建，再与 `omic_sizes` zip；缺少显式 shape 断言，新模块会补。 |
| 患者批次污染 | DD-KAC 内未发现 | 路由统计逐患者计算。MRePath 实际只支持 batch size 1；`_collate_MCAT` 对 batch>1 只保留第一个图，因此不得把增大 batch 当作有效加速。 |
| 图归一化 | 与新方案不一致 | 当前使用患者无关的归一化邻接拉普拉斯；新方案必须使用 oriented incidence、先验权重和固定参考度 `D0`。 |
| 随机复现 | 部分保证 | 已设置 Python/NumPy/Torch seed、关闭 cuDNN benchmark；未开启 `torch.use_deterministic_algorithms`，部分 scatter/CUDA 算子可能非确定。一次进程连续跑多折还会让后折 RNG 依赖前折；当前调度器逐折新进程可规避该点。 |
| 稠密图 | 当前可控但不满足新设计 | 当前每组相关矩阵和拉普拉斯均为稠密；最大 387 节点尚可。新方法禁止扩展成患者数乘全边数的大稠密算子，使用 edge list/MVP。 |
| CSV 顺序邻接 | 未发现 | 图来自训练折基因相关，不来自 CSV 相邻行。新方法也只接受显式 edge support。 |
| 验证随机增强 | 默认关闭 | 验证 bag 默认取前 4096 patches；新图增强必须额外受 `model.training` 控制。 |

## 5. 方案歧义与保守解释

以下内容在 Word/任务文本中不足以唯一决定实现，不能静默猜测：

1. 六组节点数和边数不同，单个矩阵 `P in R^(|E0| x K)` 无法直接跨六组
   共享。第一版采用**每功能组一个 P、同一超参数和同一实现**；从全模型看
   等价于块对角字典。不会在不同基因身份的边之间共享数值参数。
2. 文本同时写了对 `g` 求和和单个患者系数 `a_p`。第一版采用每组独立
   `a_(p,g)`，因为边支撑不同，而且增强要求“根据患者和功能组生成”。六组
   的损失再求和/均值。
3. `rho` 有正线性代价且只作为 `||Pa||_inf` 的上界时，给定 `a` 的最优解
   必然是 `rho=min(||Pa||_inf,rho_max)`。第一版因此把 rho 作为求解状态的
   确定性最小信赖域量，并在每次展开迭代投影，而不是另设 MLP。
4. “所有患者共享 Lambda”未明确是否六个不同图也必须共用一个数值。默认
   用六组训练先验估计值中的全局最大值，确保所有组和患者共用；同时保留
   `per_group` 诊断/消融开关。
5. `B^T diag(w) B` 只有在 B 是 oriented edge-node incidence 时才是图
   Laplacian。实现固定采用每边一个 `+1/-1` 的 oriented incidence；边方向
   只影响 B 的符号，不影响该算子。
6. 任务中的 A0--Full 标签与 Word 内消融表的 A0 命名相冲突。配置和结果
   内部使用描述性名称（如 `original_ddkac`、`moment_only`、
   `full_pc_cmka`），导出表时再附来源标签，避免误覆盖结果。
7. “低秩 H 模式”没有指定秩和余项。实现采用可配置 top-r eigenspace，并
   对剩余空间使用阻尼对角/各向同性余项，避免奇异逆平方根。

## 6. 文件边界

新增：

- `models/layers/pc_cmka_spectral.py`：参考谱算子、探针、Chebyshev 递推与矩。
- `models/layers/pc_cmka_calibration.py`：字典、目标网络和展开求解器。
- `models/layers/pc_cmka_augmentation.py`：Hessian/对照增强与 Krylov 约束。
- `models/layers/pc_cmka_ddkac_core.py`：共享路由 DD-KAC、SSL 和可辨识性。
- `models/layers/pc_cmka_encoder.py`：单组与六组组合编码器。
- `models/layers/pc_cmka_ddkac.py`：向后兼容的公开导出层。
- `utils/pc_cmka_graph.py`：只从训练折构造六组先验 edge support/weights、
  缓存和 shape 验证。
- `utils/pc_cmka_diagnostics.py`：JSON/CSV/NPZ 可序列化诊断聚合。
- `tests/test_pc_cmka_ddkac.py`：数值、梯度、模式、泄漏和集成测试。
- `configs/pc_cmka_ddkac.json`：默认值和消融矩阵。
- `docs/PC_CMKA_DDKAC.md`：公式、假设、运行命令和结果字段。

最小修改：

- `models/layers/genomic_encoders.py`：只注册新 encoder 名称和构造入口。
- `models/model_HGNN.py`：透传新配置及细粒度辅助损失/诊断，不改变旧分支。
- `utils/core_utils.py`：训练折先验构建、总损失分项、fold 诊断落盘；另行修复
  严格训练折生存分箱路径。
- `utils/process_args.py`：增加所有可关闭的命令行参数。
- `scripts/run_cached_multicohort_experiments.py`：显式透传配置，不改变旧实验
  默认值。

## 7. 分阶段实施与验收门

1. Phase 1：`ReferenceSpectralOperator`、固定 D0/共享 Lambda、dense/sparse
   一致性与稳定性测试。
2. Phase 2：归一化探针、显式 Chebyshev 递推和响应矩；验证六组 shape、
   `a=0` 回到先验。
3. Phase 3：每组低秩字典、加权正交正则、目标 offset/precision 网络、固定
   迭代可微求解器，以及 detach/fixed/target-only/joint 模式。
4. Phase 4：低维 Gauss--Newton exact/diagonal/low-rank 逆平方根、反向配对
   和正值/中心不变量。
5. Phase 5：患者 Krylov 基、安全缩放、基础图共享路由、无负样本 SSL。
6. Phase 6：随机 JVP 的可辨识性切空间正则及关闭/近似/完整模式。
7. Phase 7：接入 MRePath、总损失配置、逐患者/逐折诊断、消融配置、训练折
   分箱无泄漏修正和 COREAD/STAD 小样本 smoke test。

每个阶段只有在单元测试、有限值检查和梯度检查通过后才进入下一阶段。
旧 `dd_kac` 的默认执行路径和数值行为保持不变。
