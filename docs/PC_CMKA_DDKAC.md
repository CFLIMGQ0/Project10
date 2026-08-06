# PC-CMKA-DDKAC 实现说明

## 1. 方法范围与接口

新编码器配置名为 `pc_cmka_ddkac`。它只替换 MRePath 的六组基因编码器，
输出保持 `[batch, 6, d]`；病理 ResNet50 特征、病理图、SHGNN、动态模态
加权、IFA、离散时间生存头和公开五折成员均不改。旧 `dd_kac`、`original`、
MLP、GCN 和五个 KAN 变体仍使用原分支。

六组仍是 `signatures.csv` 的 Tumor Suppressor Genes、Oncogenes、Protein
Kinases、Cell Differentiation Markers、Transcription Factors、Cytokines and
Growth Factors。COREAD/STAD 当前实际维数均为 53/173/216/252/387/200。

## 2. 公式与代码对应

| Word 定义 | 实现 |
|---|---|
| `S(w)=D0^-1/2 B^T diag(w) B D0^-1/2` | `ReferenceSpectralOperator.matvec`；oriented edge-node incidence 的边列表 MVP，D0 是固定 buffer。 |
| `S_hat=2S/Lambda-I` | `apply_scaled`；默认全六组共享理论界 `2 exp(rho_max)`，前向不做特征分解。 |
| `u=(m*x)/(norm+eps)` | `normalized_patient_probe`；当前组内局部输入等价于全一 mask。 |
| `mu_r=u^T T_r(S_hat)u` | `chebyshev_recurrence` 与 `ChebyshevMomentResponse`；记录 r=1..R。 |
| 只预测 `Delta_mu, pi>0` | `MomentTargetNetwork`；offset 使用有界 tanh，precision 使用 softplus。 |
| `P in R^(E x K)` | 每功能组一个 `EdgeDeformationDictionary`；加权正交初始化与 `P^T diag(w0)P-I` 正则。 |
| `w=w0*exp(Pa)` | `ReferenceSpectralOperator.patient_weights`；log deformation 限幅，防止 exp 溢出。 |
| 矩逆校准与最小 rho | `DifferentiableMomentSolver`；固定步数近端梯度，`rho=||Pa||inf` 并投影到 `rho_max`。 |
| fixed/detach/target-only/joint | `calibration_mode` 四种 solver 模式；A2 另有 `direct_edge_gate` 对照。 |
| `H=J^T Pi J+beta I` | `CalibrationUncertaintyAugmentor` 在 K 维空间计算矩 Jacobian 与 H。 |
| exact/diagonal/low-rank | `hessian_mode` 三种逆平方根方向。 |
| `w+/-=w*(1+/-xi)` | 同一 augmentor；测试严格验证中心和正值。 |
| Krylov 安全约束 | `build_krylov_basis`、`krylov_operator_error`；用 Frobenius 上界缩放 xi。 |
| 基础图共享频率路由 | `SharedRouteDDKACPathway`；base/+/- 只使用一次 route logits；A8 前的独立路由可作为消融。 |
| 值域/结构/残差/门控 | 保留 8 中心值域基函数、结构 Chebyshev、线性残差、sigmoid 双分支 gate 和 LayerNorm。 |
| 无负样本 SSL | `negative_free_consistency_loss`；结构、DD 内部融合 token、gate 三项，支持 cosine + VICReg variance/covariance。 |
| `J_a` 与 `J_theta` | `IdentifiabilityTangentRegularizer`；theta 明确为基础图 frequency-route logits，支持 off/randomized/full。 |
| 总损失 | 生存损失外加 moment/trust/ssl/id/dictionary/DD-KAC consistency；所有 lambda 来自 JSON。 |

## 3. 保守实现假设

六组图的节点和边身份不同，所以不能共享同一个形状固定的 P。实现采用每组
一个 P、共享超参数；整体等价于块对角字典。患者系数也采用 `a_(p,g)`，与
“扰动按患者和功能组生成”一致。

给定 a 后，Word 目标中 rho 只有正线性代价和上界约束，最优 rho 必为
`||Pa||inf`。因此 rho 由求解状态确定，不使用单独 MLP。

仓库没有发布 PPI/Reactome 边文件。当前可运行回退先验是**仅训练折 RNA**
的每组 absolute-Pearson top-8 图，输出明确标记为 `training_correlation`；它
是固定折内先验，但不能在论文中称为外部生物相互作用图。接入真正的生物
先验时只需提供同一 `PathwayGraphPrior` 接口。

## 4. 泄漏控制

- 六组先验只读取 `train_split`；验证/测试不参与边支撑、w0、D0、Lambda、
  字典初始化或矩统计。
- PC-CMKA 运行会强制用训练折未删失病例拟合 4 个 DSS 边界，再固定应用到
  训练和验证病例。`--fold_survival_bins` 可让 A0 等对照采用同一严格标签。
- RNA scaler 仍保持仓库现有方式：仅训练折拟合，但将整个训练矩阵展平为
  一个全局 MinMax scaler。此行为为了第一阶段最小侵入而保留。
- 验证和测试在 `eval()` 下不采样增强；求 a 所需的一阶局部梯度由内部短暂
  `enable_grad()` 计算，随后 detach，不建立训练高阶图。

## 5. 配置与消融

默认配置是 [pc_cmka_ddkac.json](../configs/pc_cmka_ddkac.json)。内部使用描述性
名称，避免任务文本与 Word 表中 A0 命名冲突。已支持：

- A0 原 DD-KAC 固定图；
- A1 参考度算子固定图；
- A2 患者直接全边门控；
- A3 Chebyshev 矩逆校准；
- A4 普通随机删边；
- A5 独立随机双视图；
- A6 Hessian 反向配对；
- A7 加 Krylov；
- A8 加共享路由；
- A9 加结构一致性；
- A10 加切空间正则；
- Full 完整结构/融合/gate SSL；
- 额外有效电阻增强对照。

Hessian 默认 diagonal；`exact` 和 `low_rank` 是可选模式。完整 tangent 模式
只建议做小规模诊断，正式五折默认 randomized JVP。

## 6. 运行命令

先激活当前环境并进入仓库：

```bash
conda activate mrepath-train
cd /mnt/e/MRePath
```

COREAD 单折正式训练：

```bash
python scripts/run_pc_cmka_ablations.py \
  --dataset coadread \
  --ablations full_pc_cmka_ddkac \
  --folds 0 \
  --max-epochs 30 \
  --num-workers 8
```

COREAD 五折顺序训练：

```bash
python scripts/run_pc_cmka_ablations.py \
  --dataset coadread \
  --ablations full_pc_cmka_ddkac \
  --max-epochs 30 \
  --num-workers 8
```

STAD 五折只需把 `--dataset coadread` 改为 `--dataset stad`。运行全部消融时
省略 `--ablations`：

```bash
python scripts/run_pc_cmka_ablations.py \
  --dataset coadread \
  --max-epochs 30 \
  --num-workers 8
```

先检查全部命令但不训练：

```bash
python scripts/run_pc_cmka_ablations.py --dataset coadread --dry-run
```

## 7. 诊断产物

每折在结果目录保存：

- `s_F_pc_cmka_config.json`：最终解析配置、训练折分箱、prior source；
- `diagnostics/fold_F_patients.jsonl/csv`：rho、组偏移、缩放、Krylov、gate 等；
- `diagnostics/fold_F_diagnostics.npz`：目标/实际矩、残差、solver 曲线、边权、
  top edges、Hessian、routes、tangent correlation；
- `diagnostics/fold_F_summary.json`：C-index、best epoch、时间和 peak GPU bytes；
- 原训练器的 checkpoint、`summary.csv` 和病例级结果保持不变。

损失分项每个训练 epoch 单独打印，并保存在 encoder 的
`auxiliary_losses`，不再只有总 loss。

## 8. 已完成测试与 smoke test

单元/集成测试覆盖 shape、六图、稀疏/稠密一致性、显式 Chebyshev、多模式
solver、先验回退、正负中心/正值、三种 Hessian、Krylov、共享/独立路由、
随机种子、eval 无增强、图/route JVP、直接门控及所有增强对照、训练折分箱、
诊断导出和 MRePath 整体前向/反向。

真实 COREAD fold-0 基因链 smoke test：237 个训练病例、58 个验证病例；六组
边数 309/1126/1389/1716/2535/1330；输出 `(1,6,32)` 全有限；训练折 DSS
边界为 `[-inf, 11.2333, 18.4833, 33.95, inf]`；零 offset 初始化时六组 rho
均严格为 0。

训练器级 smoke test 也已完成：COREAD fold-0、A1 参考算子固定图、1 epoch、
16 patches、病理 MLP 调试聚合，完整经过 optimizer、验证、best checkpoint
和 58 例结构化诊断。运行 90.16 秒、峰值 GPU 308,871,168 bytes，验证
C-index 0.7561。该数值只证明工程链路可运行，不能与 4096-patch、SHGNN、
30-epoch 正式实验比较，也不得作为方法性能结论。

## 9. 已知限制

1. 当前先验是训练折相关图回退，不是外部生物知识图；正式方法论文前应补充
   版本化 PPI/Reactome 来源、基因 ID 映射与覆盖率。
2. MRePath 的 WSI 图路径仍要求 batch size 1；`_collate_MCAT` 对更大 batch
   不能表示不同患者图。
3. 有效电阻只是对照，初始化时会在单组小图上计算一次稠密伪逆；最终方法
   不使用它。
4. exact Hessian 仍只在 K 维系数空间计算，但比 diagonal 慢；完整 tangent
   Jacobian 只用于高级消融。
5. 当前基因输入是 RNA-only，不应描述为论文完整多组学条件。
