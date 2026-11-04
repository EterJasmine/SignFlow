# 专用时序模型工作流

## 一、边界检测（现在走这条）

真值在 `evaluation/truth.json`，特征在 `data/assets/<video_id>/result.json` 的 `features` 里，不需要重新跑视觉检测：

```powershell
# 训练集 2/5/6 三条，留出 3/4 两条；两族都训，选型只看训练集留一
../.venv/Scripts/python.exe scripts/train_boundary.py
# 只训变化能量 + 线性（几秒），或只训 TCN
../.venv/Scripts/python.exe scripts/train_boundary.py --kinds linear
../.venv/Scripts/python.exe scripts/train_boundary.py --kinds tcn --epochs 60
```

产出 `training/runs/<时间>-boundary-<族>/best.pt` 与 `history.json`，报告写 `evaluation/results/train-boundary.md`（同目录 `.json` 存完整选型表与逐条结果）。

协议固定为：按源视频分组切分 → 训练集内部留一挑「模型族/归一化/阈值/最小步长」→ 全量重训 → 留出集只评一次。两族都报，不因留出好坏改选型。

两个族的分工：

| 族 | 输入 | 特点 |
| --- | --- | --- |
| `tcn` | 76 维原始特征 × 64 帧窗 | 能表达上下文，但 3 条视频的量级会直接记住（训练 0.98 / 留出 0.35） |
| `linear` | 76 维压成 14 维「变化能量」，逐帧线性 | 容量小、跨视频稳（训练 0.56 / 留出 0.62），当前入选 |

`linear` 的变化能量就是手型换态检测器在用的量：每个特征组（双手手型、双手运动、杂项）算「前 2 帧中位数」与「后 2 帧中位数」的逐维差，取均值与最大值，再逐视频标准化。训练与推理共用 `training/features_diff.py` 的 `change_features()`，避免"训练一套、上线另一套"。

启用：`python scripts/serve_model.py` 自动取最新的 `*-boundary-linear/best.pt`，设好 `GESTUREFLOW_MODEL` 与 `GESTUREFLOW_SEGMENTER=model` 起服务；阈值、最小间隔、合并强度都从权重的 metadata 读，`training/boundary_post.py` 是训练与部署共用的取峰+合并实现。回退纯规则分段用 `启动视觉系统.cmd`。

## 二、按动作实例训练（类别/阶段/手型）

先在页面标注真实视频并人工确认，通过 `/api/v1/datasets/exports?length=32&group_by=video` 导出NPZ及其JSON manifest。两个文件必须同名放在一起。至少3个互不泄漏的来源组才能进入训练；实际有效训练还需要足够覆盖每个类别的标注，不能把3组视为充分数据量。

```powershell
# 当前机器使用工作区 ../.venv；CUDA包从PyTorch官方源安装
python -m pip --python ../.venv/Scripts/python.exe install torch --index-url https://download.pytorch.org/whl/cu128
../.venv/Scripts/python.exe -m training.train data/datasets/<id>.npz --model tcn --task phase --batch-size 4 --device cuda
../.venv/Scripts/python.exe -m training.train data/datasets/<id>.npz --model stgcn --task category --batch-size 4 --device cuda
```

可选task：phase/category/static/boundary。训练和标签集分别独立选择，mask明确进入网络/损失。每个run保存随机种子、数据SHA256、最佳权重、early stopping日志、混淆矩阵及峰值CUDA内存；不存在数据时不会自动运行。

TCN输入B×T×F，mask拼接为额外通道，4层膨胀残差卷积。ST-GCN-style输入B×T×75×2+mask，75节点明确为33个MediaPipe Pose+21左手+21右手；self/inward/outward三分区图卷积与时间卷积，不使用COCO/NTU索引。没有可靠解剖学手侧时相应节点mask为false。它是本项目轻量实现，不声称复现MMAction2原模型指标。

模型输出phase(4)、boundary、category(9)、static(9)。`GESTUREFLOW_MODEL`指定人工数据训练的best.pt后启用可选推理，校验特征版本和数据来源；缺文件/缺依赖/低置信度回退规则。高置信边界或hold→transition可辅助细分规则步骤，输出仍要求人工审核。

`scripts/check_models.py`只用随机张量测试前后向、内存和执行时间，不生成可用于识别的权重，不代表训练成功。

真实评测：`python -m evaluation.run <human-paired-predictions.json>`。按原视频计算类别/场景Macro-F1、Boundary-F1、Edit、IDF1和人工图解一致性。覆盖率单列；没有人工真值不计算准确率。参考：[PyTorch安装](https://pytorch.org/get-started/locally/)、[AMP](https://docs.pytorch.org/docs/stable/amp.html)、[MMAction2骨骼模型](https://mmaction2.readthedocs.io/en/latest/model_zoo/skeleton.html)。
