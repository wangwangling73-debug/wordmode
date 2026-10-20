# PushT 改进结果（2026-10-18，本机实测）

已修改 `wm_pusht8g.py`、`visualize_pusht.py`，默认模型位于 `runs_improved/policy.pt`，可以直接运行，无需先重新训练。

## 最终独立对照

两组均使用 gym-pusht 0.1.6 的原生成功判定（覆盖率 > 95%），每局最多 300 步，CPU 推理，50 个相同初始种子 **30000–30049**。未修改目标、物理参数或成功阈值。此组种子未用于开发时的模型/动作长度对比。

| 模型 | 成功局数 | 成功率 | 每局最高奖励的平均值 |
| --- | ---: | ---: | ---: |
| 原版已有的 30000 步模型 | 0/50 | 0% | 0.126 |
| 改进后的时序 U-Net / EMA | 30/50 | 60% | 0.957 |

仍有 **20 局失败**。奖励接近 1 不代表成功；判定采用环境 `is_success`，没有四舍五入后计成功。这是本机有限样本对照，不是论文官方基准，也不是等训练算力的单因素消融。

原始记录：
- `backups/before_improvement/baseline_final50.txt`：原版完整控制台记录。
- `outputs/baseline_final50.json`：从原版控制台整理，单局奖励精度为 3 位小数。
- `outputs/final_holdout50.json`：新版逐局奖励、成功标记、种子、模型配置及运行参数。

## 训练与模型选择

- 使用本地真实 PushT 数据：206 条轨迹、25650 帧。按完整轨迹划分，23190 个训练窗口、2460 个验证窗口；划分种子 42。
- RTX 5060 / PyTorch 2.8.0+cu128 完成 **30000 次优化器更新**，batch size 128，初始学习率 1e-4。
- 轻量时序 U-Net：64/128/256 通道，约 188 万参数；2 帧观测，预测 16 步动作，执行前 4 步后再规划。
- 100 步余弦扩散训练、v-prediction；部署使用 32 步 DDIM。状态转换为 T 轮廓关键点及小球局部坐标。
- 评估使用验证误差最低的 **第 10000 步 EMA 权重**。其验证损失为 0.039321；第 30000 步验证损失反而升至 0.052456。没有直接采用最后一步模型。
- `runs_improved/policy_last.pt` 保存第 30000 步 EMA；`runs_improved/training_summary.json` 保存元信息。
- 选出的模型 SHA256：`B5BFBB68C17D95DD640CD087B9B89A057E39CD5CD64349612706163D42BCC271`。

训练 10000 步的单独实验不等价于上述 30000 步训练过程中的第 10000 步：学习率衰减长度不同。请用下面的训练命令复现整个流程。

## 根因和修复

1. 原版 32 步线性日程的末端累计信号为 0.7234，却从纯高斯噪声开始推理；改为末端约 2.43e-7 的余弦日程。
2. 高噪声下 epsilon 反演会放大误差；新训练使用稳定的 v-prediction，每一步约束干净动作估计。
3. 原版小 MLP 对轨迹与接触几何的建模不足；改成带条件缩放/偏移的时序 U-Net，并显式提供 T 的几何特征。
4. 原版每个环境步都重新随机采样，只执行一个动作；现在使用共享控制器连续执行 4 步，逐步更新历史，避免频繁改变推动方向。
5. 增加 EMA、预热/衰减、全部留出轨迹的固定噪声验证、最佳权重保存、完整评估报告。后续训练会自动记录 `training_metrics.jsonl`。

开发时的 MLP 改进版本虽然增加了平均最高奖励，但仍有整组测试零成功，因此未作为最终默认模型。它们保留在 `runs_improved_epsilon/`、`runs_improved_velocity/`、`runs_improved_geometry/`；最终训练目录为 `runs_improved_unet/`。

## 直接运行与复现

在 `E:\WorldM8g` 目录激活 `wm8g` 环境：

```powershell
conda activate wm8g
python wm_pusht8g.py eval --device cpu --episodes 50 --seed 30000 --report outputs/repeat_final50.json
python visualize_pusht.py --mode policy --device cpu --seed 10002
```

`outputs/improved_success_example.gif` 是种子 10002 的成功案例，199 步完成；这是单个示例，整体效果以 50 局结果为准。CPU 与 CUDA 的随机数序列不同，使用 CUDA 推理时不保证相同种子逐局得到相同结果。

从头训练新实验，使用新目录避免覆盖本次模型：

```powershell
python wm_pusht8g.py train-policy --device cuda --steps 30000 --batch-size 128 --hidden 64 --horizon 16 --log-every 1000 --out-dir runs_new
python wm_pusht8g.py eval --device cpu --checkpoint runs_new/policy.pt --episodes 50 --seed 30000 --report outputs/new_eval.json
```

原版的 `runs/policy.pt`、v2 文件和权重均保留，原始代码备份在 `backups/before_improvement/`。本次没有重新训练世界模型；`world` 模式仍是可选实验，可能受预测误差影响。

## 验证范围

- 9 项回归测试通过：扩散末端噪声、两种预测方式的采样公式、角度周期性、几何与物理环境一致性、窗口边界与轨迹划分、连续动作执行与重置、旧权重兼容、无效参数、U-Net 多种轨迹长度及反向传播。
- CUDA 前向/反向/采样检查通过。
- 合成数据的短训练、保存、重新加载、真实环境两步推理链路通过；合成权重隔离存储，不用于上面的任务指标。
- GIF 已验证帧数、播放时长和末帧；训练数据与 50 局评估均为真实示范/物理仿真，不是合成成绩。

设计参考：[Diffusion Policy 官方配置](https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/config/train_diffusion_unet_lowdim_workspace.yaml)、[gym-pusht 环境源码](https://github.com/huggingface/gym-pusht/blob/main/gym_pusht/envs/pusht.py)。
