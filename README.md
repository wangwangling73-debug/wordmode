# PushT：训练与推理修正版

文件用途与目录关系见 [PROJECT_GUIDE.md](PROJECT_GUIDE.md)。

入口仍是 `wm_pusht8g.py` 和 `visualize_pusht.py`。使用真实 PushT 专家示范训练状态输入的轻量扩散策略；不是图像策略，也不是论文官方复现。

## 下载仓库后运行

仓库包含代码、默认改进模型 `runs_improved/policy.pt`、最终训练权重、旧模型对照、评估记录和示例动画，以及 `runs_world/` 与 `runs_world_manual/` 两组世界模型权重、训练日志和摘要。训练数据、重复实验权重、缓存和论文阅读临时文件不纳入版本管理，本地文件仍保留。

先创建 Python 环境，安装适合自己硬件的 PyTorch，再安装项目依赖：

```powershell
python -m pip install -r requirements.txt
python wm_pusht8g.py check
python visualize_pusht.py --mode policy --device cpu --seed 10002 --steps 300
```

已有模型的评估和动画无需重新训练。需要训练时，先运行 `python wm_pusht8g.py download` 获取数据，再按下文“从头训练”操作。文档中的 `E:\Condaenv\wm8g` 是原开发机器的环境位置，在其他电脑上请使用自己的 Python 环境。

## 本次修复

- 原版 32 步线性噪声日程末端仍保留 72.34% 的信号，推理却从纯噪声开始。改为 100 步余弦日程，末端信号约 2.43e-7。
- 新训练使用 v-prediction，在高噪声阶段重建动作时不除以接近零的数；每次去噪都约束动作范围。旧权重仍按其原来的 epsilon 预测方式加载。
- 使用 sin/cos 将状态转换成 T 字的 8 个轮廓顶点，再加入小球在物块坐标系中的相对位置，消除 0 与 2π 的跳变。策略默认采用轻量时序 U-Net（64/128/256 通道），通过条件缩放/偏移注入观测与去噪时刻；卷积沿未来动作序列建模。
- 默认预测从当前时刻开始的 16 个未来动作，连续执行前 4 个再规划。评估和动画共用 `ChunkController`，每步更新观测，每局清空历史和动作队列。
- 增加 EMA 权重、学习率预热与余弦衰减、全部留出轨迹验证、固定验证噪声。训练/验证按完整轨迹划分，使用 `--seed`，不跨轨迹取窗口。
- 保存验证噪声损失最低的 EMA 为 `policy.pt`，另存最后一次检查的 EMA 为 `policy_last.pt`；后续训练会记录 `training_metrics.jsonl`。验证损失不是任务成功率，仍须执行仿真评估。
- 允许小于 batch size 的数据集训练；合成调试权重自动放进 `synthetic_debug` 子目录。
- 评估和动画默认最多 300 步，显式设置环境时限；GIF 使用毫秒时长并包含最后一帧。

旧 `runs/`、`runs_v2/` 和 v2 脚本保留。修改前的原版代码和 README 在 `backups/before_improvement/`。

## 使用已有改进模型

在项目目录和原来的 `wm8g` 环境中运行：

```powershell
conda activate wm8g
python wm_pusht8g.py eval --device cuda --mode policy --episodes 50 --seed 20000 --report outputs/eval_50.json
python visualize_pusht.py --mode policy --device cpu --seed 10002 --steps 300
python visualize_pusht.py --mode policy --device cpu --seed 10002 --steps 300 --record outputs/policy.gif
```

默认读取脚本旁 `runs_improved/policy.pt`。上面的动画命令是一个已验证的成功示例；成功率请看多局评估。当前模型、失败数量和本机测试结果见 `RESULTS.md`。

## 从头训练

```powershell
python wm_pusht8g.py check
python wm_pusht8g.py smoke --device cuda
python wm_pusht8g.py download
python wm_pusht8g.py inspect
python wm_pusht8g.py train-policy --device cuda --steps 30000 --batch-size 128 --hidden 64 --horizon 16 --log-every 1000 --out-dir runs_new
python wm_pusht8g.py eval --device cuda --checkpoint runs_new/policy.pt --episodes 50 --seed 20000 --report outputs/new_eval.json
python visualize_pusht.py --mode policy --device cuda --checkpoint runs_new/policy.pt --record outputs/new.gif
```

每次 `train-policy` 都从头训练，`--steps` 表示优化器更新次数；没有隐式接着旧权重训练。请为新实验指定新的 `--out-dir`，同目录下的同名文件会被覆盖。新结构需要重新训练，旧权重不能直接转换成新结构。

`--action-steps 4` 控制连续执行长度，可以用独立验证种子比较 4 与 8。`--sample-steps 32` 控制去噪次数，必须在 2 与模型训练扩散步数之间。不要只根据单次动画判断成功率。

## 旧模型对照

使用备份代码能严格重现原版采样行为：

```powershell
python backups/before_improvement/wm_pusht8g.py eval --device cpu --mode policy --episodes 10 --seed 10000 --max-steps 300
```

新代码也支持显式加载旧权重，但使用修复后的采样限幅及动作队列，因此不等于原版基线：

```powershell
python wm_pusht8g.py eval --device cuda --checkpoint runs/policy.pt --sample-steps 32 --action-steps 4
```

## 世界模型辅助（新版）

现在的世界模型使用最近 4 帧状态和已执行的 3 个动作，预测接下来 4 步。3 个独立初始化、分别按回合重采样训练的预测器组成集成。训练时连续使用模型自己的预测作为下一步输入，避免只学会单步预测、部署时却要求连续预测的问题。

控制流程是：原策略提出动作 → 再生成候选 → 世界模型模拟短期结果 → 按 T 形覆盖率、目标位置和角度、动作平滑度以及预测分歧评分 → 执行前 4 步 → 用真实观测重新规划。覆盖率按环境中的 T 形多边形交叠面积计算。预测器分歧较大、候选偏离训练数据，或相对原动作的改善不够明显时，保留原策略动作。可靠性阈值来自完整验证集的 95% 分位数，是筛选启发式，不能保证预测正确或成功率提升。

本次训练、对照结果和局限见 [WORLD_MODEL_RESULTS.md](WORLD_MODEL_RESULTS.md)。原策略权重无需重新训练。

```powershell
conda activate wm8g
python wm_pusht8g.py train-world --device cuda --steps 4000 --batch-size 256 --hidden 128 --lr 0.0003 --world-history 4 --world-horizon 4 --world-members 3 --out-dir runs_world
python wm_pusht8g.py eval --device cpu --mode policy --episodes 20 --seed 50000 --report outputs/policy_compare.json
python wm_pusht8g.py eval --device cpu --mode world --world-checkpoint runs_world/world.pt --episodes 20 --seed 50000 --report outputs/world_compare.json
python visualize_pusht.py --mode world --device cpu --world-checkpoint runs_world/world.pt --seed 42 --steps 300
```

如果希望手动训练更久并保留本次结果，可以运行：

```powershell
python wm_pusht8g.py train-world --device cuda --steps 10000 --batch-size 256 --lr 0.0003 --out-dir runs_world_manual
python wm_pusht8g.py eval --device cpu --mode world --world-checkpoint runs_world_manual/world.pt --episodes 20 --seed 60000 --report outputs/manual_world20.json
python wm_pusht8g.py eval --device cpu --mode policy --episodes 20 --seed 60000 --report outputs/manual_policy20.json
```

训练次数增加不保证控制效果提高；请比较相同种子下的成功率。如果将某组种子用于调整参数，最终汇报时再使用未参与调整的新种子。

`runs_world/world.pt` 保存完整验证集上表现最好的世界模型，`world_last.pt` 保存最后一次；`world_summary.json` 记录逐步预测误差、数据划分和筛选阈值。每次训练从头开始，请用不同输出目录保留不同实验。`runs_world/` 和 `runs_world_manual/` 的权重、训练日志与摘要已纳入版本管理，克隆后可直接加载。前者为 4,000 次更新的已评估实验；后者为手动训练 20,000 次更新、学习率 0.0004 的实验，目前没有对应的独立成功率报告。旧单步世界模型仍可加载，但没有新版可靠性检查，会显示提示。

- `--candidates 4`：候选数，含原策略动作。
- `--action-steps 4`：连续执行步数。
- `--plan-steps 4`：预测评分步数，默认与执行步数相同；必须满足执行步数 ≤ 预测步数 ≤ 世界模型训练跨度，且不能超出策略输出长度。
- `--uncertainty-weight 0.15`：评分中对预测器分歧的惩罚。
- `--switch-margin 0.02`：候选评分至少优于原动作多少才允许替换，分数越小越好。

`sample4` 生成候选但只执行第一个，适合核对采样对照。新版第一个候选单独生成，额外候选不会消耗原策略的随机数流；在相同设备、种子和观测下，回退动作与 `policy` 模式相同。该行为与旧版批量采样不同。评估 JSON 同时记录成功率、最终覆盖率、每步控制耗时、动作替换次数与回退原因。

模型当前只用演示数据训练。失败或偏离演示轨迹后的状态可能预测不准；若对照没有稳定提高成功率，后续应优先补充策略实际运行产生的转移数据，再重新验证，而非仅提高候选数或延长预测。

## 环境和检查

本机已有环境 `E:\Condaenv\wm8g`。依赖见 `requirements.txt`；PyTorch 单独安装。RTX 5060 使用支持其架构的 CUDA 构建，本次实测环境是 PyTorch 2.8.0+cu128。

```powershell
python -m unittest test_pusht test_world -v
python wm_pusht8g.py train-policy --device cpu --steps 2 --hidden 32 --batch-size 512 --synthetic --out-dir outputs/test_checkpoints
```

合成数据只用于检查代码，不用于报告任务效果。显存不足时先减小 `--batch-size`。数据下载地址未改变：<https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip>。

## 参考

- [Diffusion Policy 官方配置](https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/config/train_diffusion_unet_lowdim_workspace.yaml)：余弦噪声、EMA、动作序列等设计参考；这里使用更小的条件时序 U-Net、几何状态特征和 v-prediction，并非论文配置的直接复现。
- [gym-pusht 环境源码](https://github.com/huggingface/gym-pusht/blob/main/gym_pusht/envs/pusht.py)：状态顺序、动作坐标和覆盖率成功判定。
