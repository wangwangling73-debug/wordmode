# PushT：训练与推理修正版

入口仍是 `wm_pusht8g.py` 和 `visualize_pusht.py`。使用真实 PushT 专家示范训练状态输入的轻量扩散策略；不是图像策略，也不是论文官方复现。

## 下载仓库后运行

仓库包含代码、默认改进模型 `runs_improved/policy.pt`、最终训练权重、旧模型对照、评估记录和示例动画。训练数据、重复实验权重、缓存和论文阅读临时文件不纳入版本管理，本地文件仍保留。

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

## 世界模型（可选实验）

先验证 `policy` 模式。单步世界模型的多步误差可能使候选选择变差，不能保证 `world` 比 `policy` 好。

```powershell
python wm_pusht8g.py train-world --device cuda --steps 10000 --batch-size 128
python wm_pusht8g.py eval --device cuda --mode world --episodes 50 --seed 20000
```

世界模型默认保存在 `runs_improved/world.pt`。若在其他位置，评估或动画中传入 `--world-checkpoint 路径`。`sample4` 仍是生成多个候选只执行第一个的采样批量对照，不进行择优。

## 环境和检查

本机已有环境 `E:\Condaenv\wm8g`。依赖见 `requirements.txt`；PyTorch 单独安装。RTX 5060 使用支持其架构的 CUDA 构建，本次实测环境是 PyTorch 2.8.0+cu128。

```powershell
python -m unittest test_pusht -v
python wm_pusht8g.py train-policy --device cpu --steps 2 --hidden 32 --batch-size 512 --synthetic --out-dir outputs/test_checkpoints
```

合成数据只用于检查代码，不用于报告任务效果。显存不足时先减小 `--batch-size`。数据下载地址未改变：<https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip>。

## 参考

- [Diffusion Policy 官方配置](https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/config/train_diffusion_unet_lowdim_workspace.yaml)：余弦噪声、EMA、动作序列等设计参考；这里使用更小的条件时序 U-Net、几何状态特征和 v-prediction，并非论文配置的直接复现。
- [gym-pusht 环境源码](https://github.com/huggingface/gym-pusht/blob/main/gym_pusht/envs/pusht.py)：状态顺序、动作坐标和覆盖率成功判定。
