# Windows 11 / RTX 5060 8GB GPU: 轻量 PushT 世界模型 + 小脑策略

这个项目是基于**真实 PushT 专家示范数据**的小规模研究入门工程：
- 小脑：自编写的轻量 **MLP DDPM 噪声预测策略**，输入过去2帧的5维状态，生成8步、每步2维的动作序列。
- 世界模型：自编写的 **1步动作条件状态转移 MLP**，用预测终点对若干动作候选进行评分。
- 测试环境：Hugging Face 官方 `gym-pusht/PushT-v0`，Windows 原生 CPU 物理仿真；PyTorch 在 NVIDIA GPU 上训练。

**注意：不是 Diffusion Policy / TD-MPC2 论文作者仓库的逐行复现。** 官方数据及任务来自开放基准，模型为专门适配8GB显存的教学科研简化版本。**不保证训练后能够完成 PushT**，需要以实际成功率、奖励和测试集误差为准。


## RTX 5060（sm_120）兼容性与排错

RTX 5060 属于 NVIDIA Blackwell，旧的 `torch==2.5.1+cu121` 不支持 `sm_120`，会出现 `no kernel image is available for execution on the device`。

在原 `wm8g` 环境中升级（不必重装其他依赖）：

```powershell
conda activate wm8g
python -m pip install --upgrade "torch==2.8.0" --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.get_arch_list()); x=torch.randn(256,256,device='cuda',requires_grad=True); (x@x).sum().backward(); print('GPU PASS')"
python wm_pusht8g.py smoke --device cuda
```

应看到 CUDA 12.8 和包含 `sm_120` 的架构列表。如果安装包下载中断，可使用官方轮子目录 `https://download.pytorch.org/whl/cu128/torch/` 中的 Windows Python 3.10 包并断点续传。

## Windows Python 3.10 兼容性修复（2026-10-16）

之前的依赖文件误锁定了 `numcodecs==0.15.1`（Python >=3.11），
本包已改为 `numcodecs==0.13.1`（Python >=3.10），与 `zarr==2.18.3` 配套。

如果已经有 `wm8g` 环境，无需重新创建。如果你的显卡是 RTX 5060（sm_120），需要将旧版 PyTorch 2.5.1+cu121 升级为 PyTorch 2.8.0+cu128。
在项目目录下直接执行：

```powershell
conda activate wm8g
python -m pip install -r requirements.txt -i https://pypi.org/simple
python -m pip check
python -c "import zarr, numcodecs; print(zarr.__version__, numcodecs.__version__)"
python wm_pusht8g.py check
```

## 1 环境依赖（建议 PowerShell / Anaconda Prompt）

先安装 [Miniforge](https://github.com/conda-forge/miniforge)、更新 NVIDIA 驱动，然后在已安装 conda 的终端执行：

```powershell
conda create -n wm8g python=3.10 -y
conda activate wm8g
python -m pip install --upgrade pip
python -m pip install --upgrade "torch==2.8.0" --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -m pip check
```

CUDA 12.8 是所选 **PyTorch wheel** 的构建版本，通常不必另外安装完整 CUDA Toolkit；但 NVIDIA 驱动需要支持 CUDA 12.8。RTX 5060 为 Blackwell（sm_120），不可使用旧版 2.5.1+cu121 wheel。

**请按顺序测试：**

```powershell
nvidia-smi
python wm_pusht8g.py check
python wm_pusht8g.py smoke --device cuda
```

如 GPU 不能使用，先试 `python wm_pusht8g.py smoke --device cpu`，诊断 PyTorch 与驱动，不要开始完整训练。

## 2 下载原始数据

Stanford Diffusion Policy 官方数据页：
https://diffusion-policy.cs.columbia.edu/data/training/

`pusht.zip` 约29 MB；内部包含 `pusht_cchi_v7_replay.zarr`。**脚本从官网直接下载，绝不生成假数据冒充专家示范。**

```powershell
python wm_pusht8g.py download
python wm_pusht8g.py inspect
```

若 Python 下载因网络失败，可以在浏览器中打开官方链接并手动将 `pusht.zip` 保存到此工程的 `data` 文件夹，然后再次执行 `download`。如果已经解压数据也会跳过。

期望 `inspect` 打印：`state=(N,5)`, `action=(N,2)`、episode数量；确切数据条数由下载数据而定。

## 3 快速训练检查（非论文结果）

```powershell
python wm_pusht8g.py train-policy --device cuda --steps 100 --batch-size 16 --hidden 128
python wm_pusht8g.py train-world --device cuda --steps 100 --batch-size 32 --hidden 128
```

检查 `runs/policy.pt`、`runs/world.pt` 是否出现；控制台是否持续输出损失。

## 4 正式进行第一轮研究实验

```powershell
python wm_pusht8g.py train-policy --device cuda --steps 30000 --batch-size 16 --hidden 256 --horizon 8 --log-every 500
python wm_pusht8g.py train-world --device cuda --steps 10000 --batch-size 64 --hidden 128 --log-every 500
```

`30,000 / 10,000` 是推荐的研究起点，不是论文官方配置，也不保证策略成功率。脚本默认覆盖上次同名检查点；实验前自行备份 checkpoint。

## 5 在真实 PushT 环境运行推理对照

```powershell
python wm_pusht8g.py eval --device cuda --mode policy --episodes 10 --seed 42
python wm_pusht8g.py eval --device cuda --mode sample4 --episodes 10 --seed 42
python wm_pusht8g.py eval --device cuda --mode world --episodes 10 --seed 42 --candidates 4
```

三种策略：
- `policy`：普通 Diffusion Policy-like 模型（1个候选）。
- `sample4`：生成4个候选但只取第1个，作为**相同采样批量**的对照。
- `world`：使用世界模型滚动预测候选未来状态，并选择目标代价较低的方案。

最后记录 success_rate 与 max_reward_mean。世界模型预测可能不准确，**`world` 可能表现比 `policy` 更差**，这属于应如实报告的实验结果。

## 6 资源约束和低显存处理

- 只用5维状态输入，**不是 RGB 视觉世界模型**。
- `batch-size` 从16开始，显存不足时降到8/4；`hidden` 从256降到128/64。
- Windows 数据加载 `num_workers=0`，避免 Windows multiprocessing 占用大量内存。
- `smoke` 使用内置的**合成数据**做张量与反向传播自检，**不用于真正训练、评价或报告论文效果**。
- Windows 11 的 `gym-pusht` 是CPU物理仿真，可以不用 Linux GPU 物理模拟器。

## 7 已验证范围（必须区分）

在本项目的开发容器里：已用 CPU 执行 `smoke` 以及合成数据的前向、反向、采样和短训练，均正常。由于开发容器无法联网下载真实 PushT 数据且没有 NVIDIA GPU，**没有对 Windows 11、下载流程、正式训练成功率与完整环境推理做端到端实测。**

遇到问题按 `check` → `smoke` → `download` → `inspect` → `train-policy` → `train-world` → `eval` 顺序排查。

## 8 引用与出处

- 原论文及官方数据：https://diffusion-policy.cs.columbia.edu/
- PushT Gym 环境：https://github.com/huggingface/gym-pusht
- Diffusion Policy 官方代码：https://github.com/real-stanford/diffusion_policy
- 未来可升级：TD-MPC2 https://github.com/nicklashansen/tdmpc2 （原官方环境偏 Linux，不纳入这套 Windows 基础方案）

## 9 仿真画面可视化（新增）

这个更新包包含 `visualize_pusht.py`，提供 **Windows 上弹出物理仿真窗口**，以及**保存动画 GIF** 两种方式。

**A. 无须训练就能看仿真（随机动作，主要测试画面）**

```powershell
python visualize_pusht.py --mode random --steps 150
```

这将弹出 Pygame 窗口。随机动作不等于学会推物块，也不一定能完成任务。

**B. 录制 GIF（不弹窗，GIF保存在 outputs 文件夹）**

```powershell
python visualize_pusht.py --mode random --steps 150 --record outputs/random.gif
```

**C. 先训练小脑，再可视化已训练的策略**

```powershell
python wm_pusht8g.py download
python wm_pusht8g.py train-policy --device cuda --steps 30000 --batch-size 16 --hidden 256
python visualize_pusht.py --mode policy --device cuda --steps 150
python visualize_pusht.py --mode policy --device cuda --steps 150 --record outputs/policy.gif
```

**D. 同时训练世界模型，观察预测辅助的策略（需要已有 runs/world.pt）**

```powershell
python wm_pusht8g.py train-world --device cuda --steps 10000 --batch-size 64 --hidden 128
python visualize_pusht.py --mode world --device cuda --steps 150 --record outputs/world.gif
```

`--mode policy` 与 `--mode world` 读取 `runs/policy.pt`（后者还需 `runs/world.pt`）。实时弹窗默认在运行结束时关闭；想反复查看可使用 `--record` 生成 GIF。由于每步进行扩散采样，**模型推理速度不一定等于真实时间**。

`--mode random` 是画面和环境测试；成功率等科研结果要以已训练策略的多随机种子 `eval` 输出为准。README中的安装流程未改变，新增两个通用 GIF 依赖 `imageio` 与 `pillow`。
