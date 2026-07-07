# FastWAM 训练文档（XPolicyLab 适配版）

> 上游项目：[Fast-WAM: Do World Action Models Need Test-time Future Imagination?](https://arxiv.org/abs/2603.16666)
> GitHub: https://github.com/yuantianyuan01/FastWAM
> 上游训练入口：`FastWAM/scripts/train.py` + `FastWAM/scripts/train_zero1.sh`
> XPolicyLab 入口：`XPolicyLab/policy/FastWAM/train.sh`

本文聚焦“**使用 LeRobot 格式数据训练 FastWAM**”所需的步骤、配置含义与踩坑点。
完成 Step 1–Step 4 后，单卡或多卡均可直接训练。

---

## 0. 上游与本仓库的关系

```
XPolicyLab/policy/FastWAM/
├── eval.sh / setup_eval_policy_server.sh / setup_eval_env_client.sh
├── train.sh                              # 本仓库的训练入口（封装上游 train_zero1.sh）
├── deploy.yml / model.py / __init__.py   # XPolicyLab 推理适配
├── INSTALLATION.md / TRAINING.md
├── data/<dataset_id>/                    # 训练用 LeRobot v2.1 + dataset_stats（需自行准备）
│   ├── lerobot/{data,videos,meta}/
│   └── dataset_stats.json
├── checkpoints/                          # 训练输出根目录
│   ├── ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt  # Step 2 产物
│   └── <ckpt_setting>/                   # 每次训练运行的 Hydra `output_dir`
└── FastWAM/                              # 上游源码（保持纯净，不要直接改）
    ├── scripts/{train.py,train_zero1.sh,preprocess_action_dit_backbone.py,precompute_text_embeds.py}
    ├── configs/{data,model,task}/        # 上游 Hydra 配置
    ├── src/fastwam/                      # 上游核心代码
    ├── data/text_embeds_cache/xpolicylab/<dataset_id>/  # T5 cache，由 precompute_text_embeds.py 写入
    └── checkpoints/                      # 上游下载的 Wan2.2 等基础模型
```

为什么 `train.sh` 强制走上游 `scripts/train_zero1.sh` 而不是直接调 `train.py`？
1. `train_zero1.sh` 已经包好 `accelerate launch + DeepSpeed ZeRO-1` 配置；
2. 多机训练时 `RUN_ID` 同步逻辑全在那个脚本里；
3. 我们只在外层补 XPolicyLab 需要的 Hydra overrides，不污染上游配置。

---

## 1. 训练前需要准备什么

`train.sh` 启动时会检查以下路径，缺任一项直接退出：

| 产物 | 默认路径 | 如何准备 |
|---|---|---|
| LeRobot v2.1 数据集 | `data/<dataset_id>/lerobot/` | 使用上游数据流程或已有 LeRobot 数据集；`dataset_id` 默认为 `<bench>-<ckpt>-<env>-<action>`，可用 `FASTWAM_DATASET_ID` 覆盖 |
| Normalization 统计 | `data/<dataset_id>/dataset_stats.json` | 与 LeRobot 数据集配套；训练首次启动时上游也可能二次计算 |
| ActionDiT backbone | `FastWAM/checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt` | Step 2 预处理 |
| T5 prompt 缓存 | `FastWAM/data/text_embeds_cache/xpolicylab/<dataset_id>/*.pt` | Step 3 用上游 `precompute_text_embeds.py` |

`train.sh` 会通过 Hydra override 把 `action_dim`、数据集路径、text cache 与 checkpoint 输出目录对齐到 XPolicyLab 约定。

---

## 2. Step 1：环境

参见 `INSTALLATION.md`。简单流程：

```bash
cd XPolicyLab/policy/FastWAM
bash install.sh                       # 创建并装填 conda env `fastwam`
conda activate fastwam
```

---

## 3. Step 2：准备 Wan2.2 基础模型 + ActionDiT backbone

这是**只跑一次**的预处理：把 Wan2.2 的视频 DiT 权重按 FastWAM 的形状插值好，保存到本地。

```bash
conda activate fastwam
cd XPolicyLab/policy/FastWAM/FastWAM
mkdir -p checkpoints
export DIFFSYNTH_MODEL_BASE_PATH="$(pwd)/checkpoints"

python scripts/preprocess_action_dit_backbone.py \
  --model-config configs/model/fastwam.yaml \
  --output checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt \
  --device cuda \
  --dtype bfloat16
```

产物：`XPolicyLab/policy/FastWAM/FastWAM/checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt`
`train.sh` 启动时会显式检查这个文件，不存在直接报错并打印重跑命令。

> `DIFFSYNTH_MODEL_BASE_PATH` 在我们的 `train.sh` / `setup_eval_policy_server.sh` 里**也会再次 export**，所以
> 训练 / 推理时 Wan2.2 主权重会从 `<policy>/FastWAM/checkpoints/` 下取。

---

## 4. Step 3：准备 LeRobot 数据与 T5 cache

### 4.1 数据集布局

默认 `dataset_id = <bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>`。训练读取：

```
XPolicyLab/policy/FastWAM/data/<dataset_id>/
├── dataset_stats.json                     # FastWAM Normalizer 直接消费
└── lerobot/
    ├── meta/{info.json, tasks.jsonl, episodes.jsonl, episodes_stats.jsonl}
    ├── data/chunk-000/episode_000000.parquet ...
    └── videos/chunk-000/observation.images.{cam_high,cam_left_wrist,cam_right_wrist}/episode_000000.mp4 ...
```

若输出目录名不同，训练前设置 `FASTWAM_DATASET_ID=<your_dataset_id>`。

### 4.2 预计算 T5 text embedding

在 FastWAM 环境中，对准备好的 LeRobot 数据集运行上游脚本：

```bash
conda activate fastwam
cd XPolicyLab/policy/FastWAM/FastWAM

dataset_dir="../data/<dataset_id>/lerobot"
text_cache_dir="data/text_embeds_cache/xpolicylab/<dataset_id>"

python scripts/precompute_text_embeds.py \
  task=robotwin_uncond_3cam_384_1e-4 \
  "data.train.dataset_dirs=[${dataset_dir}]" \
  "data.val.dataset_dirs=[${dataset_dir}]" \
  "data.train.text_embedding_cache_dir=${text_cache_dir}" \
  "data.val.text_embedding_cache_dir=${text_cache_dir}"
```

T5 缓存键是“**Wan-AI/Wan2.2-TI2V-5B + DEFAULT_PROMPT + context_len=128**”的 sha256，
所以同一 `dataset_id` 不同 seed/batch_size 都可以复用同一份缓存；改任务（改 instruction）才需要重算。

---

## 5. Step 4：训练

### 5.1 入口

```bash
conda activate fastwam
bash XPolicyLab/policy/FastWAM/train.sh \
    <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id> [num_gpus]
```

例：

```bash
# 单卡，单任务
bash XPolicyLab/policy/FastWAM/train.sh \
    RoboDojo test_data arx_x5 joint 0 0
```

```bash
# 多卡，单任务（8 卡）
bash XPolicyLab/policy/FastWAM/train.sh \
    RoboDojo test_data arx_x5 joint 0 0,1,2,3,4,5,6,7
```

```bash
# 多卡 + cotrain（前提是 data/<dataset_id>/ 已准备好合并后的 LeRobot 数据集）
bash XPolicyLab/policy/FastWAM/train.sh \
    RoboDojo cotrain arx_x5 joint 0 0,1,2,3,4,5,6,7
```

checkpoint 输出目录为 `checkpoints/<bench>-<ckpt>-<env>-<action>-<seed>`，
这个目录名就是 eval 时要传的 `ckpt_name`。

### 5.2 `train.sh` 内部做了什么

1. `get_action_dim.sh` 算出本次实际 `action_dim`（沿用 XPolicyLab robot info 的 `arm_dim + ee_dim`）；
2. 根据 `FASTWAM_DATASET_ID`（或默认 data_key）解析出 lerobot 目录与 dataset_stats；
3. 校验 LeRobot 数据集、ActionDiT backbone 和 T5 cache **必须存在**，不存在直接退出；
4. cd 到 `FastWAM/`，调上游 `scripts/train_zero1.sh` 并传入下列 Hydra overrides：

```
task=robotwin_uncond_3cam_384_1e-4
seed=${train_seed}
batch_size=${batch_size}                         # 默认 8，可 FASTWAM_BATCH_SIZE 覆盖
gradient_accumulation_steps=${gradient_accumulation_steps}  # 默认 1
num_workers=${num_workers}                       # 默认 8

# 数据集与 normalization 统计
data.train.dataset_dirs=[<policy>/data/<dataset_id>/lerobot]
data.val.dataset_dirs=[<policy>/data/<dataset_id>/lerobot]
data.train.text_embedding_cache_dir=<policy>/FastWAM/data/text_embeds_cache/xpolicylab/<dataset_id>
data.val.text_embedding_cache_dir=...（同上）
data.train.pretrained_norm_stats=<policy>/data/<dataset_id>/dataset_stats.json
data.val.pretrained_norm_stats=...（同上）

# action/state 维度对齐
data.train.shape_meta.action.0.raw_shape=${action_dim}
data.train.shape_meta.action.0.shape=${action_dim}
data.train.shape_meta.state.0.raw_shape=${action_dim}
data.train.shape_meta.state.0.shape=${action_dim}
data.val.shape_meta.<...>=${action_dim}
data.train.processor.action_output_dim=${action_dim}
data.train.processor.proprio_output_dim=${action_dim}
data.val.processor.<...>=${action_dim}

# 训练输出根目录
output_dir=<policy>/checkpoints/<ckpt_setting>
```

### 5.3 可调环境变量

| 环境变量 | 默认 | 含义 |
|---|---|---|
| `FASTWAM_DATASET_ID` | `<bench>-<ckpt>-<env>-<action>` | 选取已准备数据集（自定义目录名时指定） |
| `FASTWAM_CKPT_SETTING` | `<bench>-<ckpt>-<env>-<action>-<seed>` | 训练输出子目录名（写到 `checkpoints/<ckpt_setting>/`），即 eval 的 `ckpt_name` |
| `FASTWAM_BATCH_SIZE` | `8` | 单卡 batch size。OOM 就调小（如 4/2/1） |
| `FASTWAM_GRADIENT_ACCUMULATION_STEPS` | `1` | 等效 batch = batch × accumulation × world_size |
| `FASTWAM_NUM_WORKERS` | `8` | DataLoader workers |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | 减少显存碎片；可覆盖 |

可调 Hydra overrides（追加到 `train.sh` 末尾即可，因为我们没有锁死 `train_zero1.sh` 的 `EXTRA_ARGS`，要透传需手动改 `train.sh`；
更常见的做法是直接 `export FASTWAM_BATCH_SIZE=...` 等环境变量，或临时复制一份 `train.sh` 修改 `train_common=()`）。

如果要改 epoch / save_every / eval_every 等，请直接编辑：
`XPolicyLab/policy/FastWAM/FastWAM/configs/task/robotwin_uncond_3cam_384_1e-4.yaml`
（5 个 epoch 是 RoboTwin 完整数据集的设定，对小样本远远不够，**实际训练时务必显著上调** `num_epochs`，
或者改成 `max_steps`，详见 §6）。

### 5.4 训练输出

```
XPolicyLab/policy/FastWAM/checkpoints/<ckpt_setting>/
├── checkpoints/                           # 上游 Trainer 写的所有 step 子目录
│   ├── step_<N>/
│   │   ├── weights/                       # safetensors 分片
│   │   ├── optimizer/
│   │   └── scheduler/ ...
│   └── weights/                           # 软链 / 副本最新 step（视 ZeRO-1 配置）
├── dataset_stats.json                     # 训练首次启动时由 BaseLerobotDataset 二次计算并落盘
└── train_log.txt / accelerate logs ...
```

eval 端的 `setup_eval_policy_server.sh` 会从 `checkpoints/<ckpt_setting>/checkpoints/weights/step_*.pt` 取最大编号那份；
如果 ZeRO-1 写法是分片 safetensors（而不是单文件 `step_*.pt`），请检查上游 `save_every` / Trainer 输出格式
并在 `setup_eval_policy_server.sh` 里把 glob 改成相应的 pattern（当前默认 `step_*.pt`，必要时可设
`FASTWAM_CHECKPOINT_PATH` 直接指定）。

---

## 6. 关键技术细节 / 容易踩坑的点

### 6.1 action/state 归一化模式

`configs/data/robotwin.yaml` 默认：
```yaml
processor:
  use_stepwise_action_norm: False
  norm_default_mode: "z-score"
  action_state_transforms: null
```

- `z-score` 用 `global_mean/global_std`，输出近似 N(0,1) 后 clamp 到 `[-5, 5]`。
- 训练与推理用的是**同一份** `dataset_stats.json`，所以分布不会因为 dataset 重算导致漂移。
- 我们没有改这个 default。如果观察到 gripper（双峰）维度被 z-score 拍扁导致动作不响应，可在 `train.sh`
  里临时追加 `data.train.processor.norm_default_mode=q01/q99 data.val.processor.norm_default_mode=q01/q99`。

### 6.2 `task_name` 与 `ckpt_name` 的区别

- 训练时 `<ckpt_name>` 参与拼 dataset 文件夹名与 checkpoint 目录名。
- 推理时 `eval.sh` 的第 2/3 个参数分别是 `task_name`（仿真任务，决定环境）与 `ckpt_name`
  （`checkpoints/` 下完整 run 目录名，即训练输出的 `<bench>-<ckpt>-<env>-<action>-<seed>`），
  这样可以用同一个 cotrain checkpoint 评估不同的下游任务。

### 6.3 action_dim 与 `ee` 表示

- `train.sh` 把 `action_output_dim / proprio_output_dim` 改成 `get_action_dim.sh` 的结果。
- 参考 `aa4d9c1` 的 FastWAM 路径，FastWAM 本地不额外解释或转换 `ee_pose`；训练和评测都直接依赖
  XPolicyLab 的 `pack_robot_state` / `unpack_robot_state` 约定。
- 评测时必须传入与 checkpoint 训练时一致的 `action_type`。不要仅凭 FastWAM 本地代码推断 `ee`
  是 16 维或 14 维；以数据集的 pack 结果和训练 checkpoint 的 processor 形状为准。

### 6.4 epoch / steps

- 上游 RoboTwin 用 `num_epochs=5` 但数据集是几十万 episode；
- 小样本（3–几十 episode）建议直接走 `max_steps`：
  在 `configs/task/robotwin_uncond_3cam_384_1e-4.yaml` 把 `num_epochs: 5` 改成 `num_epochs: null` 并设
  `max_steps: 20000` 之类；
- `save_every` 当前 2500、`eval_every` 500，对小数据集偏大，可改 `save_every: 500 eval_every: 100`。

### 6.5 GPU 资源

**Wan2.2-TI2V-5B 单卡训不起来——不是 batch_size 问题，是优化器状态问题。**

AdamW fp32 优化器状态（master + m + v）= 5B × 12 byte ≈ **60 GB**；ZeRO-1 只在多 rank 间分片优化器状态，
单进程 = 单分片 = 60 GB 完整保留。加上 bf16 权重 10 GB、bf16 梯度 10 GB、33 帧×3 路×384×320 视频激活 ~15 GB，
合计 **~95 GB**，单张 A800 80G 必爆 OOM（典型崩在 `deepspeed/runtime/zero/stage_1_and_2.py:step()`）。

把 `FASTWAM_BATCH_SIZE` 调到 1 也救不了：激活只是 ~15 GB 那一项的一个零头，瓶颈在固定的 60 GB 优化器状态。

可行配置（按推荐顺序）：

| 卡数 | 启动命令第 7 个参数 | 每卡优化器状态 | 备注 |
|---|---|---|---|
| 8 卡 80G | `0,1,2,3,4,5,6,7` | 60 / 8 ≈ 7.5 GB | **推荐**，对齐上游 LIBERO 配置 |
| 4 卡 80G | `0,1,2,3` | 15 GB | 紧但能跑，`batch_size=2` 起步 |
| 2 卡 80G | `0,1` | 30 GB | 几乎不行，要切 ZeRO-2 + offload |
| 1 卡 80G | `0` | 60 GB | **不可行**，需改 DS 配置加 CPU offload |

`train.sh` 第 7 个位置参数支持逗号分隔的 GPU 列表，会自动把卡数传给 `train_zero1.sh` → `accelerate launch --num_processes N`。
全局 batch = `batch_size × gradient_accumulation_steps × num_gpus`，
8 卡 + `FASTWAM_BATCH_SIZE=4` 全局就是 32，已经接近 RoboTwin 上游 batch=64 的一半。

**单卡兜底**（确实只有 1 卡可用时）需要改 `FastWAM/scripts/ds_configs/ds_zero1_config.json`：

```json
"zero_optimization": {
    "stage": 2,
    "offload_optimizer": { "device": "cpu", "pin_memory": true },
    ...
}
```

把优化器状态甩到 CPU RAM（需要 ≥ 80 GB 空 RAM），训练会慢 2–4×，但能塞下。**不推荐**作为常态。

### 6.6 显存 / 性能开关

- `model.mot_checkpoint_mixed_attn: true`（来自 `fastwam.yaml`）配 `use_gradient_checkpointing=true`，能省一半 attn 激活；
  上游 RoboTwin task yaml 反而把它设成 `false` 以提速，但需要更多显存。我们暂时沿用 task yaml 的 `false`。
- 加 `bf16` 已经是默认（`mixed_precision: bf16`）。

### 6.7 多任务训练时 T5 prompt 编码

- `precompute_text_embeds.py` 的缓存键基于 `DEFAULT_PROMPT.format(task=<unique instruction>)` 的 sha256。
- cotrain 多任务时，`tasks.jsonl` 会有多条 instruction，每条都会被独立编码并缓存。
- 训练 / 推理时 `RobotVideoDataset` 按 `task_index → tasks.jsonl[idx]` 取 instruction，再去 cache 取嵌入；
  cache miss 直接 KeyError，请确保 `text_embedding_cache_dir` 与 `dataset_dirs` 一一对应（`train.sh` 已自动对齐）。

### 6.8 视频解码

- LeRobot 内部用 `torchcodec==0.5`，它依赖 conda env 里的 ffmpeg（`libavutil.so.58`），
  我们的 `install.sh` 已经处理；
- DataLoader `num_workers>0` 时上游有 deque-based 视频读取，注意 `_query_videos` 文档里写过：
  **不要再在主进程开第二个 num_workers=0 的 DataLoader**，否则 video reader 句柄共享会段错误。

---

## 7. Step 5：评估

```bash
conda activate fastwam
bash XPolicyLab/policy/FastWAM/eval.sh \
    <bench_name> <task_name> <ckpt_name> <env_cfg_type> \
    <action_type> <seed> <policy_gpu_id> <env_gpu_id> \
    <policy_conda_env> <eval_env_conda_env>
```

例（统一 10 参数，`ckpt_name` 为 `checkpoints/` 下完整 run 目录名）：

```bash
bash XPolicyLab/policy/FastWAM/eval.sh \
    RoboDojo test_data RoboDojo-test_data-arx_x5-joint-0 arx_x5 \
    joint 0 0 0 \
    fastwam XPolicyLab
```

- `task_name` 与 `ckpt_name` 可不同：用 `cotrain` 的 checkpoint 跑各子任务时只需把
  `ckpt_name=RoboDojo-cotrain-arx_x5-joint-<seed>`，换 `task_name` 即可。
- `eval.sh` 自己分配空闲端口、起 server 进程、等就绪、再起 client；server 异常退出会立即报错。
- 详见 `INSTALLATION.md` 的 Step 6 与 `model.py`。

---

## 8. 已知开放问题 / 需用户确认

1. **`num_epochs / max_steps` 默认值**：上游 RoboTwin 是 `num_epochs=5`，对小样本数据集严重欠拟合，是否需要在 `train.sh` 提供一个 XPolicyLab 默认覆盖（例如 `num_epochs=null max_steps=10000`）？
2. **`norm_default_mode`**：要不要在 train.sh 把默认改成 `q01/q99`？（diffusion policy 实践更常见）
3. **单卡默认 batch size**：当前 `FASTWAM_BATCH_SIZE=8` 仍可能单卡 OOM。要不要把默认下调到 4（或配 accumulation），多卡再手动覆盖回去？
4. **`save_every / eval_every` 与 `num_epochs`**：是否需要按数据集规模自动缩放？例如 `save_every = max(100, total_steps // 20)`。

把答案告诉我，我会按优先级继续把这些默认值落到 `train.sh` / task yaml 里。
