# InternVLA_A1_5

**Contributor:** RoboDojo Team | **Paper:** InternVLA-A1.5 technical report | **arXiv:** TBD

`InternVLA_A1_5` is the XPolicyLab/RoboDojo inference adapter for the InternVLA-A1.5 policy. It wraps the `InternVLAA15Policy` (flow-matching action expert + Qwen3.5 VLM) into the `ModelTemplate` interface so it can be served by the XPolicyLab policy server and evaluated against RoboDojo simulation tasks.

The adapter only covers inference. Its runtime source is included under `internvla_a1_5/`, matching the self-contained layout of `InternVLA_A1`. Training scripts are not included; use the upstream InternVLA-A1.5 training recipe to produce checkpoints.

<details>
<summary>File Structure</summary>

| Path | Purpose |
|---|---|
| `README.md` | Supplemental documentation. |
| `install.sh` | Creates/updates the policy environment and installs the included runtime. |
| `eval.sh` | Runs a same-machine policy server plus RoboDojo environment client evaluation. |
| `setup_eval_policy_server.sh` | Starts only the policy server for distributed/debug evaluation. |
| `setup_eval_env_client.sh` | Starts only the RoboDojo environment client and connects to a policy server. |
| `deploy.py` | Policy wrapper used by the XPolicyLab model server (`eval_one_episode` / `eval_one_episode_batch`). |
| `model.py` | Model adapter loaded by `deploy.py` or the policy server. |
| `deploy.yml` | Runtime configuration and default checkpoint/model parameters. |
| `internvla_a1_5/` | InternVLA-A1.5/LeRobot runtime source used by the adapter. |

</details>

## Installation

The adapter no longer depends on a separate local `lerobot_lab` checkout.

```bash
cd XPolicyLab/policy/InternVLA_A1_5
bash install.sh
conda activate internvla_a1_5
```

The upstream details are in `internvla_a1_5/tutorials/installation.md`. For
development, `LEROBOT_SRC_PATH` can still override the included source directory.

## Differences from InternVLA_A1

| Aspect | A1 | A1.5 |
|---|---|---|
| Image input | History frame pairs (current + past interval) | Single-frame 3 views (cam_high / cam_left_wrist / cam_right_wrist) |
| Chat processor | `Qwen3_VLProcessorTransformFn` | `InternVLAA15ChatProcessorTransformFn` (eval mode) |
| State/Action | Native dim, no pad/reorder | Padded to 32 dims + reordered via `get_schema(stats_key)` |
| Action recovery | Direct unnormalize | `invert_action_reorder` (uses `schema.action_reorder`) then unnormalize |
| Env vars | `COSMOS_PATH` / `QWEN3_2B_PATH` | `HF_HOME` for offline Qwen3.5 weights (`config.vlm_model_name_or_path`) |

The inference flow mirrors `evaluation/RoboTwin/inference.py`: build a single-frame sample, run `ResizeImagesWithPadFn -> RemapImageKeyTransformFn -> NormalizeTransformFn -> InternVLAA15ChatProcessorTransformFn -> PadStateAndActionTransformFn -> ReorderStateActionTransform`, call `predict_action_chunk`, then invert the reorder back to the robot's native action layout, unnormalize, and (in delta mode) add the current state.

## Deployment and Evaluation

What it does: serves the policy through XPolicyLab and connects it to a RoboDojo evaluation client.

Parameters used by `eval.sh`:

| Parameter | Description |
|---|---|
| `bench_name` | Benchmark or dataset family, usually `RoboDojo`. |
| `task_name` | RoboDojo simulation task to evaluate, for example `stack_bowls`. |
| `ckpt_name` | Checkpoint/run directory name, usually under `checkpoints/`. |
| `env_cfg_type` | Robot/environment configuration, for example `arx_x5`. |
| `action_type` | Action representation, for example `joint`. |
| `seed` | Evaluation seed. |
| `policy_gpu_id` | GPU used by the policy server. |
| `env_gpu_id` | GPU used by the RoboDojo simulation client. |
| `policy_conda_env` | Conda environment for the policy server. |
| `eval_env_conda_env` | Conda environment for RoboDojo simulation/client. |

```bash
cd XPolicyLab/policy/InternVLA_A1_5
# Template: run same-machine policy server and RoboDojo environment client.
bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <policy_gpu_id> <env_gpu_id> <policy_conda_env> <eval_env_conda_env>

# Example: evaluate an exported local checkpoint on stack_bowls.
INTERNVLA_CKPT_PATH=/path/to/pretrained_model \
  bash eval.sh RoboDojo stack_bowls internvla_a1_5 arx_x5 joint 0 0 0 <policy_conda_env> <eval_env_conda_env>

# Using the Hugging Face repo
INTERNVLA_CKPT_PATH=hxma/internvla_a15_robodojo_60k \
  bash eval.sh RoboDojo stack_bowls internvla_a1_5 arx_x5 joint 0 0 0 <policy_conda_env> <eval_env_conda_env>
```

For split server/client evaluation, use `setup_eval_policy_server.sh` and `setup_eval_env_client.sh` separately (same CLI as `InternVLA_A1`):

```bash
# Terminal 1 (policy machine): start the policy server.
bash setup_eval_policy_server.sh \
  <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <policy_gpu_id> <policy_conda_env> <policy_server_port> <policy_server_host>

# Terminal 2 (env machine): connect RoboDojo to the policy server.
bash setup_eval_env_client.sh \
  <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <env_gpu_id> <eval_env_conda_env> <additional_info> \
  <policy_server_port> <policy_server_ip>
```

Set `EVAL_ENV_TYPE=debug` for offline shape/IO checks; leave it unset or set `EVAL_ENV_TYPE=sim` for RoboDojo simulation.

## Important Parameters

Policy-specific `deploy.yml` keys worth checking before evaluation:

| Key | Notes |
|---|---|
| `stats_key` | Robot type used for both `get_schema(stats_key)` and `stats.json` lookup. Must match training (e.g. `aloha`). |
| `inference_backend` | `standard` or `optimized` (optimized requires `action_loss_only=True`). |
| `infer_horizon` | Number of action steps executed per inference call (default 20). |
| `action_mode` | `delta` (add to current state) or `abs` (absolute). |
| `tokenize_state` | Whether the chat processor tokenizes the state into the prompt. |
| `max_state_dim` / `max_action_dim` | Padding dims (default 32, matches A1.5 config). |
| `resize_size` | Image resize target (default 224). |
| `checkpoint_num` | Step number used to select a checkpoint under the run directory. |

Frequently used environment variables:

| Variable | Notes |
|---|---|
| `HF_HOME` | Local HuggingFace cache; needed for offline Qwen3.5 processor/VLM weights. |
| `CUDA_VISIBLE_DEVICES` | GPU selection, set by the eval scripts. |
| `INTERNVLA_CKPT_PATH` | Exported checkpoint directory or Hugging Face model repo id. |
| `LEROBOT_SRC_PATH` | Optional development override for the included `internvla_a1_5/src`. |

## Notes

- The checkpoint's `config.json` must have `"type": "internvla_a1_5"` so `PreTrainedConfig.from_pretrained` resolves to `InternVLAA15Config`.
- The uploaded Hugging Face model repo must contain `model.safetensors`, `config.json`, and `stats.json` at its root.
- `stats_key` is shared between the dataset schema registry (`get_schema`) and the checkpoint's `stats.json`; they must refer to the same robot layout.
- The adapter supports `action_type='joint'` only, matching the A1 adapter.
