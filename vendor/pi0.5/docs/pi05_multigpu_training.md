# pi0.5 Multi-GPU Training Notes

这份本地 `/home/admin123/zjq/ws/pi0.5` 已按远端 `connect.bjb1.seetacloud.com:38407` 的源码快照替换。

说明：
- 这里只同步了源码、`pyproject.toml`、`uv.lock` 和文档。
- 没有同步远端 `.venv`、`.git`、缓存目录，所以现在它是“远端源码快照”，不是远端 git 工作树。
- `uv` 依赖现在以当前仓库里的 `pyproject.toml` 和 `uv.lock` 为准。

## 1. uv 依赖

仓库的安装方式按 [README.md](/home/admin123/zjq/ws/pi0.5/README.md:48)：

```bash
cd /home/admin123/zjq/ws/pi0.5
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

当前锁定的关键依赖可直接看 [pyproject.toml](/home/admin123/zjq/ws/pi0.5/pyproject.toml:1)：
- Python `>=3.11`
- `torch==2.7.1`
- `jax[cuda12]==0.5.3`
- `transformers==4.53.2`
- `lerobot` 通过 git revision 固定

如果 `uv sync` 冲突，仓库 README 的建议是先删本地环境再重建：

```bash
rm -rf .venv
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

## 2. 训练入口

PyTorch 多卡入口是 [scripts/train_pytorch.py](/home/admin123/zjq/ws/pi0.5/scripts/train_pytorch.py:1)。

标准单机多卡命令：

```bash
uv run torchrun --standalone --nnodes=1 --nproc_per_node=<GPU数> scripts/train_pytorch.py <config_name> --exp_name <run_name>
```

这个脚本是标准 DDP：
- 用 `torchrun`
- `WORLD_SIZE > 1` 时自动初始化 `torch.distributed`
- 有 GPU 时 backend 用 `nccl`

另外它的 `batch_size` 是“总 batch size”，不是单卡 batch size。代码在 [scripts/train_pytorch.py](/home/admin123/zjq/ws/pi0.5/scripts/train_pytorch.py:351) 明确写了：
- 总 batch size = `config.batch_size`
- 单卡 batch size = `config.batch_size / GPU数`

## 3. 通用默认训练参数

基础默认值定义在 [src/openpi/training/config.py](/home/admin123/zjq/ws/pi0.5/src/openpi/training/config.py:525) 和 [src/openpi/training/optimizer.py](/home/admin123/zjq/ws/pi0.5/src/openpi/training/optimizer.py:1)。

`TrainConfig` 默认值：
- `project_name="openpi"`
- `pytorch_training_precision="bfloat16"`
- `batch_size=32`
- `num_workers=2`
- `num_train_steps=30000`
- `log_interval=100`
- `save_interval=1000`
- `keep_period=5000`
- `seed=42`
- `wandb_enabled=True`
- `fsdp_devices=1`
- `ema_decay=0.99`

默认学习率调度 `CosineDecaySchedule`：
- `warmup_steps=1000`
- `peak_lr=2.5e-5`
- `decay_steps=30000`
- `decay_lr=2.5e-6`

默认优化器 `AdamW`：
- `b1=0.9`
- `b2=0.95`
- `eps=1e-8`
- `weight_decay=1e-10`
- `clip_gradient_norm=1.0`

## 4. 现成模板 config

当前直接借用仓库里的模板 config：

`pi05_hdf5_dscrew_v3_jax`

## 5. 当前直接可跑命令

先算 `norm_stats.json`：

```bash
cd /home/admin123/checkrecord
python hdf5_dscrew/compute_openpi_official_equiv_fast.py \
  --dataset-root /home/admin123/checkrecord/foldtask_value_dense_subset \
  --output-root /home/admin123/checkrecord/foldtask_value_dense_subset \
  --action-horizon 50 \
  --batch-size 4096
```

4 卡训练：

```bash
cd /home/admin123/zjq/ws/pi0.5
TOKENIZERS_PARALLELISM=false \
GIT_LFS_SKIP_SMUDGE=1 \
uv run torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  scripts/train_pytorch.py pi05_hdf5_dscrew_v3_jax \
  --exp_name foldtask_value_dense_subset_pi05_ddp \
  --pytorch_weight_path /home/admin123/models/pi05_base \
  --checkpoint_base_dir /home/admin123/checkrecord/openpi_ckpts \
  --batch_size 32 \
  --num_train_steps 20000 \
  --save_interval 10000 \
  --log_interval 100 \
  --wandb_enabled false \
  --data.repo_id /home/admin123/checkrecord/foldtask_value_dense_subset \
  --data.assets.assets_dir /home/admin123/checkrecord \
  --data.assets.asset_id foldtask_value_dense_subset \
  --data.video_backend pyav \
  --data.prompt_from_task true \
  --data.default_prompt ""
```

## 6. 说明

- 这条训练命令虽然借的是 `pi05_hdf5_dscrew_v3_jax`，但实际数据已经切成 `/home/admin123/checkrecord/foldtask_value_dense_subset`。
- `--batch_size 32` 是 4 卡总 batch 32，每卡 8。
- PyTorch base 权重走本地 `/home/admin123/models/pi05_base`。
- `norm_stats.json` 用的是你 `checkrecord` 里的 fast 脚本 [compute_openpi_official_equiv_fast.py](/home/admin123/checkrecord/hdf5_dscrew/compute_openpi_official_equiv_fast.py:1)。
- 输出目录规则按 [src/openpi/training/config.py](/home/admin123/zjq/ws/pi0.5/src/openpi/training/config.py:602)：

```text
<checkpoint_base_dir>/<config_name>/<exp_name>
```

这条命令的输出会落到：

```text
/home/admin123/checkrecord/openpi_ckpts/pi05_hdf5_dscrew_v3_jax/foldtask_value_dense_subset_pi05_ddp
```

继续训练就在训练命令最后加 `--resume`。
强制覆盖旧输出就在训练命令最后加 `--overwrite`。
