# Bundled admin123 PI0.5 source

- Source: `/home/admin123/zjq/ws/pi0.5`
- Source commit: `48926b94a2278fdfa58dd71ea5a10882f8912147`
- Snapshot date: `2026-08-22` (Asia/Shanghai)
- Purpose: travel with `atomic_latent_vla_layerwise` so remote training and
  inference do not import `/mnt/cunchu/yc/pi05/src`.
- Excluded: `.git`, `.venv`, caches and generated Python bytecode.
- Included: `src/openpi`, `packages/openpi-client`, project metadata, lockfile,
  scripts, examples and third-party source.

The source checkout was dirty when snapshotted. These paths contain the
working-tree versions, not merely the commit versions:

```text
M scripts/serve_policy.py
M src/openpi/models/model.py
M src/openpi/models/pi0.py
M src/openpi/policies/policy.py
M src/openpi/policies/policy_config.py
M src/openpi/serving/websocket_policy_server.py
M src/openpi/training/config.py
M src/openpi/transforms.py
?? docs/代码审查.md
?? scripts/analyze_action_delay_diff.py
?? scripts/manual_hdf5_pointcloud_rtc_client.py
?? scripts/manual_rtc_client.py
```

Key SHA-256 values:

```text
216fc090c01a67a6c0d4bfbbfb20889404307656916a28869f6a7fe4cb13d3eb  src/openpi/policies/marvin_policy.py
cdb0c104bd14eb84a4740cbac1b84b0a20bf8bd11a8f9cf603469cad2324e771  src/openpi/transforms.py
a1c8ca1d8d7781e4c596297c7ca91d51a83b51281ec5707ea375bbbb339ed71e  src/openpi/models/pi0.py
212cf56c7ec9249272665ce93c33525ed1e11c41b03e99c294088e4e0f981e43  src/openpi/models/model.py
```

Remote source order:

```bash
export OPENPI_ROOT="${RAM_REPO}/vendor/pi0.5"
export PYTHONPATH="${OPENPI_ROOT}/src:${OPENPI_ROOT}/packages/openpi-client/src:${RAM_REPO}/src"
```
