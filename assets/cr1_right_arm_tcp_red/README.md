# CR1 standalone right arm with red TCP

This bundle is derived from the current FK model:
`/home/admin123/cr1_recordclient/model/CR1_UPPER/urdf/CR1ARMR.urdf`.

Contents:

- `urdf/CR1ARMR_TCP_RED.urdf`: standalone seven-axis right arm.
- `meshes/*.STL`: only the seven meshes referenced by this URDF.
- `tcp`: a fixed TCP frame offset by `0.15 m` from `right_wrist_x_link`;
  its visual marker is a red `0.04 m` cube.

Render the bundled model from the repository root:

```bash
.venv/bin/python scripts/render_right_arm_urdf.py
```

Pass a recorded right-arm pose in URDF joint order when needed:

```bash
.venv/bin/python scripts/render_right_arm_urdf.py \
  --q 0.89 -0.28 -0.19 0.29 0 -1.26 -0.17 \
  --output outputs/visualizations/cr1_right_arm_tcp_red.png
```
