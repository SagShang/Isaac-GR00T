# Franka Pick Cube

Convert the raw recordings to GR00T LeRobot format with joint-angle control:

```bash
python scripts/lerobot_conversion/convert_franka_pick_cube_joint.py \
  --input-dir datasets/franka_pick_and_place_cube \
  --output-dir datasets/franka_pick_and_place_cube_lerobot \
  --target-fps 20 \
  --force
```

Generate normalization statistics:

```bash
python -m gr00t.data.stats \
  --dataset-path datasets/franka_pick_and_place_cube_lerobot \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path examples/FRANKA_PICK_CUBE/franka_pick_cube_config.py
```

Fine-tune with the converted dataset:

```bash
uv run bash examples/finetune.sh \
  --base-model-path /data/wentao/checkpoints/GR00T-N1.7-3B \
  --dataset-path datasets/franka_pick_and_place_cube_lerobot \
  --modality-config-path examples/FRANKA_PICK_CUBE/franka_pick_cube_config.py \
  --embodiment-tag NEW_EMBODIMENT \
  --output-dir outputs/franka_pick_cube_finetune
```

The converted dataset stores absolute values in parquet:

- `observation.state`: absolute `joint_position` + raw `gripper_position`
- `action`: absolute `joint_position` + binary `gripper_action`

The dataset stores the same-frame absolute `joint_position` in both state and action,
with no time shift. The config trains all actions as `ABSOLUTE`: joint actions are
absolute joint-angle targets, while gripper actions are the recorded binary
`gripper_action` signal. The gripper state uses continuous `gripper_position`.
