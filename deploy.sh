uv run python gr00t/eval/run_gr00t_server.py \
  --model-path outputs/franka_pick_cube_finetune/checkpoint-25000 \
  --embodiment-tag NEW_EMBODIMENT \
  --device cuda:0 \
  --host 0.0.0.0 \
  --port 5556

# 本机测试连通性：
# uv run python - <<'PY'
# from gr00t.policy.server_client import PolicyClient

# p = PolicyClient(host="127.0.0.1", port=5555, strict=False)
# print("ping:", p.ping())
# PY
