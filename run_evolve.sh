#!/bin/zsh
# 一键 MCTS 演化(online 版:Go Agent 工程,UI 18090)
# 需求增量 R1~R3 已由 spec_overrides 施加;固定接口与编排话术受 --protected/--protected-literals 双层保护
cd "$(dirname "$0")"
source env.sh
# 可选:限制 LLM 调用预算并记录调用轨迹(便于成本控制与复现)
export LLM_MAX_CALLS="${LLM_MAX_CALLS:-0}"          # 0=不限
export LLM_TRACE="${LLM_TRACE:-evolve_reports/llm_trace.jsonl}"
exec python3 -u main.py evolve \
  --agent-url http://127.0.0.1:18090 \
  --repo .. \
  --iterations "${ITER:-12}" --rollouts "${ROLLOUTS:-5}" --target 0.95 \
  --seed "${SEED:-42}" \
  --sources "internal/**/*.go,conf/**/*.yaml" \
  --protected "internal/tools/route/*.go,UI-test/*,conf/deployment/*" \
  --protected-literals "${PROTECTED_LITERALS:-师傅，您当前的违规原因为|本次将沿用上次的申诉类型进行申诉}" \
  --build-cmd "go build ./..." \
  --restart-cmd "./scripts/restart_online.sh"

