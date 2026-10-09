#!/bin/zsh
# agent_evolver 环境配置(复用项目 conf/ 中的内网 LLM 网关)
export LLM_BASE_URL="http://ai-llm-oneapi-v2.jryghq.com/v1"
export LLM_API_KEY="sk-kw9zkHPCRGBhycZA6jvpYKeeARe8EpItXP0nJoMnxV2tKD4q"
export LLM_MODEL="deepseek-v4-pro"
# 如网关有多模态模型可在此配置(处理纯图片流程图时才需要)
# export LLM_VISION_MODEL="qwen-vl-max"
