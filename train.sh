#!/usr/bin/env bash
set -euo pipefail


# 测试 opd baseline
bash train_grpo.sh

# 测试 新的rcod配置
bash train_rcod.sh
