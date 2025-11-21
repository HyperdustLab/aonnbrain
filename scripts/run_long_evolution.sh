#!/bin/bash
# 长时间演化实验运行脚本
# 默认运行 15000 步（约 8 小时）

set -e  # 遇到错误立即退出

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# 进入项目根目录
cd "$PROJECT_ROOT"

# 默认参数
STEPS=${1:-15000}
SAVE_INTERVAL=${2:-100}
OUTPUT_DIR="data/long_evolution_$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$OUTPUT_DIR/evolution.log"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/checkpoints"

echo "=========================================="
echo "AONN 长时间演化实验"
echo "=========================================="
echo "开始时间: $(date)"
echo "演化步数: $STEPS"
echo "保存间隔: $SAVE_INTERVAL 步"
echo "输出目录: $OUTPUT_DIR"
echo "日志文件: $LOG_FILE"
echo "=========================================="
echo ""

# 检查虚拟环境
if [ -d "venv" ]; then
    echo "激活虚拟环境..."
    source venv/bin/activate
elif [ -d ".venv" ]; then
    echo "激活虚拟环境..."
    source .venv/bin/activate
fi

# 运行实验
echo "开始运行实验..."
python scripts/run_general_ai_experiment.py \
    --steps "$STEPS" \
    --save-interval "$SAVE_INTERVAL" \
    --checkpoint-dir "$OUTPUT_DIR/checkpoints" \
    --output "$OUTPUT_DIR/experiment_results.json" \
    2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "=========================================="
echo "实验完成"
echo "结束时间: $(date)"
echo "退出代码: $EXIT_CODE"
echo "结果保存在: $OUTPUT_DIR"
echo "=========================================="

exit $EXIT_CODE

