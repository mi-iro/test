#!/bin/bash

# 原始模板文件路径
TEMPLATE_FILE="configs/mmlong_template.yaml"
# Prompt文件所在的基础路径
PROMPT_BASE_PATH="/mnt/shared-storage-user/mineru2-shared/wangzhengren/test/prompts"
# 输出目录的基础前缀
OUTPUT_BASE_DIR="outputs/0217_mmlong_gemini25"

# 要测试的Prompt列表
PROMPTS=("mmlong0.txt" "mmlong4.txt" "mmlong21.txt")

# 检查模板文件是否存在
if [ ! -f "$TEMPLATE_FILE" ]; then
    echo "Error: Template file '$TEMPLATE_FILE' not found!"
    exit 1
fi

# 创建configs目录存放临时配置文件
mkdir -p configs/tmp

# 循环遍历每个 Prompt
for PROMPT_FILE in "${PROMPTS[@]}"; do
    # 提取Prompt名称用于命名 (如 prompt0)
    PROMPT_NAME=$(basename "$PROMPT_FILE" .txt)
    
    # 构建完整的Prompt路径
    FULL_PROMPT_PATH="${PROMPT_BASE_PATH}/${PROMPT_FILE}"

    # 循环遍历 6 种参数设置
    for SETTING_ID in {1..6}; do
        echo "----------------------------------------------------------------"
        echo "Running: Prompt=[$PROMPT_NAME] | Setting=[$SETTING_ID]"
        
        # 定义临时配置文件名
        CONFIG_FILE="configs/tmp/${PROMPT_NAME}_set${SETTING_ID}.yaml"
        
        # 定义输出目录
        OUTPUT_DIR="${OUTPUT_BASE_DIR}_${PROMPT_NAME}_set${SETTING_ID}"

        # 复制模板文件
        cp "$TEMPLATE_FILE" "$CONFIG_FILE"

        # 1. 修改 rag_system_prompt
        # 使用 | 作为分隔符以避免路径中的 / 冲突
        sed -i "s|rag_system_prompt: .*|rag_system_prompt: ${FULL_PROMPT_PATH}|g" "$CONFIG_FILE"

        # 2. 修改 output_dir
        sed -i "s|output_dir: .*|output_dir: ${OUTPUT_DIR}|g" "$CONFIG_FILE"

        # 3. 根据设置ID修改参数开关
        # 为了确保配置准确，我们先将所有相关开关重置为 false 或默认状态，然后根据需求开启
        # 注意：这里直接替换对应的行。假设模板中这些key是存在的且每行一个。
        
        case $SETTING_ID in
            1)
                # Setting 1: use_page: true, use_page_ocr: false, use_crop: false, use_ocr: false
                sed -i 's/use_page: .*/use_page: true/g' "$CONFIG_FILE"
                sed -i 's/use_page_ocr: .*/use_page_ocr: false/g' "$CONFIG_FILE"
                sed -i 's/use_crop: .*/use_crop: false/g' "$CONFIG_FILE"
                sed -i 's/use_ocr: .*/use_ocr: false/g' "$CONFIG_FILE"
                ;;
            2)
                # Setting 2: use_page: true, use_page_ocr: true, use_crop: false, use_ocr: false
                sed -i 's/use_page: .*/use_page: true/g' "$CONFIG_FILE"
                sed -i 's/use_page_ocr: .*/use_page_ocr: true/g' "$CONFIG_FILE"
                sed -i 's/use_crop: .*/use_crop: false/g' "$CONFIG_FILE"
                sed -i 's/use_ocr: .*/use_ocr: false/g' "$CONFIG_FILE"
                ;;
            3)
                # Setting 3: use_page: true, use_crop: true, use_ocr: false
                # (use_page_ocr 保持模板默认或设为false，此处显式设为false以防干扰)
                sed -i 's/use_page: .*/use_page: true/g' "$CONFIG_FILE"
                sed -i 's/use_page_ocr: .*/use_page_ocr: false/g' "$CONFIG_FILE"
                sed -i 's/use_crop: .*/use_crop: true/g' "$CONFIG_FILE"
                sed -i 's/use_ocr: .*/use_ocr: false/g' "$CONFIG_FILE"
                ;;
            4)
                # Setting 4: use_page: true, use_crop: true, use_ocr: true, use_ocr_both: true
                sed -i 's/use_page: .*/use_page: true/g' "$CONFIG_FILE"
                sed -i 's/use_page_ocr: .*/use_page_ocr: false/g' "$CONFIG_FILE"
                sed -i 's/use_crop: .*/use_crop: true/g' "$CONFIG_FILE"
                sed -i 's/use_ocr: .*/use_ocr: true/g' "$CONFIG_FILE"
                sed -i 's/use_ocr_both: .*/use_ocr_both: true/g' "$CONFIG_FILE"
                ;;
            5)
                # Setting 5: use_page: true, use_crop: true, use_ocr: true, use_ocr_raw: true
                sed -i 's/use_page: .*/use_page: true/g' "$CONFIG_FILE"
                sed -i 's/use_page_ocr: .*/use_page_ocr: false/g' "$CONFIG_FILE"
                sed -i 's/use_crop: .*/use_crop: true/g' "$CONFIG_FILE"
                sed -i 's/use_ocr: .*/use_ocr: true/g' "$CONFIG_FILE"
                sed -i 's/use_ocr_raw: .*/use_ocr_raw: true/g' "$CONFIG_FILE"
                ;;
            6)
                # Setting 6: use_page: true, use_crop: true, use_ocr: true, use_ocr_raw: false
                sed -i 's/use_page: .*/use_page: true/g' "$CONFIG_FILE"
                sed -i 's/use_page_ocr: .*/use_page_ocr: false/g' "$CONFIG_FILE"
                sed -i 's/use_crop: .*/use_crop: true/g' "$CONFIG_FILE"
                sed -i 's/use_ocr: .*/use_ocr: true/g' "$CONFIG_FILE"
                sed -i 's/use_ocr_raw: .*/use_ocr_raw: false/g' "$CONFIG_FILE"
                ;;
        esac

        # 执行 Python 命令
        echo "Executing: python run_generation.py --config $CONFIG_FILE"
        python run_generation.py --config "$CONFIG_FILE"
        python run_generation.py --config "$CONFIG_FILE"
        python run_generation.py --config "$CONFIG_FILE"
        
        # 检查执行状态
        if [ $? -eq 0 ]; then
            echo "Success: [$PROMPT_NAME - Setting $SETTING_ID]"
        else
            echo "Failed: [$PROMPT_NAME - Setting $SETTING_ID]"
            # 可以选择在这里 exit 1 终止脚本，或者继续运行下一个
        fi
        echo "----------------------------------------------------------------"
        echo ""

    done
done

echo "All tasks completed."