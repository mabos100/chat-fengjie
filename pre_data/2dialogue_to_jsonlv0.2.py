import json
import re


def convert_feng_to_chatml_robust(input_file, output_file):
    dataset = []

    # System Prompt: 设定王熙凤的人设
    system_prompt = (
        "你现在是《红楼梦》中的王熙凤（人称凤姐、琏二奶奶）。"
        "你的性格泼辣张狂、精明强干、言语爽利，说话常带三分笑，"
        "善于察言观色，同时也有些心狠手辣。请根据前文情境，模仿她的语气进行回答。"
    )

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # 1. 拆分数据块
        # 兼容 【ID:102】 和 【对话序列 ID: 102】 两种格式
        blocks = re.split(r'【(?:对话序列 )?ID:\s*\d+】', content)

        valid_count = 0
        skip_count = 0

        for block in blocks:
            block = block.strip()
            if not block:
                continue

            # --- 清洗脏数据 ---
            # 去掉开头可能残留的右引号（针对 ID:298, ID:300）
            block = block.lstrip('”').strip()
            block_s = block.split(" ")
            print(block_s)

            # 去掉可能存在的 "内容：" 前缀
            if block.startswith("内容："):
                block = block.replace("内容：", "", 1).strip()
            # --- 核心逻辑：智能正则匹配 ---
            # 解释：
            # (?P<context>.*?)       -> 捕获前面的所有铺垫
            # (?P<trigger>凤姐[^。！？]*?[道说问笑骂][：:]) -> 捕获“凤姐道：”这个动作，确保是凤姐说的
            # \s* -> 允许有空格
            # [“'‘]                  -> 引号可能是 “ 或 ' 或 ‘
            # (?P<content>.*?)       -> 捕获说话内容
            # [”'’]?                 -> 结尾引号，可能是 ” 或 ' 或 ’，甚至可能缺失
            # $                      -> 必须是结尾

            pattern = re.compile(
                r'(?P<context>.*?)(?P<trigger>凤姐[^。！？]*?[道说问笑骂][：:])\s*[“\'‘](?P<content>.*)',
                re.DOTALL
            )

            match = pattern.search(block)

            full_user_input = block_s[0]
            print(len(block_s[1]))
            if len(block_s[1]) == 0:
                block_s[1] = block_s[2]
            # 构建 ChatML
            entry = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": full_user_input},
                    {"role": "assistant", "content": block_s[1]}
                ]
            }
            dataset.append(entry)
            valid_count += 1

            # 如果匹配不到“凤姐...道：”，说明这条数据可能是坏的（如 ID:102）
            # print(f"跳过无效数据: {block[:20]}...")
            skip_count += 1

        # 写入 JSONL
        with open(output_file, 'w', encoding='utf-8') as f_out:
            for entry in dataset:
                f_out.write(json.dumps(entry, ensure_ascii=False) + '\n')

        print(f"处理完成！")
        print(f"有效对话: {valid_count} 条")
        print(f"过滤无效/格式错误: {skip_count} 条")
        print(f"结果已保存至: {output_file}")

    except FileNotFoundError:
        print(f"错误：找不到文件 {input_file}")


# --- 运行 ---
# 假设你的文件名为 feng_dialogue_flow.txt
convert_feng_to_chatml_robust('feng_dialogue_flow3.1.2.txt', 'feng_chatml3.1.2.jsonl')