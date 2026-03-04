import re


def extract_feng_dialogue_chain_strict(file_path):
    # --- 0. 配置 ---
    # 王熙凤的常见称呼
    feng_aliases = ["王熙凤", "凤姐", "凤姐儿", "凤丫头", "琏二奶奶", "凤辣子"]

    # ============================
    # 步骤 1: 处理文本 (清洗)
    # ============================
    try:
        with open(file_path, 'r', encoding='ANSI') as f:
            raw_content = f.read()

        # 去掉换行、回车、空格，使文本成为连续的一条长龙
        text = raw_content.replace('\n', '').replace('\r', '').replace(' ', '')
        print(f"步骤1完成：文本清洗完毕，总长度 {len(text)} 字。")

    except FileNotFoundError:
        print(f"错误：找不到文件 {file_path}")
        return []

    # ============================
    # 步骤 2: 找到所有对话并标记
    # ============================
    # 我们使用一个列表来存储所有对话的元数据，这相当于给每个对话打上了序号(ID)
    dialogues_list = []

    # 正则：匹配引号内的内容
    pattern = re.compile(r'“(.*?)”')

    # finditer 返回所有匹配对象的迭代器
    for match in pattern.finditer(text):
        dialogues_list.append({
            "id": len(dialogues_list),  # 序号标记 (0, 1, 2, ...)
            "start": match.start(),  # 引号开始在全文的绝对位置
            "end": match.end(),  # 引号结束在全文的绝对位置
            "content": match.group(1)  # 对话内容
        })

    print(f"步骤2完成：共标记了 {len(dialogues_list)} 处对话。")

    # ============================
    # 步骤 3: 找到王熙凤的话(n)，输出 (n-1) 到 (n)
    # ============================
    results = []

    # 从第 1 个开始遍历 (因为我们需要 n-1，所以不能从 0 开始)
    for i in range(1, len(dialogues_list)):
        dialogue_n = dialogues_list[i]  # 当前对话 (n)
        dialogue_prev = dialogues_list[i - 1]  # 上一句对话 (n-1)

        # --- A. 判断 n 是否为王熙凤 ---
        # 逻辑：查看当前引号前面的一段文字(前文语境)，看有没有王熙凤的名字
        # 我们向前看 50 个字，通常"凤姐道"紧挨着引号
        lookback_range = 50
        search_start = max(0, dialogue_n['start'] - lookback_range)
        # 截取引语部分 (从 n-1 结束 到 n 开始 之间的文字)
        # 注意：为了防止误判，我们最好只检查 n-1 结束之后的部分
        check_start = max(dialogue_prev['end'], search_start)
        pre_context = text[check_start: dialogue_n['start']]

        is_feng = False
        for name in feng_aliases:
            if name in pre_context:
                is_feng = True
                break

        # --- B. 如果是王熙凤，执行提取 ---
        if is_feng:
            # 我们要找提取的【起点】和【终点】

            # 终点很简单：就是 n 的结束位置
            extract_end = dialogue_n['end']

            # 起点比较讲究：
            # 题目要求包括 n-1。但为了阅读通顺，我们需要找到 n-1 的“引语”开始的地方。
            # 方法：从 n-1 的引号开始位置，向前找最近的一个句号/感叹号/问号。

            # 向前回溯 100 字找标点
            search_limit_prev = 100
            search_text_start = max(0, dialogue_prev['start'] - search_limit_prev)
            text_before_prev = text[search_text_start: dialogue_prev['start']]

            # 找最后一个结束标点
            last_punct_match = None
            for m in re.finditer(r'[。！？]', text_before_prev):
                last_punct_match = m

            if last_punct_match:
                # 起点设为标点之后
                # search_text_start + 标点位置 + 1
                extract_start = search_text_start + last_punct_match.end()
            else:
                # 没找到标点，就直接从 n-1 引号前一点截取 (兜底)
                extract_start = max(0, dialogue_prev['start'] - 20)

            # --- C. 截取并保存 ---
            full_segment = text[extract_start: extract_end]

            results.append({
                "seq_id": i,  # 记录这是全书第几句对话
                "text": full_segment,
                "n_1_content": dialogue_prev['content'],  # 方便调试看上一句是啥
                "n_content": dialogue_n['content']  # 方便调试看这句是啥
            })

    return results




# ============================
# 主程序运行
# ============================
filename = 'hongloumeng.txt'  # 请确保文件存在
extracted_data = extract_feng_dialogue_chain_strict(filename)

# 打印结果
print(f"\n步骤3完成：共提取到 {len(extracted_data)} 段符合要求的对话流。\n")
print("=" * 60)

# 展示前 5 条结果
for item in extracted_data[:5]:

    print(f"【对话序列 ID: {item['seq_id']}】")
    print(f"前句：{item['n_1_content']}")
    print(f"内容：{item['n_content']}")
    print("-" * 60)

# (可选) 保存到文件
with open('feng_dialogue_flow3.txt', 'w', encoding='utf-8') as f:
     for item in extracted_data:
       #f.write(f"【ID:{item['seq_id']}】 {item['text']} {item['n_1_content']} {item['n_content']}\n")    #feng_dialogue_flow2.txt
       f.write(f"【ID:{item['seq_id']}】 {item['n_1_content']} {item['n_content']}\n")  #feng_dialogue_flow3.txt