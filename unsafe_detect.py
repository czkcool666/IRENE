import re
import jsonlines
import os
import pandas as pd
from codebleu import calc_codebleu
import re
import warnings
warnings.filterwarnings("ignore")

def analyze_rust_unsafe_blocks(code: str):
    lines = code.strip().splitlines()
    total_lines = len(lines)

    def extract_block(start_regex):
        block_lines = set()
        inside = False
        brace_count = 0
        for idx, line in enumerate(lines):
            if not inside and re.search(start_regex, line):
                inside = True
                brace_count += line.count("{") - line.count("}")
                block_lines.add(idx)
                continue
            if inside:
                block_lines.add(idx)
                brace_count += line.count("{") - line.count("}")
                if brace_count <= 0:
                    inside = False
        return block_lines

    # 1. Multi-line compiler directives like #![allow(...)]
    directive_lines = set()
    inside_directive = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("#![allow"):
            inside_directive = True
            directive_lines.add(i)
            if ")" in stripped:
                inside_directive = False
        elif inside_directive:
            directive_lines.add(i)
            if ")" in stripped:
                inside_directive = False

    # 2. extern "C" blocks
    extern_c_block = extract_block(r'extern\s+"C"')

    # 3. unsafe blocks or unsafe fn
    unsafe_block = extract_block(r'\bunsafe\b')

    # 4. implicit casts using `as`
    implicit_casts = {i for i, line in enumerate(lines) if "libc::" in line}

    # # 5. unsafe functions like printf/scanf
    unsafe_calls = {i for i, line in enumerate(lines) if re.search(r"\b(printf|scanf)\b", line)}

    issue_blocks = {
        "compiler_directives": directive_lines,
        "extern_c": extern_c_block,
        "unsafe_block": unsafe_block,
        "implicit_casting": implicit_casts,
        "unsafe_functions": unsafe_calls,
    }

    issue_counts = {k: len(v) for k, v in issue_blocks.items()}
    issue_percentages = {k: f"{(len(v)/total_lines*100):.2f}%" for k, v in issue_blocks.items()}

    # Add: all unsafe line numbers merged
    all_unsafe_lines = set().union(*issue_blocks.values())
    issue_counts["all_unsafe"] = len(all_unsafe_lines)
    issue_percentages["all_unsafe"] = len(all_unsafe_lines)/total_lines

    return issue_counts, issue_percentages


if __name__ == "__main__":
    file_path = '/data/luofeng/c2rust/baselines/c2rust/result_local/infer_result.jsonl'
    # file_path = '/data/luofeng/c2rust/code/infer_result/summary/Qwen2.5-Coder-7B-Instruct/infer_result.jsonl'
    data = list(jsonlines.open(file_path, mode='r'))
    unsafe_rate = 0
    unsafe_loc_rate = 0
    codebleu_score = 0
    for item in data:
        rust_code = item['oai_response']['response'][0]
        counts, percentages = analyze_rust_unsafe_blocks(rust_code)
                # 计算unsafe rate 和 unsafe loc rate
        unsafe_loc_rate += percentages['all_unsafe']
        if percentages['all_unsafe'] != 0:
            unsafe_rate += 1
        reference = item['source_data']['target_code']
        prediction = rust_code
        result = calc_codebleu([reference], [prediction], lang="rust", weights=(0.33, 0.33, 0.33, 0), tokenizer=None)
        if result['dataflow_match_score'] != 0:
            result = calc_codebleu([reference], [prediction], lang="rust", weights=(0.25, 0.25, 0.25, 0.25), tokenizer=None)
        codebleu_score += result['codebleu']
    unsafe_rate /= len(data)       
    unsafe_loc_rate /= len(data)
    codebleu_score /= len(data)

    

    score_df = pd.DataFrame({
        "CodeBLEU": [f"{codebleu_score * 100:.2f}%"],
        "Unsafe Rate": [f"{unsafe_rate * 100:.2f}%"],
        "Unsafe Loc Rate": [f"{unsafe_loc_rate * 100:.2f}%"],
    })
    print(f"CodeBLEU: {codebleu_score * 100:.2f}%")
    print(f"unsafe_rate: {unsafe_rate}")
    print(f"unsafe_loc_rate: {unsafe_loc_rate}")
    score_df.to_csv(f"/data/luofeng/c2rust/baselines/c2rust/result_local/score.csv", index=False)
    # rust_code = data[0]['oai_response']['response'][0]
    # print(rust_code)

    # counts, percentages = analyze_rust_unsafe_blocks(rust_code)
    # print("行数统计:", counts)
    # print("占比统计:", percentages)
