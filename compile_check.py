import jsonlines
import subprocess
import tempfile
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import re

def check_rust_syntax(code_str):
    with tempfile.NamedTemporaryFile(suffix=".rs", delete=False, mode="w+", encoding="utf-8") as tmp_file:
        tmp_path = tmp_file.name
        tmp_file.write(code_str)
        tmp_file.flush()

    try:
        if 'fn main' in code_str:
            result = subprocess.run(
                ["rustc", tmp_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8"
            )
        else:
            result = subprocess.run(
                ["rustc", "--crate-type=lib", tmp_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8"
            )
    finally:
        try:
            os.remove(tmp_path)
            if result.returncode == 0:
                os.remove(tmp_path.split(".")[0].split("/")[-1])
        except Exception as e:
            # print(f"Warning: Failed to delete temp file {tmp_path}: {e}")
            pass
    return result


def post_process(dirty_code):
    match = re.search(r"```(?:\w+)?\n(.*?)```", dirty_code, re.DOTALL)
    clean_code = match.group(1).strip() if match else dirty_code.strip()
    return clean_code

def process_item(rust_code):
    if "```" in rust_code:
        code_str = post_process(rust_code)
    else:
        code_str = rust_code
    compile_result = check_rust_syntax(code_str)
    returncode = compile_result.returncode
    compile_text = compile_result.stderr.strip()
    compile_text = compile_text[:1000]
    return returncode, compile_text


if __name__ == "__main__":
    rust_code = "use std::ffi::c_void;\r\nuse std::ptr;\r\n\r\n#[repr(C)]\r\npub struct CuckooHash {\r\n    table: *mut c_void,\r\n    // 其他字段...\r\n}\r\n\r\n// 同样保留 XPROBES_SITE 为占位\r\n#[allow(non_snake_case)]\r\nfn XPROBES_SITE(_system: &str, _event: &str, _sig: (&str,), _args: (*const CuckooHash,)) {\r\n    // 占位：可替换为日志、tracing 或其他调试功能\r\n}\r\n\r\npub unsafe fn cuckoo_hash_destroy(hash: *const CuckooHash) {\r\n    XPROBES_SITE(\r\n        \"cuckoo_hash\",\r\n        \"destroy\",\r\n        (\"*const CuckooHash\",),\r\n        (hash,),\r\n    );\r\n\r\n    if !hash.is_null() {\r\n        let table = (*hash).table;\r\n        if !table.is_null() {\r\n            libc::free(table);\r\n        }\r\n    }\r\n}\r\n"
    returncode, compile_text = process_item(rust_code)
    print("Return code:", returncode)
    print("Compile text:", compile_text)
    # rust_path = "/data/luofeng/c2rust/code/infer_result/merged_5/deepseekcoder-1.3b/infer_result.jsonl"
    # rust_data = list(jsonlines.open(rust_path))
    # # rust_code = rust_data[1]["oai_response"]["response"][0]
    # sum_1 = 0
    # idx = 0
    # for item in rust_data:
    #     rust_code = item["oai_response"]["response"][0]
    #     returncode, compile_text = process_item(rust_code)
    #     # print(returncode)
    #     if returncode == 1:
    #         print(len(compile_text))
    #         print(compile_text)
    #         input("sss")
    #         sum_1 += len(compile_text)
    #         idx += 1
    # print(sum_1)
    # print(sum_1/idx)


            