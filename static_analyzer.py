from tree_sitter import Language, Parser
import graphviz
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
import json
import jsonlines
import os
from tqdm import tqdm
import re

class StaticAnalyzer:
    """
    静态代码分析器\n
    初始化：tree-sitter的so文件路径、语言名称、代码，例如：analyzer = StaticAnalyzer('build/libtree-sitter-c.so', 'c', code)\n
    方法：\n
    find_pointers找到指针名称\n
    find_arrays找到数组名称\n
    get_array_declaration_text获取数组声明文本\n
    print_array_declarations打印数组声明\n
    generate_dot生成语法树图片\n
    save_syntax_tree_image保存语法树图片\n

    """
    def __init__(self, language_path, language_name, code):
        self.language = Language(language_path, language_name)
        self.parser = Parser()
        self.parser.set_language(self.language)
        self.code = bytes(code, 'utf8')
        self.tree = self.parser.parse(self.code)
        self.root_node = self.tree.root_node


    def find_arrays(self, node=None):
        if node is None:
            node = self.root_node
        arrays = []
        if node.type == 'array_declarator':
            arrays.append(node)
        for child in node.children:
            arrays.extend(self.find_arrays(child))
        return arrays

    def get_array_declaration_text(self, array_node, code=None):
        if code is None:
            code = self.code
        start_byte = array_node.start_byte
        end_byte = array_node.end_byte
        return code[start_byte:end_byte]

    def print_array_declarations(self, code=None):
        if code is None:
            code = self.code
        result = []
        arrays = self.find_arrays()
        for array in arrays:
            array_declaration = self.get_array_declaration_text(array, code)
            result.append(array_declaration)
            print(array_declaration)
        return result
    
    def get_array_declarations(self, code=None):
        if code is None:
            code = self.code
        result = []
        arrays = self.find_arrays()
        for array in arrays:
            array_declaration = self.get_array_declaration_text(array, code)
            result.append(array_declaration)
        if result == []:
            return None
        return result

    def generate_dot(self, node=None, graph=None, parent_id=None):
        if node is None:
            node = self.root_node
        if graph is None:
            graph = graphviz.Digraph()
        node_id = str(id(node))
        label = f"{node.type}"
        if node.is_named:
            label += f" ({node.text.decode('utf-8')})"
        graph.node(node_id, label)
        if parent_id:
            graph.edge(parent_id, node_id)
        for child in node.children:
            self.generate_dot(child, graph, node_id)
        return graph

    def save_syntax_tree_image(self, output_path, format='png'):
        dot = self.generate_dot()
        dot.render(output_path, format=format, cleanup=True)
    
    # 识别数组
    def find_array_declarations(self):
        array_declarations = []
        for child in self.root_node.children:
            if child.type == 'array_declarator':
                array_declarations.append(child)
        return array_declarations
   
    # 检测有没有变量被修改，并返回被修改后的变量名
    # 例如：int a = 1; a = 2; printf("%d", a); 这里会返回a，但是如果只是int a = 1; printf("%d", a); 这里不会返回a
    def find_modified_variables(self, node=None):
        if node is None:
            node = self.root_node
        modified_variables = set()
        
        # 遍历语法树，寻找赋值操作
        if node.type == 'assignment_expression':
            # 获取赋值左侧的变量名
            left_node = node.child_by_field_name('left')
            if left_node and left_node.type == 'identifier':
                modified_variables.add(left_node.text.decode('utf-8'))
        
        # 递归检查子节点
        for child in node.children:
            modified_variables.update(self.find_modified_variables(child))
        
        return list(modified_variables)
    
    def generate_symbol_table(self, node=None):
        """
        自动生成符号表，提取变量名及其类型。

        参数:
        - node: 当前语法树节点，默认为根节点。

        返回:
        - symbol_table: 字典，存储变量名及其类型。
        """
        if node is None:
            node = self.root_node

        symbol_table = {}

        # 遍历语法树，寻找变量声明
        for child in node.children:
            if child.type == 'declaration':
                # 提取类型和变量名
                type_node = child.child_by_field_name('type')
                declarator_node = child.child_by_field_name('declarator')

                if type_node and declarator_node:
                    var_type = type_node.text.decode('utf-8')
                    var_name = declarator_node.text.decode('utf-8')
                    var_name = var_name.split('=')[0].strip()  # 去除赋值部分
                    symbol_table[var_name] = var_type

            # 递归检查子节点
            symbol_table.update(self.generate_symbol_table(child))

        return symbol_table
    
    #直接找语法树中的cast_expression
    def find_cast_expression(self, node=None):
        """
        查找语法树中所有的cast_expression节点。

        参数:
        - node: 当前语法树节点，默认为根节点。

        返回:
        - List[Dict]: 包含强转的行号和代码片段的列表。
        """
        if node is None:
            node = self.root_node

        results = []

        def walk(node):
            if node.type == 'cast_expression':
                line_number = node.start_point[0] + 1  # Line numbers start from 1
                code_snippet = self.code[node.start_byte:node.end_byte].decode('utf-8')
                results.append({
                    "Line": line_number,
                    "Code snippet": code_snippet
                })
            for child in node.children:
                walk(child)

        walk(node)
        return results

    def detect_mixed_type_operations(self, node=None, symbol_table=None):
        """
        根据变量表检测二元运算中是否存在混合类型运算（如 int + long long)。

        返回:
        - 1: 如果检测到混合类型运算。
        - 0: 如果未检测到混合类型运算。
        """
    
        if node is None:
            node = self.root_node
        if symbol_table is None:
            raise ValueError("Symbol table is required to determine variable types.")
        
        # 如果symbol_table里的变量类型都是一样的，直接返回0，例如Symbol Table: {'a': 'int', 'b': 'int', 'e': 'int', 'c': 'int', 'd': 'int'}，字典的value都是int，直接返回0
        # first_value = symbol_table.get(list(symbol_table.keys())[0])
        # if all(value == first_value for value in symbol_table.values()):
        #     return 0
        
        return self.find_cast_expression()

    def infer_rust_allocation(self, malloc_code: str) -> str:
        if 'char' in malloc_code:
            return 'String or Vec<u8>'
        if 'sizeof(' in malloc_code and '*' not in malloc_code:
            return 'Box<T>'
        if '*' in malloc_code and 'sizeof' in malloc_code:
            if any(word in malloc_code for word in ['rows', 'cols']):
                return 'Vec<Vec<T>>'
            if any(char.isdigit() for char in malloc_code):
                return 'Box<[T; N]>'
            return 'Vec<T>'
        return 'Box<[T]>'

    def gen_array_context(self):
        result = []
        arrays = self.find_arrays()
        code = self.code.decode('utf-8')

        for array in arrays:
            line_number = array.start_point[0] + 1  # Line numbers start from 1
            code_snippet = self.get_array_declaration_text(array, code)
            # notation = "When indexing into arrays with expressions, ensure that the index is cast to usize in Rust"
            result.append({
                "Line": line_number,
                "Code snippet": code_snippet
                # "Notation": notation
            })

        return result
    
    def extract_malloc_expressions(self):
        root_node = self.root_node
        results = []         
        malloc_vars = set()
        code = self.code.decode('utf-8')

        def walk(node):
            if node.type == "declaration":
                children_code = code[node.start_byte:node.end_byte]
                if "alloc" in children_code.lower():
                    equal_pos = children_code.find('=')
                    if equal_pos != -1:
                        left = children_code[:equal_pos].strip()
                        tokens = left.split()
                        if len(tokens) >= 1:
                            var_name = tokens[-1].replace('*', '').strip()
                            malloc_vars.add(var_name)
               
                if "alloc" in children_code.lower():
                    rust_guess = self.infer_rust_allocation(children_code)
                    line_number = node.start_point[0] + 1  # 从1开始
                    results.append({
                        "code": children_code,
                        "rust_type": rust_guess,
                        "line": line_number
                    })

            for child in node.children:
                walk(child)

        walk(root_node)
        return {
            "malloc_info": results,
            "malloc_vars": list(malloc_vars)
        }
    
    def gen_malloc_context(self) -> dict:
        """
        基于规则库构建结构化输出:
        {
            "Line": \n
            "Code snippet": \n
            "Suggested_rust_type":\n
            "Reason":\n
        }
        """
        results = []
        rules_path = "/data/luofeng/c2rust/code/Rules.jsonl"
        with open(rules_path, 'r', encoding='utf-8') as f:
            rules = json.load(f)
        reason_map = rules["malloc"]["reason_map"]
        analysis = self.extract_malloc_expressions()
        malloc_info = analysis["malloc_info"]
        for item in malloc_info:
            reason = reason_map.get(item["rust_type"], "N/A")
            results.append({
                "Line": item["line"],
                "Code snippet": item["code"],
                "Suggested_rust_type": item["rust_type"],
                "Reason": reason
            })

        return results
    def infer_rust_pointer_type(self, pointer_code: str) -> str:
        code = pointer_code.lower()

        # 判断是否是 const 指针
        if 'const' in code:
            return '*const T'

        # 如果是在声明中（有类型名）且含 malloc，可能是拥有者类型
        if 'malloc' in code:
            if 'char' in code and ('256' in code or 'sizeof(char)' in code):
                return 'String or Vec<u8>'
            if any(word in code for word in ['rows', 'cols']):
                return 'Vec<Vec<T>>'
            if '*' in code and 'sizeof' in code:
                return 'Vec<T>'
            return 'Box<T>'

        # 如果是声明（比如 int* ptr;），可能是引用或 Box
        if any(tp in code for tp in ['int', 'float', 'double']):
            return 'Box<T> or &T'
        if 'char' in code:
            return 'String or Vec<u8>'

        # 如果是 *ptr 这种用法（解引用），默认是裸指针
        if '*' in code and '=' in code:
            return '*mut T'
        if '*' in code:
            return '*const T'

        return '&T'  # fallback
       

    def gen_pointer_context(self, malloc_vars):
        root_node = self.root_node
        code = self.code.decode("utf-8")
        results = []

        # 加载推理规则
        rules_path = "/data/luofeng/c2rust/code/Rules.jsonl"
        with open(rules_path, 'r', encoding='utf-8') as f:
            rules = json.load(f)
        reason_map = rules["pointers"]["reason_map"]

        def find_ancestor(node, target_type):
            while node is not None and node.type != target_type:
                node = node.parent
            return node
        
        def walk(node):
            if node.type == 'pointer_declarator':
                decl_node = find_ancestor(node, 'declaration')
                if decl_node:
                    code_snippet = code[decl_node.start_byte:decl_node.end_byte]
                    # line_number = node.start_point[0] + 1
                    line_number = decl_node.start_point[0] + 1

                    # 跳过 malloc 变量
                    if any(var in code_snippet for var in malloc_vars):
                        return

                    suggested_type = self.infer_rust_pointer_type(code_snippet)
                    reason = reason_map.get(suggested_type, "N/A")
                    results.append({
                        "Line": line_number,
                        "Code snippet": code_snippet,
                        "Suggested_rust_type": suggested_type,
                        "Reason": reason
                    })

            elif node.type == 'unary_expression':
                operator = node.child_by_field_name('operator')
                if operator and operator.text == b'*':
                    decl_node = find_ancestor(node, 'declaration')
                    if decl_node:
                        code_snippet = code[decl_node.start_byte:decl_node.end_byte]
                        # line_number = node.start_point[0] + 1
                        line_number = decl_node.start_point[0] + 1

                        # 跳过 malloc 变量（保守检查）
                        if any(var in code_snippet for var in malloc_vars):
                            return

                        suggested_type = self.infer_rust_pointer_type(code_snippet)
                        reason = reason_map.get(suggested_type, "N/A")
                        results.append({
                            "Line": line_number,
                            "Code snippet": code_snippet,
                            "Suggested_rust_type": suggested_type,
                            "Reason": reason
                        })

            for child in node.children:
                walk(child)

        walk(root_node)
        return results
    
    def detect_multiple_scanf_inputs(self):
        c_code = self.code
        c_code = c_code.decode("utf-8")
        results = []
        lines = c_code.splitlines()
        # print(c_code)

        # Pattern: scanf with multiple format specifiers（匹配"%d%d%s"等）
        scanf_pattern = re.compile(r'scanf\s*\(\s*"(.*?)"\s*,(.*?)\);')

        for idx, line in enumerate(lines):
            match = scanf_pattern.search(line)
            if not match:
                continue

            format_string = match.group(1)
            num_specifiers = len(re.findall(r'%[dscfuxlg]', format_string))  # 基本的格式控制符

            if num_specifiers >= 2:
                results.append({
                    "Line": idx + 1,
                    "Code snippet": line.strip()
                    # "Suggested_rust": (
                    #     "use std::io::{self, Read};\n"
                    #     "fn main() {\n"
                    #     "let mut input = String::new();\n"
                    #     "io::stdin().read_to_string(&mut input).unwrap();\n"
                    #     "let mut parts = input.split_whitespace();\n"
                    #     "let value1: usize = parts.next().unwrap().parse().unwrap();\n"
                    #     "let value2: usize = parts.next().unwrap().parse().unwrap();}"
                    # ),
                    # "Reason": (
                    #     "C's `scanf` can automatically read multiple variables across lines, "
                    #     "whereas in Rust, `read_to_string` can achieve cross-line reading, and it needs to be used together with `split_whitespace()` for splitting."
                    # )
                })

        return results
if __name__ == "__main__":
    # 读取示范库
    examples = list(jsonlines.open('your_demonstrate_corpus.jsonl', mode='r'))
    malloc_info = {"number": 0, "text":[], "code":[], "idx":[], "src_uid":[]}
    pointers_info = {"number": 0, "text":[], "code":[], "idx":[], "src_uid":[]}
    array_info = {"number": 0, "text":[], "code":[], "idx":[], "src_uid":[]}
    mixtype_info = {"number": 0, "text":[], "code":[], "idx":[], "src_uid":[]}
    io_info = {"number": 0, "text":[], "code":[], "idx":[], "src_uid":[]}
    
    
    for data in tqdm(examples, desc="Processing datasets"):
        c_code = data['source_code']
        analyzer = StaticAnalyzer('build/libtree-sitter-c.so', 'c', c_code)
        is_malloc = 0
        is_pointers = 0
        is_array = 0
        is_mixtype = 0
        is_io = 0
        # malloc检测功能测试
        if analyzer.extract_malloc_expressions()["malloc_vars"] != []:
            is_malloc = 1
            malloc_context = analyzer.gen_malloc_context()
            malloc_info["number"] += 1
            malloc_info['src_uid'].append(data["src_uid"])
            output_file = "./dataset/example/malloc.jsonl"
            with jsonlines.open(output_file, mode='a') as writer:
                writer.write(data)
        # 指针变量检测功能测试
        malloc_vars = analyzer.extract_malloc_expressions()["malloc_vars"]
        if analyzer.gen_pointer_context(malloc_vars):
            is_pointers = 1
            pointers_context = analyzer.gen_pointer_context(malloc_vars)
            pointers_info["number"] += 1
            pointers_info['src_uid'].append(data["src_uid"])
            output_file = "./dataset/example/pointers.jsonl"
            with jsonlines.open(output_file, mode='a') as writer:
                writer.write(data)
        # 数组检测功能测试
        if analyzer.gen_array_context():
            is_array = 1
            array_context = analyzer.gen_array_context()
            array_info["number"] += 1
            output_file = "./dataset/example/array.jsonl"
            with jsonlines.open(output_file, mode='a') as writer:
                writer.write(data)
        #混合类型运算检测功能测试
        symbol_table = analyzer.generate_symbol_table()
        if analyzer.detect_mixed_type_operations(symbol_table=symbol_table) != []:
            is_mixtype = 1
            mixtype_context = analyzer.detect_mixed_type_operations(symbol_table=symbol_table)
            mixtype_info["number"] += 1
            output_file = "./dataset/example/mixtype.jsonl"
            with jsonlines.open(output_file, mode='a') as writer:
                writer.write(data)
        
        # io检测功能
        io_context = analyzer.detect_multiple_scanf_inputs()
        rules = ""
        if io_context != []:
            is_io = 1
            io_info["number"] += 1
            output_file = "./dataset/example/io.jsonl"
            with jsonlines.open(output_file, mode='a') as writer:
                writer.write(data)
    
    print("malloc_num:", malloc_info["number"])
    print("pointers_num:", pointers_info["number"])
    print("array_num:", array_info["number"])   
    print("mixtype_num:", mixtype_info["number"])
    print("io_num:", io_info["number"])
    print("===================================")
    print("malloc_srcuid:", malloc_info["src_uid"])
    print("pointers_srcuid:", pointers_info["src_uid"])
                
        
        
