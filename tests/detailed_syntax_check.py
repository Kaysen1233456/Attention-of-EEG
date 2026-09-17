"""详细语法检查：检查所有Python文件的AST结构、导入、类定义、函数签名"""
import ast
import sys
from pathlib import Path

project_root = Path(r"D:\Attention\src\attention_model")
scripts_dir = Path(r"D:\Attention\scripts")

all_files = list(project_root.rglob("*.py")) + list(scripts_dir.glob("*.py"))
all_files = [f for f in all_files if "__pycache__" not in str(f)]

print(f"共检查 {len(all_files)} 个Python文件")
print("=" * 70)

errors = []
warnings = []

for filepath in sorted(all_files):
    rel_path = filepath.relative_to(Path(r"D:\Attention"))
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            source = f.read()
        tree = ast.parse(source, filename=str(filepath))

        # 检查导入
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)

        # 检查类定义
        classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]

        # 检查函数定义
        functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]

        print(f"  OK  {rel_path}")
        print(f"       导入: {len(imports)}个, 类: {classes}, 函数: {len(functions)}个")

    except SyntaxError as e:
        errors.append((rel_path, e))
        print(f"  FAIL  {rel_path}: {e}")
    except Exception as e:
        errors.append((rel_path, e))
        print(f"  ERROR  {rel_path}: {e}")

print()
print("=" * 70)
if errors:
    print(f"发现 {len(errors)} 个错误:")
    for path, err in errors:
        print(f"  - {path}: {err}")
else:
    print("所有文件语法检查通过！")

# 检查跨模块导入一致性
print()
print("=" * 70)
print("跨模块导入一致性检查:")
print("=" * 70)

# 检查__init__.py导出的模块是否都存在
init_files = list(project_root.rglob("__init__.py"))
for init_file in init_files:
    rel_path = init_file.relative_to(Path(r"D:\Attention"))
    try:
        with open(init_file, "r", encoding="utf-8") as f:
            source = f.read()
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module and node.level == 1:  # 相对导入
                    module_path = init_file.parent / (node.module.replace(".", "/") + ".py")
                    if not module_path.exists():
                        # 可能是包
                        package_path = init_file.parent / node.module.replace(".", "/") / "__init__.py"
                        if not package_path.exists():
                            warnings.append(f"{rel_path}: 导入的模块不存在: .{node.module}")
                            print(f"  警告: {rel_path} -> .{node.module} 模块不存在")
    except Exception as e:
        print(f"  检查 {rel_path} 时出错: {e}")

if not any("模块不存在" in str(w) for w in warnings):
    print("  所有跨模块导入一致！")

print()
print("=" * 70)
print(f"总结: {len(all_files)}个文件, {len(errors)}个错误, {len(warnings)}个警告")
print("=" * 70)
