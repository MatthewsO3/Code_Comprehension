import os
from pathlib import Path
from tree_sitter import Language, Parser
import subprocess
import sys

# Paths
BUILD_DIR = Path('build')
BUILD_DIR.mkdir(exist_ok=True)
CPP_GRAMMAR_DIR = Path('tree-sitter-cpp')
CPP_DLL = BUILD_DIR / 'my-languages.dll'  # Windows DLL

# Step 1: Compile C++ grammar into DLL
if not CPP_DLL.exists():
    print("Building Tree-sitter C++ parser DLL...")

    # Make sure the grammar exists
    if not CPP_GRAMMAR_DIR.exists():
        print("Error: C++ grammar not found!")
        print("Please clone it: git clone https://github.com/tree-sitter/tree-sitter-cpp")
        sys.exit(1)

    parser_c_file = CPP_GRAMMAR_DIR / 'src' / 'parser.c'
    if not parser_c_file.exists():
        print("Error: parser.c not found in tree-sitter-cpp/src/")
        sys.exit(1)

    # Compile with gcc (Windows)
    compile_cmd = [
        'gcc',
        '-shared',
        '-o', str(CPP_DLL),
        '-fPIC',
        str(parser_c_file)
    ]

    try:
        subprocess.run(compile_cmd, check=True)
        print(f"Compiled DLL: {CPP_DLL}")
    except subprocess.CalledProcessError as e:
        print("Failed to compile C++ parser DLL:", e)
        sys.exit(1)
else:
    print(f"Using existing DLL: {CPP_DLL}")

# Step 2: Load parser in Python
CPP_LANGUAGE = Language(str(CPP_DLL), 'cpp')
parser = Parser()
parser.set_language(CPP_LANGUAGE)
print("Tree-sitter C++ parser ready!")

# Optional: Test parsing
code = b"""
int add(int a, int b) {
    return a + b;
}
"""
tree = parser.parse(code)
print(tree.root_node.sexp())
