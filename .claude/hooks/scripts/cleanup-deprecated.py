#!/usr/bin/env python3
"""
Cleanup deprecated code after modifications.
Identifies and removes:
- Unused imports
- Dead code blocks
- Deprecated functions/classes
- Redundant files
"""

import os
import sys
import ast
import argparse
import subprocess
from pathlib import Path
from typing import Set, List, Dict, Tuple

class DeprecatedCodeAnalyzer(ast.NodeVisitor):
    """Analyze Python files for deprecated code patterns."""
    
    def __init__(self):
        self.imports: Set[str] = set()
        self.defined_names: Set[str] = set()
        self.used_names: Set[str] = set()
        self.deprecated_patterns: List[str] = []
        
    def visit_Import(self, node):
        for alias in node.names:
            self.imports.add(alias.name)
        self.generic_visit(node)
        
    def visit_ImportFrom(self, node):
        if node.module:
            for alias in node.names:
                self.imports.add(f"{node.module}.{alias.name}")
        self.generic_visit(node)
        
    def visit_FunctionDef(self, node):
        self.defined_names.add(node.name)
        # Check for deprecated decorators or comments
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name) and decorator.id == 'deprecated':
                self.deprecated_patterns.append(f"Function '{node.name}' is marked as deprecated")
        self.generic_visit(node)
        
    def visit_ClassDef(self, node):
        self.defined_names.add(node.name)
        self.generic_visit(node)
        
    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):
            self.used_names.add(node.id)
        self.generic_visit(node)

def analyze_file(filepath: Path) -> Dict[str, List[str]]:
    """Analyze a single file for deprecated code."""
    issues = {
        'unused_imports': [],
        'unused_definitions': [],
        'deprecated_items': []
    }
    
    try:
        with open(filepath, 'r') as f:
            content = f.read()
            
        tree = ast.parse(content)
        analyzer = DeprecatedCodeAnalyzer()
        analyzer.visit(tree)
        
        # Find unused imports
        for imp in analyzer.imports:
            imp_name = imp.split('.')[-1]
            if imp_name not in content.replace(f"import {imp}", "").replace(f"from {imp}", ""):
                issues['unused_imports'].append(imp)
                
        # Find unused definitions
        for name in analyzer.defined_names:
            if name not in analyzer.used_names and not name.startswith('_'):
                # Check if it's exported or used elsewhere
                if f"__all__" not in content or name not in content:
                    issues['unused_definitions'].append(name)
                    
        issues['deprecated_items'] = analyzer.deprecated_patterns
        
    except Exception as e:
        print(f"Error analyzing {filepath}: {e}")
        
    return issues

def find_deprecated_files(repo_root: Path, modified_files: List[str]) -> List[Path]:
    """Find files that might be deprecated after recent modifications."""
    deprecated_files = []
    
    # Check for files that are no longer imported
    all_imports = set()
    
    for root, dirs, files in os.walk(repo_root):
        # Skip hidden directories and common non-code directories
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['node_modules', '__pycache__', 'venv']]
        
        for file in files:
            if file.endswith('.py'):
                filepath = Path(root) / file
                try:
                    with open(filepath, 'r') as f:
                        content = f.read()
                        tree = ast.parse(content)
                        
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Import):
                            for alias in node.names:
                                all_imports.add(alias.name)
                        elif isinstance(node, ast.ImportFrom) and node.module:
                            all_imports.add(node.module)
                            
                except Exception:
                    continue
                    
    # Check if modified files removed imports to other files
    for file in Path(repo_root).rglob("*.py"):
        module_path = file.relative_to(repo_root).with_suffix('').as_posix().replace('/', '.')
        if module_path not in all_imports and str(file) not in modified_files:
            # Check if it's a main file or test file
            if not any(pattern in str(file) for pattern in ['__main__', 'test_', '_test.py', 'main.py']):
                deprecated_files.append(file)
                
    return deprecated_files

def remove_deprecated_code(filepath: Path, issues: Dict[str, List[str]]):
    """Remove deprecated code from a file."""
    if not any(issues.values()):
        return
        
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
            
        modified = False
        new_lines = []
        skip_next = False
        
        for i, line in enumerate(lines):
            if skip_next:
                skip_next = False
                continue
                
            # Remove unused imports
            should_remove = False
            for unused_import in issues['unused_imports']:
                if f"import {unused_import}" in line or f"from {unused_import}" in line:
                    should_remove = True
                    modified = True
                    print(f"  Removing unused import: {unused_import}")
                    break
                    
            if not should_remove:
                new_lines.append(line)
                
        if modified:
            with open(filepath, 'w') as f:
                f.writelines(new_lines)
            print(f"✓ Cleaned up {filepath}")
            
    except Exception as e:
        print(f"Error cleaning {filepath}: {e}")

def main():
    parser = argparse.ArgumentParser(description='Clean up deprecated code')
    parser.add_argument('repo_root', help='Repository root directory')
    parser.add_argument('--modified-files', action='store_true', 
                       help='Read modified files from stdin')
    args = parser.parse_args()
    
    repo_root = Path(args.repo_root)
    
    if args.modified_files:
        # Read modified files from stdin
        modified_files = [line.strip() for line in sys.stdin if line.strip()]
    else:
        # Get modified files from git
        result = subprocess.run(['git', 'diff', '--name-only'], 
                              capture_output=True, text=True, cwd=repo_root)
        modified_files = result.stdout.strip().split('\n') if result.stdout else []
        
    print(f"Analyzing {len(modified_files)} modified files...")
    
    # Analyze modified Python files
    for file in modified_files:
        filepath = repo_root / file
        if filepath.suffix == '.py' and filepath.exists():
            print(f"\nAnalyzing {file}...")
            issues = analyze_file(filepath)
            
            if any(issues.values()):
                print(f"Found issues in {file}:")
                for issue_type, items in issues.items():
                    if items:
                        print(f"  {issue_type}: {', '.join(items)}")
                        
                # Remove deprecated code
                remove_deprecated_code(filepath, issues)
                
    # Find potentially deprecated files
    deprecated_files = find_deprecated_files(repo_root, modified_files)
    if deprecated_files:
        print("\nPotentially deprecated files (no longer imported):")
        for file in deprecated_files:
            print(f"  - {file.relative_to(repo_root)}")
            
if __name__ == '__main__':
    main()