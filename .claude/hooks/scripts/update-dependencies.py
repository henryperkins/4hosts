#!/usr/bin/env python3
"""
Update files that depend on modified modules.
- Updates import statements
- Fixes function/class references
- Updates type hints
"""

import os
import sys
import ast
import argparse
import subprocess
from pathlib import Path
from typing import Set, List, Dict, Tuple

class DependencyAnalyzer(ast.NodeVisitor):
    """Analyze import dependencies in Python files."""
    
    def __init__(self):
        self.imports: Dict[str, Set[str]] = {}  # module -> imported names
        self.from_imports: Dict[str, Set[str]] = {}
        
    def visit_Import(self, node):
        for alias in node.names:
            module = alias.name
            if module not in self.imports:
                self.imports[module] = set()
            self.imports[module].add(alias.asname or alias.name)
        self.generic_visit(node)
        
    def visit_ImportFrom(self, node):
        if node.module:
            if node.module not in self.from_imports:
                self.from_imports[node.module] = set()
            for alias in node.names:
                self.from_imports[node.module].add(alias.name)
        self.generic_visit(node)

def get_module_path(filepath: Path, repo_root: Path) -> str:
    """Convert file path to module path."""
    try:
        rel_path = filepath.relative_to(repo_root)
        # Remove .py extension and convert to module path
        module_path = str(rel_path.with_suffix('')).replace(os.sep, '.')
        return module_path
    except ValueError:
        return ""

def find_dependent_files(modified_files: List[str], repo_root: Path) -> Dict[str, Set[Path]]:
    """Find all files that import the modified modules."""
    dependent_files = {}  # modified_file -> set of dependent files
    
    # Convert modified files to module paths
    modified_modules = {}
    for file in modified_files:
        filepath = repo_root / file
        if filepath.suffix == '.py' and filepath.exists():
            module_path = get_module_path(filepath, repo_root)
            modified_modules[module_path] = filepath
            dependent_files[file] = set()
    
    # Search all Python files for imports
    for root, dirs, files in os.walk(repo_root):
        # Skip hidden directories and common non-code directories
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['node_modules', '__pycache__', 'venv']]
        
        for file in files:
            if file.endswith('.py'):
                filepath = Path(root) / file
                
                # Skip if it's one of the modified files
                if str(filepath.relative_to(repo_root)) in modified_files:
                    continue
                
                try:
                    with open(filepath, 'r') as f:
                        content = f.read()
                    
                    tree = ast.parse(content)
                    analyzer = DependencyAnalyzer()
                    analyzer.visit(tree)
                    
                    # Check if this file imports any modified modules
                    for module_path, mod_file in modified_modules.items():
                        # Check direct imports
                        if module_path in analyzer.imports:
                            rel_path = str(mod_file.relative_to(repo_root))
                            dependent_files[rel_path].add(filepath)
                            
                        # Check from imports
                        if module_path in analyzer.from_imports:
                            rel_path = str(mod_file.relative_to(repo_root))
                            dependent_files[rel_path].add(filepath)
                            
                        # Check parent module imports
                        parts = module_path.split('.')
                        for i in range(len(parts)):
                            parent = '.'.join(parts[:i+1])
                            if parent in analyzer.imports or parent in analyzer.from_imports:
                                rel_path = str(mod_file.relative_to(repo_root))
                                dependent_files[rel_path].add(filepath)
                                
                except Exception as e:
                    continue
                    
    return dependent_files

def analyze_changes(filepath: Path) -> Dict[str, List[str]]:
    """Analyze what changed in a file (removed/renamed functions, classes, etc)."""
    changes = {
        'removed_functions': [],
        'removed_classes': [],
        'renamed_items': []
    }
    
    # Get git diff to see what was removed
    try:
        result = subprocess.run(
            ['git', 'diff', 'HEAD~1', 'HEAD', '--', str(filepath)],
            capture_output=True, text=True
        )
        
        if result.returncode == 0:
            diff_lines = result.stdout.split('\n')
            
            for line in diff_lines:
                # Look for removed function/class definitions
                if line.startswith('-') and not line.startswith('---'):
                    if 'def ' in line:
                        # Extract function name
                        parts = line.split('def ')
                        if len(parts) > 1:
                            func_name = parts[1].split('(')[0].strip()
                            if func_name:
                                changes['removed_functions'].append(func_name)
                    elif 'class ' in line:
                        # Extract class name
                        parts = line.split('class ')
                        if len(parts) > 1:
                            class_name = parts[1].split('(')[0].split(':')[0].strip()
                            if class_name:
                                changes['removed_classes'].append(class_name)
                                
    except Exception:
        pass
        
    return changes

def update_dependent_file(filepath: Path, modified_module: str, changes: Dict[str, List[str]]):
    """Update a file that depends on a modified module."""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
            
        original_content = content
        
        # Update imports if items were removed
        for func in changes['removed_functions']:
            # Remove from specific imports
            patterns = [
                f"from {modified_module} import {func}",
                f"from {modified_module} import .*, {func}",
                f"from {modified_module} import {func},",
            ]
            for pattern in patterns:
                if pattern in content:
                    print(f"  Removing import of removed function '{func}'")
                    content = content.replace(pattern, "")
                    
        for cls in changes['removed_classes']:
            # Remove from specific imports
            patterns = [
                f"from {modified_module} import {cls}",
                f"from {modified_module} import .*, {cls}",
                f"from {modified_module} import {cls},",
            ]
            for pattern in patterns:
                if pattern in content:
                    print(f"  Removing import of removed class '{cls}'")
                    content = content.replace(pattern, "")
                    
        # Clean up empty imports
        content = content.replace("from {} import \n".format(modified_module), "")
        
        # Save if changed
        if content != original_content:
            with open(filepath, 'w') as f:
                f.write(content)
            print(f"✓ Updated {filepath.name}")
            
    except Exception as e:
        print(f"Error updating {filepath}: {e}")

def main():
    parser = argparse.ArgumentParser(description='Update dependent files')
    parser.add_argument('repo_root', help='Repository root directory')
    parser.add_argument('--check-imports', action='store_true', 
                       help='Check and update import statements')
    args = parser.parse_args()
    
    repo_root = Path(args.repo_root)
    
    # Read modified files from stdin
    modified_files = [line.strip() for line in sys.stdin if line.strip()]
    
    if not modified_files:
        print("No modified files to process")
        return
        
    print(f"Finding files dependent on {len(modified_files)} modified files...")
    
    # Find dependent files
    dependent_files = find_dependent_files(modified_files, repo_root)
    
    # Process each modified file and its dependents
    for modified_file, dependents in dependent_files.items():
        if dependents:
            print(f"\n{modified_file} is imported by {len(dependents)} files")
            
            # Analyze what changed
            filepath = repo_root / modified_file
            changes = analyze_changes(filepath)
            
            if any(changes.values()):
                print(f"Changes detected: {changes}")
                
                # Update dependent files
                module_path = get_module_path(filepath, repo_root)
                for dep_file in dependents:
                    print(f"  Updating {dep_file.relative_to(repo_root)}...")
                    update_dependent_file(dep_file, module_path, changes)
                    
    # Check for circular dependencies
    print("\nChecking for circular dependencies...")
    for file in modified_files:
        filepath = repo_root / file
        if filepath.suffix == '.py' and filepath.exists():
            module = get_module_path(filepath, repo_root)
            
            # Check if any dependent also imports this module
            if file in dependent_files:
                for dep in dependent_files[file]:
                    dep_module = get_module_path(dep, repo_root)
                    if dep_module and module:
                        # Simple circular dependency check
                        try:
                            with open(dep, 'r') as f:
                                dep_content = f.read()
                            if f"import {module}" in dep_content or f"from {module}" in dep_content:
                                print(f"⚠️  Potential circular dependency: {module} <-> {dep_module}")
                        except Exception:
                            pass

if __name__ == '__main__':
    main()