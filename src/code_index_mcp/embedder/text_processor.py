#!/usr/bin/env python3
import re
import os
from typing import Dict, List, Any, Optional
import inflection

def extract_code_structure(content: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Extract functions and classes from Python code.
    
    Args:
        content: Python code content
        
    Returns:
        Dictionary containing functions and classes
    """
    # Simple regex-based extraction
    functions = []
    classes = []
    
    # Find function definitions
    function_pattern = r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(([^)]*)\)\s*(?:->\s*([^:]+))?\s*:'
    for match in re.finditer(function_pattern, content):
        name = match.group(1)
        params = match.group(2)
        return_type = match.group(3)
        
        # Skip if it's a dunder method
        if name.startswith('__') and name.endswith('__'):
            continue
        
        functions.append({
            'name': name,
            'params': params.strip(),
            'return_type': return_type.strip() if return_type else None,
            'start': match.start(),
            'end': match.end()
        })
    
    # Find class definitions
    class_pattern = r'class\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:\(([^)]*)\))?\s*:'
    for match in re.finditer(class_pattern, content):
        name = match.group(1)
        inherits = match.group(2)
        
        classes.append({
            'name': name,
            'inherits': inherits.strip() if inherits else None,
            'start': match.start(),
            'end': match.end()
        })
    
    return {
        'functions': functions,
        'classes': classes
    }

def humanize_identifier(identifier: str) -> str:
    """
    Convert a code identifier to a human-readable string.
    
    Args:
        identifier: Code identifier (e.g., function_name, ClassName)
        
    Returns:
        Human-readable string
    """
    if not identifier:
        return ""
        
    # Handle snake_case
    if '_' in identifier:
        parts = identifier.split('_')
        return ' '.join(part.lower() for part in parts if part)
    
    # Handle camelCase or PascalCase
    humanized = inflection.humanize(inflection.underscore(identifier))
    return humanized.lower()

def format_function_signature(name: str, params: str, return_type: Optional[str] = None) -> str:
    """
    Format a function signature as human-readable text.
    
    Args:
        name: Function name
        params: Function parameters
        return_type: Function return type
        
    Returns:
        Human-readable function signature
    """
    humanized_name = humanize_identifier(name)
    
    # Format parameters
    params_list = [p.strip() for p in params.split(',') if p.strip()]
    param_names = []
    
    for param in params_list:
        # Handle type annotations
        if ':' in param:
            param_name = param.split(':')[0].strip()
        # Handle default values
        elif '=' in param:
            param_name = param.split('=')[0].strip()
        else:
            param_name = param.strip()
        
        # Remove self or cls
        if param_name not in ['self', 'cls']:
            param_names.append(humanize_identifier(param_name))
    
    # Build the signature
    signature = f"Function '{humanized_name}'"
    
    if param_names:
        signature += f" takes {', '.join(param_names)}"
    
    if return_type and return_type.lower() != 'none':
        signature += f" and returns {return_type}"
    
    return signature

def textify_code(chunk: Dict[str, Any]) -> str:
    """
    Convert code to natural language text for embedding, similar to the Qdrant example.
    
    Args:
        chunk: Document dictionary with content and metadata
        
    Returns:
        Natural language representation of the code
    """
    # Extract code properties
    code_type = None
    if "function_name" in chunk and chunk["function_name"]:
        code_type = "Function"
        name = chunk["function_name"]
    elif "class_name" in chunk and chunk["class_name"]:
        code_type = "Class"
        name = chunk["class_name"] 
    else:
        code_type = "File"
        name = chunk.get("file_name", "")
    
    # Get human-readable name
    human_name = humanize_identifier(name)
    
    # Extract docstring if available
    docstring = ""
    content = chunk.get("content", "")
    docstring_match = re.search(r'"""(.*?)"""', content, re.DOTALL)
    if docstring_match:
        docstring = f"that does {docstring_match.group(1).strip()} "
    
    # Build context
    context = ""
    module_name = chunk.get("module_name", "")
    if module_name:
        context += f"module {humanize_identifier(module_name)} "
    
    file_name = chunk.get("file_name", "")
    if file_name:
        context += f"file {file_name} "
    
    class_name = chunk.get("class_name", "")
    if class_name and code_type == "Function":  # For methods
        context = f"defined in class {humanize_identifier(class_name)} {context}"
    
    # Generate simplified representation of the code content
    signature = content.split("\n")[0]
    
    # Build the full text representation
    text_representation = f"{code_type} {human_name} {docstring}defined as {signature} {context}"
    
    # Remove special characters and convert to space-separated tokens
    tokens = re.split(r"\W", text_representation)
    tokens = filter(lambda x: x, tokens)
    
    return " ".join(tokens)

def code_to_text(code: str, file_path: str, metadata: Dict[str, Any]) -> str:
    """
    Convert code to natural language text for embedding.
    
    Args:
        code: Code content
        file_path: Path to the code file
        metadata: Additional metadata about the code
        
    Returns:
        Natural language representation of the code
    """
    file_name = os.path.basename(file_path)
    file_ext = os.path.splitext(file_name)[1].lower()
    
    # Create a chunk dictionary for the textify function
    chunk = {
        "content": code,
        "file_path": file_path,
        "file_name": file_name,
        "file_type": file_ext.lstrip("."),
        "module_name": metadata.get("module_name", ""),
        "class_name": metadata.get("class_name", ""),
        "function_name": metadata.get("function_name", ""),
    }
    
    # Use the improved textify function
    return textify_code(chunk)

def extract_keywords_from_code(code: str, max_keywords: int = 10) -> List[str]:
    """
    Extract significant keywords from code.
    
    Args:
        code: Code content
        max_keywords: Maximum number of keywords to extract
        
    Returns:
        List of extracted keywords
    """
    # Remove comments (both block and line comments)
    code_without_comments = re.sub(r'""".*?"""', '', code, flags=re.DOTALL)
    code_without_comments = re.sub(r"'''.*?'''", '', code_without_comments, flags=re.DOTALL)
    code_without_comments = re.sub(r'#.*$', '', code_without_comments, flags=re.MULTILINE)
    
    # Find function and class names
    function_names = re.findall(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)', code_without_comments)
    class_names = re.findall(r'class\s+([a-zA-Z_][a-zA-Z0-9_]*)', code_without_comments)
    
    # Find variable names and other identifiers
    identifiers = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]{2,})\b', code_without_comments)
    
    # Remove common words and Python keywords
    stop_words = {
        'self', 'None', 'True', 'False', 'class', 'def', 'return', 'import',
        'from', 'as', 'if', 'else', 'elif', 'for', 'while', 'in', 'not', 'and',
        'or', 'is', 'with', 'try', 'except', 'finally', 'raise', 'assert',
        'print', 'lambda', 'pass', 'break', 'continue', 'del', 'global', 'nonlocal'
    }
    
    # Prioritize function and class names
    keywords = []
    for name in function_names + class_names:
        if name not in stop_words and name not in keywords:
            keywords.append(name)
    
    # Add other identifiers
    for ident in identifiers:
        if ident not in stop_words and ident not in keywords:
            keywords.append(ident)
    
    return keywords[:max_keywords] 