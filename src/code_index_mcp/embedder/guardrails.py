#!/usr/bin/env python3
from typing import Dict, Any
from agents import GuardrailFunctionOutput, InputGuardrail

async def validate_query_guardrail(ctx, agent, input_data):
    """
    Validate that search queries are appropriate.
    
    Args:
        ctx: Context
        agent: Agent instance
        input_data: User input
        
    Returns:
        Guardrail output
    """
    # Check if the input is asking about inappropriate content
    inappropriate_terms = [
        "hack", "crack", "steal", "illegal", "exploit", "bypass", "porn",
        "offensive", "attack", "vulnerability", "malware", "virus"
    ]
    
    # Simple check for inappropriate terms
    has_inappropriate_terms = any(term in input_data.lower() for term in inappropriate_terms)
    
    if has_inappropriate_terms:
        return GuardrailFunctionOutput(
            output_info={
                "query": input_data,
                "is_appropriate": False,
                "reason": "The query contains terms that suggest inappropriate intent."
            },
            tripwire_triggered=True,
        )
    
    return GuardrailFunctionOutput(
        output_info={
            "query": input_data,
            "is_appropriate": True,
        },
        tripwire_triggered=False,
    )

def create_query_guardrail() -> InputGuardrail:
    """
    Create an input guardrail for query validation.
    
    Returns:
        InputGuardrail instance
    """
    return InputGuardrail(guardrail_function=validate_query_guardrail)

async def validate_directory_guardrail(ctx, agent, input_data):
    """
    Validate that directory paths are safe to process.
    
    Args:
        ctx: Context
        agent: Agent instance
        input_data: User input
        
    Returns:
        Guardrail output
    """
    import os
    import re
    
    # Check if the input contains directory paths
    dir_pattern = r'(?:^|\s)((?:/|\.{1,2}/|~/)(?:[^"\'\s]|/)+)(?:\s|$)'
    paths = re.findall(dir_pattern, input_data)
    
    if not paths:
        return GuardrailFunctionOutput(
            output_info={
                "input": input_data,
                "is_safe": True,
            },
            tripwire_triggered=False,
        )
    
    # Check each path for safety
    unsafe_paths = []
    sensitive_dirs = ["/etc", "/var", "/usr/bin", "/bin", "/sbin", "/usr/sbin", "/sys", "/tmp"]
    
    for path in paths:
        # Expand path
        expanded_path = os.path.abspath(os.path.expanduser(path))
        
        # Check if it's a sensitive system directory
        if any(expanded_path.startswith(sensitive_dir) for sensitive_dir in sensitive_dirs):
            unsafe_paths.append(path)
    
    if unsafe_paths:
        return GuardrailFunctionOutput(
            output_info={
                "input": input_data,
                "is_safe": False,
                "unsafe_paths": unsafe_paths,
                "reason": "The input contains paths to sensitive system directories."
            },
            tripwire_triggered=True,
        )
    
    return GuardrailFunctionOutput(
        output_info={
            "input": input_data,
            "is_safe": True,
        },
        tripwire_triggered=False,
    )

def create_directory_guardrail() -> InputGuardrail:
    """
    Create an input guardrail for directory validation.
    
    Returns:
        InputGuardrail instance
    """
    return InputGuardrail(guardrail_function=validate_directory_guardrail) 