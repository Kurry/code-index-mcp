#!/usr/bin/env python3
import functools
import inspect
from typing import Any, Callable, Dict, List, Optional, TypeVar, cast, get_type_hints

T = TypeVar("T", bound=Callable[..., Any])

def function_tool(func: T) -> T:
    """
    Decorator to mark a function as a tool for the OpenAI Assistant API.
    This adds the necessary metadata for OpenAI's function calling.
    
    Args:
        func: The function to decorate
        
    Returns:
        The decorated function
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)
    
    # Add metadata to the function
    wrapper._is_function_tool = True  # type: ignore
    
    return cast(T, wrapper)

def agent(
    name: str,
    description: str,
    instructions: str,
    tools: List[Callable[..., Any]]
) -> Dict[str, Any]:
    """
    Create an agent definition for OpenAI Assistant API.
    
    Args:
        name: Agent name
        description: Agent description
        instructions: Agent instructions
        tools: List of tools available to the agent
        
    Returns:
        Agent definition
    """
    # Validate that all tools are decorated with @function_tool
    for tool in tools:
        if not hasattr(tool, "_is_function_tool"):
            raise ValueError(f"Tool {tool.__name__} must be decorated with @function_tool")
    
    # Create agent definition
    agent_definition = {
        "name": name,
        "description": description,
        "instructions": instructions,
        "tools": tools
    }
    
    return agent_definition 