"""
File watcher for auto-indexing
"""
import os
import sys
import time
from pathlib import Path
from watchdog.events import FileSystemEventHandler

class FileChangeHandler(FileSystemEventHandler):
    """Handle file changes for auto-indexing."""
    
    def __init__(self, embedding_manager):
        """Initialize the file change handler.
        
        Args:
            embedding_manager: The embedding manager to use for indexing
        """
        self.embedding_manager = embedding_manager
        self.last_modified_times = {}
        self.debounce_time = 2.0  # seconds
    
    def on_modified(self, event):
        """Process modified file event."""
        if event.is_directory:
            return
            
        # Debounce to avoid multiple indexing for rapid file changes
        path = event.src_path
        current_time = time.time()
        
        # Skip if this file was recently processed
        if path in self.last_modified_times:
            if current_time - self.last_modified_times[path] < self.debounce_time:
                return
        
        # Update the modification time
        self.last_modified_times[path] = current_time
        
        # Check if this is a file we should index
        rel_path = os.path.relpath(path, str(self.embedding_manager.base_path))
        _, ext = os.path.splitext(path)
        
        # Skip hidden files and directories
        if any(part.startswith('.') for part in Path(rel_path).parts):
            return
            
        # Skip if extension is not in supported extensions (if any are defined)
        if (self.embedding_manager.supported_extensions and 
            ext.lstrip('.') not in self.embedding_manager.supported_extensions):
            return
            
        print(f"File modified: {rel_path}, triggering re-indexing...", file=sys.stderr)
        # Re-index the entire directory
        # Could be optimized to just update this file's embeddings
        self.embedding_manager.index_directory()
    
    def on_deleted(self, event):
        """Process deleted file event."""
        if event.is_directory:
            return
            
        # Trigger re-indexing when a file is deleted
        rel_path = os.path.relpath(event.src_path, str(self.embedding_manager.base_path))
        _, ext = os.path.splitext(event.src_path)
        
        # Skip if extension is not in supported extensions (if any are defined)
        if (self.embedding_manager.supported_extensions and 
            ext.lstrip('.') not in self.embedding_manager.supported_extensions):
            return
            
        print(f"File deleted: {rel_path}, triggering re-indexing...", file=sys.stderr)
        # Re-index the entire directory
        self.embedding_manager.index_directory()
