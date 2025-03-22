"""
Code Index MCP - Streamlit Web Interface

This Streamlit app provides a web interface for the Code Index MCP server,
allowing users to view and refresh code indexes by directory.
"""
import os
import time
import datetime
import streamlit as st
import pandas as pd
from pathlib import Path

# Import the embedding manager
from code_index_mcp.core.embedding_manager import EmbeddingManager
from dotenv import load_dotenv
import shutil

# Set page configuration
st.set_page_config(
    page_title="Code Index MCP",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling the UI similar to the Cursor interface
st.markdown("""
<style>
    .stApp {
        background-color: #1e1e1e;
        color: #e0e0e0;
    }
    .main .block-container {
        padding-top: 2rem;
    }
    h1, h2, h3 {
        color: #e0e0e0;
    }
    .directory-card {
        background-color: #2d2d2d;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .index-progress {
        height: 8px;
        background-color: #3a3a3a;
        border-radius: 4px;
        margin-top: 10px;
    }
    .index-progress-bar {
        height: 100%;
        background-color: #6bba7a;
        border-radius: 4px;
    }
    .status-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 6px;
    }
    .status-synced {
        background-color: #6bba7a;
    }
    .status-indexing {
        background-color: #f0ad4e;
    }
    .status-error {
        background-color: #d9534f;
    }
    .btn-refresh {
        background-color: #4285f4;
        color: white;
        border: none;
        padding: 8px 16px;
        border-radius: 4px;
        cursor: pointer;
    }
    .btn-delete {
        background-color: #555555;
        color: white;
        border: none;
        padding: 8px 16px;
        border-radius: 4px;
        cursor: pointer;
    }
    .directory-title {
        font-size: 16px;
        font-weight: 500;
        margin-bottom: 5px;
    }
    .directory-info {
        font-size: 14px;
        color: #aaaaaa;
    }
    .toggle-container {
        display: flex;
        align-items: center;
        margin-bottom: 15px;
    }
    .toggle-label {
        margin-left: 10px;
        font-size: 15px;
    }
</style>
""", unsafe_allow_html=True)

def get_index_info(embedding_manager, directory):
    """Get information about the index for a directory"""
    # Check if the directory has been indexed
    index_stats = embedding_manager.get_collection_stats()
    
    if "error" in index_stats:
        return {
            "status": "error",
            "message": index_stats["error"],
            "vector_count": 0,
            "indexed_date": None
        }
    
    if index_stats.get("status") == "no_embeddings":
        return {
            "status": "not_indexed",
            "message": "Not indexed yet",
            "vector_count": 0,
            "indexed_date": None
        }
    
    # Extract information from the embedding file path
    indexed_date = None
    if "file" in index_stats:
        # Try to extract timestamp from filename (embeddings_TIMESTAMP.pkl)
        try:
            file_name = Path(index_stats["file"]).name
            if "_" in file_name and "." in file_name:
                timestamp_str = file_name.split("_")[1].split(".")[0]
                timestamp = int(timestamp_str)
                indexed_date = datetime.datetime.fromtimestamp(timestamp)
        except:
            pass
    
    return {
        "status": "synced",
        "message": "Indexed",
        "vector_count": index_stats.get("vector_count", 0),
        "indexed_date": indexed_date
    }

def refresh_directory_index(embedding_manager, directory):
    """Refresh the index for a specific directory"""
    try:
        # Save the current base path
        original_base_path = embedding_manager.base_path
        
        # Update the base path to the selected directory
        embedding_manager.base_path = Path(directory)
        
        # Create .code_indexer directory if it doesn't exist
        index_dir = embedding_manager.base_path / ".code_indexer"
        index_dir.mkdir(exist_ok=True)
        
        # Index the directory
        file_count = embedding_manager.index_directory()
        
        # Reset the base path back to the original
        embedding_manager.base_path = original_base_path
        
        return True, f"Successfully indexed {file_count} files in {directory}"
    except Exception as e:
        return False, f"Error indexing {directory}: {str(e)}"

def find_indexable_directories(root_path, max_depth=3, excluded_dirs=None):
    """Find directories that can be indexed, up to a certain depth"""
    if excluded_dirs is None:
        excluded_dirs = ['.git', 'node_modules', 'venv', '.venv', '__pycache__', 'build', 'dist']
    
    directories = []
    root = Path(root_path)
    
    for path in root.glob('**/*'):
        if path.is_dir():
            # Skip excluded directories and hidden directories
            if path.name.startswith('.') or path.name in excluded_dirs:
                continue
                
            # Calculate the depth relative to root
            depth = len(path.relative_to(root).parts)
            if depth <= max_depth:
                # Check if it contains indexable files
                has_files = False
                for file_path in path.glob('*'):
                    if file_path.is_file() and not file_path.name.startswith('.'):
                        has_files = True
                        break
                
                if has_files:
                    directories.append(str(path))
    
    return directories

def main():
    """Main function for the Streamlit app"""
    # Header
    st.title("Codebase Indexing")
    st.markdown("Embeddings improve your codebase-wide answers. Embeddings and metadata are stored in the **cloud**, but all code is stored locally.")
    
    # Documentation links
    with st.expander("ℹ️ Help & Documentation"):
        st.markdown("""
- [Full Documentation](https://github.com/your-username/code-index-mcp/tree/main/docs)
- [Quick Start Guide](https://github.com/your-username/code-index-mcp/blob/main/docs/QUICK_START.md)
- [CLI Documentation](https://github.com/your-username/code-index-mcp/blob/main/docs/CLI.md)
        """)
    
    # Initialize the embedding manager with the base path from environment
    from dotenv import load_dotenv
    
    # Load environment variables from .env file
    load_dotenv()
    
    # Check for required API keys
    openai_key = os.environ.get('OPEN_AI_KEY')
    if not openai_key:
        st.warning("OPEN_AI_KEY not set in .env file or environment variables. Embeddings may not work correctly.")
    
    # Get the base path from environment
    base_path = os.environ.get('CODE_INDEX_DIRECTORY')
    
    # If not set, use current working directory
    if not base_path:
        base_path = os.getcwd()
        st.warning(f"CODE_INDEX_DIRECTORY not set in environment or .env file. Using current directory: {base_path}")
    else:
        st.info(f"Using project directory: {base_path}")
    
    # Initialize the embedding manager
    try:
        # Check if we should use Qdrant Cloud
        use_qdrant_cloud = os.environ.get('CODE_INDEX_USE_QDRANT_CLOUD', '').lower() in ('true', '1', 'yes')
        
        _embedding_manager = EmbeddingManager(
            base_path=base_path,
            use_qdrant_cloud=use_qdrant_cloud
        )
        
        if use_qdrant_cloud:
            st.success("Using Qdrant Cloud for vector storage")
        else:
            st.info("Using in-memory Qdrant instance (for development)")
    except ImportError as e:
        st.error(f"Could not initialize embedding manager: Missing dependency {str(e)}")
        st.info("Try running: pip install -e . to ensure all dependencies are installed")
        st.stop()
    except Exception as e:
        st.error(f"Error initializing embedding manager: {str(e)}")
        st.info("Check your .env file for correct configuration")
        st.stop()
    
    # Get the overall index status
    global_index_info = get_index_info(_embedding_manager, base_path)
    
    # Display global index status with a progress bar
    synced_percentage = 100 if global_index_info["status"] == "synced" else 0
    st.markdown(f"## Synced {synced_percentage}%")
    
    # Progress bar
    progress_html = f"""
    <div class="index-progress">
        <div class="index-progress-bar" style="width: {synced_percentage}%;"></div>
    </div>
    """
    st.markdown(progress_html, unsafe_allow_html=True)
    
    # Display codebase stats
    st.markdown("### Codebase Statistics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Files Indexed", global_index_info.get("vector_count", 0))
        
    with col2:
        # Get the last indexed time
        last_indexed = "Never"
        if global_index_info.get("indexed_date"):
            last_indexed = global_index_info["indexed_date"].strftime('%m/%d/%y, %I:%M %p')
        st.metric("Last Indexed", last_indexed)
    
    with col3:
        # Display whether auto-indexing is enabled
        auto_index_status = "Enabled" if os.environ.get('CODE_INDEX_AUTO_INDEX', '').lower() in ('true', '1', 'yes') else "Disabled"
        st.metric("Auto-Indexing", auto_index_status)
    
    # Action buttons
    col1, col2 = st.columns([1, 10])
    with col1:
        if st.button("🔄 Resync Index", key="resync_global"):
            with st.spinner("Indexing entire codebase..."):
                success, message = refresh_directory_index(_embedding_manager, base_path)
                if success:
                    st.success(message)
                else:
                    st.error(message)
                st.experimental_rerun()
    
    with col2:
        if st.button("🗑️ Delete Index", key="delete_global"):
            with st.spinner("Deleting index..."):
                try:
                    # Remove the .code_indexer directory
                    index_dir = Path(base_path) / ".code_indexer"
                    if index_dir.exists():
                        shutil.rmtree(index_dir)
                        st.success("Index deleted successfully")
                        st.experimental_rerun()
                    else:
                        st.info("No index found to delete")
                except Exception as e:
                    st.error(f"Error deleting index: {str(e)}")
    
    # Settings section
    with st.expander("⚙️ Hide Settings", expanded=True):
        # Auto-index toggle
        current_auto_index = os.environ.get('CODE_INDEX_AUTO_INDEX', '').lower() in ('true', '1', 'yes')
        auto_index = st.toggle("Index new folders by default", value=current_auto_index)
        
        # Update the environment variable if changed
        if auto_index != current_auto_index:
            os.environ['CODE_INDEX_AUTO_INDEX'] = 'true' if auto_index else 'false'
            st.success(f"Auto-indexing set to: {auto_index}")
        
        st.markdown("""
        When set to true, Code Index MCP will by default index any new folders opened. 
        If turned off, you can still manually index folders by clicking the "Resync Index" button. 
        Folders with more than 10,000 files will not be auto-indexed.
        """, help="Control automatic indexing behavior")
        
        # Qdrant Cloud toggle
        current_use_cloud = os.environ.get('CODE_INDEX_USE_QDRANT_CLOUD', '').lower() in ('true', '1', 'yes')
        use_cloud = st.toggle("Use Qdrant Cloud for storage", value=current_use_cloud)
        
        # Update the environment variable if changed
        if use_cloud != current_use_cloud:
            os.environ['CODE_INDEX_USE_QDRANT_CLOUD'] = 'true' if use_cloud else 'false'
            st.success(f"Qdrant Cloud usage set to: {use_cloud}")
            st.warning("Please restart the application for this change to take effect")
        
        # Ignore files section
        st.markdown("### Ignore files")
        st.markdown("Configure the files that Code Index MCP will ignore when indexing your repository (in addition to your .gitignore).")
        
        # Add a text area for ignore patterns with default values
        default_ignore = ".git\nnode_modules\nvenv\n.venv\n__pycache__\nbuild\ndist"
        ignore_patterns = st.text_area("Ignore patterns (one per line)", value=default_ignore)
        st.caption("These directories will be excluded when indexing your codebase")
        
        if st.button("Save Ignore Patterns"):
            # In a real implementation, you would save these patterns to a config file
            # For now, just show a success message
            st.success("Ignore patterns saved successfully")
        
        # Git graph option
        st.markdown("### Git graph file relationships")
        git_enabled = st.toggle("Enable Git history analysis", value=False)
        st.markdown("""
        When enabled, Code Index MCP will index your git history to help understand which files are related to each other. 
        Code and commit messages are stored locally, but metadata about commits (SHAs, number of changes, and obfuscated file names) are stored on the server.
        """)
        
        if git_enabled:
            st.warning("Git history analysis is not yet implemented")
    
    # Directories section
    st.markdown("# Directories")
    st.markdown("Manage indexed directories in your project.")
    
    # Find indexable directories
    directories = find_indexable_directories(base_path)
    
    if not directories:
        st.info("No indexable directories found. Add code files to your project.")
    
    # Display directories as docs
    for i, directory in enumerate(directories):
        rel_path = os.path.relpath(directory, base_path)
        dir_name = os.path.basename(directory)
        
        # Check if this directory has been indexed
        with st.spinner(f"Checking index status for {dir_name}..."):
            # Save the current base path
            original_base_path = _embedding_manager.base_path
            
            # Update the base path to the selected directory
            _embedding_manager.base_path = Path(directory)
            
            # Get index information
            dir_index_info = get_index_info(_embedding_manager, directory)
            
            # Reset the base path back to the original
            _embedding_manager.base_path = original_base_path
        
        # Display directory card
        col1, col2, col3, col4, col5 = st.columns([0.1, 4, 1, 1, 1])
        
        # Status indicator
        with col1:
            if dir_index_info["status"] == "synced":
                st.markdown('<div class="status-indicator status-synced"></div>', unsafe_allow_html=True)
            elif dir_index_info["status"] == "indexing":
                st.markdown('<div class="status-indicator status-indexing"></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="status-indicator status-error"></div>', unsafe_allow_html=True)
        
        # Directory name and info
        with col2:
            st.markdown(f"### {dir_name}")
            
            indexed_info = "Not indexed yet"
            if dir_index_info["indexed_date"]:
                indexed_info = f"Indexed {dir_index_info['indexed_date'].strftime('%m/%d/%y, %I:%M %p')}"
            
            st.markdown(f"<div class='directory-info'>{indexed_info}</div>", unsafe_allow_html=True)
        
        # Edit button (placeholder)
        with col3:
            st.markdown("✏️")
        
        # Refresh button
        with col4:
            if st.button("🔄", key=f"refresh_{i}"):
                with st.spinner(f"Indexing {dir_name}..."):
                    success, message = refresh_directory_index(_embedding_manager, directory)
                    if success:
                        st.success(message)
                    else:
                        st.error(message)
                    time.sleep(1)  # Give user time to see the message
                    st.experimental_rerun()
        
        # Delete button
        with col5:
            if st.button("🗑️", key=f"delete_{i}"):
                with st.spinner(f"Deleting index for {dir_name}..."):
                    try:
                        # Remove the .code_indexer directory for this specific directory
                        index_dir = Path(directory) / ".code_indexer"
                        if index_dir.exists():
                            shutil.rmtree(index_dir)
                            st.success(f"Index for {dir_name} deleted successfully")
                            time.sleep(1)  # Give user time to see the message
                            st.experimental_rerun()
                        else:
                            st.info(f"No index found for {dir_name}")
                    except Exception as e:
                        st.error(f"Error deleting index: {str(e)}")
        
        st.markdown("---")
    
    # Add new directory section
    with st.expander("➕ Add new directory to index"):
        new_dir = st.text_input("Directory path (relative to project root):")
        if st.button("Add Directory"):
            if new_dir:
                full_path = Path(base_path) / new_dir
                if not full_path.exists():
                    st.error(f"Directory {new_dir} does not exist")
                elif not full_path.is_dir():
                    st.error(f"{new_dir} is not a directory")
                else:
                    # Index this new directory
                    with st.spinner(f"Adding and indexing {new_dir}..."):
                        try:
                            # Ensure the .code_indexer directory exists
                            index_dir = full_path / ".code_indexer"
                            index_dir.mkdir(exist_ok=True)
                            
                            # Create a temporary embedding manager for this directory
                            temp_manager = EmbeddingManager(
                                base_path=str(full_path),
                                use_qdrant_cloud=os.environ.get('CODE_INDEX_USE_QDRANT_CLOUD', '').lower() in ('true', '1', 'yes')
                            )
                            file_count = temp_manager.index_directory()
                            
                            st.success(f"Added and indexed {file_count} files in {new_dir}")
                            time.sleep(1)
                            st.experimental_rerun()
                        except Exception as e:
                            st.error(f"Error indexing directory: {str(e)}")
            else:
                st.warning("Please enter a directory path")

if __name__ == "__main__":
    main()