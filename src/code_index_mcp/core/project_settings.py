"""
Project Settings Management

This module provides functionality for managing project settings and persistent data
for the Code Index MCP server.
"""
import os
import json
import shutil
import pickle
from datetime import datetime

class ProjectSettings:
    """管理專案設定和索引資料的類"""
    
    SETTINGS_DIR = ".code_indexer"
    CONFIG_FILE = "config.json"
    INDEX_FILE = "file_index.pickle"
    CACHE_FILE = "content_cache.pickle"
    
    def __init__(self, base_path):
        """初始化專案設定
        
        Args:
            base_path (str): 專案的基本路徑
        """
        self.base_path = base_path
        self.settings_path = os.path.join(base_path, self.SETTINGS_DIR)
        self.ensure_settings_dir()
    
    def ensure_settings_dir(self):
        """確保設定目錄存在"""
        if not os.path.exists(self.settings_path):
            os.makedirs(self.settings_path)
            # 創建一個 README.md 文件，說明該目錄的用途
            readme_path = os.path.join(self.settings_path, "README.md")
            with open(readme_path, 'w') as f:
                f.write("# Code Indexer Cache Directory\n\nThis directory contains cached data for the Code Index MCP tool:\n\n- `config.json`: Project configuration\n- `file_index.pickle`: Index of project files\n- `content_cache.pickle`: Cached file contents\n\nThese files are automatically generated and should not be committed to version control.\n")
    
    def get_config_path(self):
        """獲取配置文件的路徑"""
        return os.path.join(self.settings_path, self.CONFIG_FILE)
    
    def get_index_path(self):
        """獲取索引文件的路徑"""
        return os.path.join(self.settings_path, self.INDEX_FILE)
    
    def get_cache_path(self):
        """獲取緩存文件的路徑"""
        return os.path.join(self.settings_path, self.CACHE_FILE)
    
    def _get_timestamp(self):
        """獲取當前時間戳"""
        return datetime.now().isoformat()
    
    def save_config(self, config):
        """保存配置數據
        
        Args:
            config (dict): 配置數據
        """
        config_path = self.get_config_path()
        # 添加時間戳
        config['last_updated'] = self._get_timestamp()
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        return config
    
    def load_config(self):
        """加載配置數據
        
        Returns:
            dict: 配置數據，如果文件不存在則返回空字典
        """
        config_path = self.get_config_path()
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, UnicodeDecodeError):
                # 如果文件損壞，返回空字典
                return {}
        return {}
    
    def save_index(self, file_index):
        """保存文件索引
        
        Args:
            file_index (dict): 文件索引數據
        """
        index_path = self.get_index_path()
        with open(index_path, 'wb') as f:
            pickle.dump(file_index, f)
    
    def load_index(self):
        """加載文件索引
        
        Returns:
            dict: 文件索引數據，如果文件不存在則返回空字典
        """
        index_path = self.get_index_path()
        if os.path.exists(index_path):
            try:
                with open(index_path, 'rb') as f:
                    return pickle.load(f)
            except (pickle.PickleError, EOFError):
                # 如果文件損壞，返回空字典
                return {}
        return {}
    
    def save_cache(self, content_cache):
        """保存內容緩存
        
        Args:
            content_cache (dict): 內容緩存數據
        """
        cache_path = self.get_cache_path()
        with open(cache_path, 'wb') as f:
            pickle.dump(content_cache, f)
    
    def load_cache(self):
        """加載內容緩存
        
        Returns:
            dict: 內容緩存數據，如果文件不存在則返回空字典
        """
        cache_path = self.get_cache_path()
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except (pickle.PickleError, EOFError):
                # 如果文件損壞，返回空字典
                return {}
        return {}
    
    def clear(self):
        """清除所有設定和緩存文件"""
        if os.path.exists(self.settings_path):
            # 保留 .gitignore 文件
            gitignore_path = os.path.join(self.settings_path, ".gitignore")
            has_gitignore = os.path.exists(gitignore_path)
            gitignore_content = None
            
            if has_gitignore:
                with open(gitignore_path, 'r') as f:
                    gitignore_content = f.read()
            
            # 刪除資料夾中的所有文件
            for filename in os.listdir(self.settings_path):
                file_path = os.path.join(self.settings_path, filename)
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            
            # 恢復 .gitignore 文件
            if has_gitignore:
                with open(gitignore_path, 'w') as f:
                    f.write(gitignore_content)
    
    def get_stats(self):
        """獲取設定目錄的統計信息
        
        Returns:
            dict: 包含文件大小和更新時間的字典
        """
        stats = {
            'exists': os.path.exists(self.settings_path),
            'files': {}
        }
        
        if stats['exists']:
            for filename in [self.CONFIG_FILE, self.INDEX_FILE, self.CACHE_FILE]:
                file_path = os.path.join(self.settings_path, filename)
                if os.path.exists(file_path):
                    file_stats = os.stat(file_path)
                    stats['files'][filename] = {
                        'size_bytes': file_stats.st_size,
                        'last_modified': datetime.fromtimestamp(file_stats.st_mtime).isoformat()
                    }
        
        return stats
