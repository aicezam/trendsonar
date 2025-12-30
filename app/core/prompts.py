import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
from app.core.config import get_settings
from app.core.logger import logger

DATA_DIR = Path("data")
PROMPTS_FILE = DATA_DIR / "prompts.json"
DEFAULTS_FILE = Path(__file__).parent / "prompts_defaults.json"

class PromptManager:
    def __init__(self):
        self._prompts: Dict[str, Dict[str, str]] = {}
        self.load_prompts()

    def _load_defaults(self) -> Dict[str, Dict[str, str]]:
        """从 JSON 文件加载默认提示词"""
        try:
            if not DEFAULTS_FILE.exists():
                logger.warning(f"默认提示词文件不存在: {DEFAULTS_FILE}")
                return {}
            with open(DEFAULTS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"加载默认提示词失败: {e}")
            return {}

    def load_prompts(self):
        """加载提示词，如果文件不存在则创建默认文件"""
        defaults = self._load_defaults()

        if not PROMPTS_FILE.exists():
            if defaults:
                self._save_prompts(defaults)
                self._prompts = defaults
            else:
                self._prompts = {}
        else:
            try:
                with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
                    user_prompts = json.load(f)
                
                # 以默认配置为基础，合并用户配置
                # 1. 复制默认配置作为起点 (包含所有系统定义的 key)
                final_prompts = defaults.copy()
                
                modified = False
                
                # 2. 遍历用户配置，覆盖默认值
                deprecated_keys = ["report_brief_keyword", "report_brief_global"]
                
                for key, user_val in user_prompts.items():
                    # 跳过废弃的 key
                    if key in deprecated_keys:
                        modified = True
                        continue
                        
                    if key in final_prompts:
                        # 如果 key 是系统定义的，使用用户的配置覆盖
                        # 但强制同步 group 字段（确保分组最新）
                        default_group = final_prompts[key].get("group")
                        if user_val.get("group") != default_group:
                            user_val["group"] = default_group
                            modified = True
                        
                        final_prompts[key] = user_val
                    else:
                        # 用户文件中有，但默认配置中没有的 key
                        # 可能是用户自定义的，保留它
                        final_prompts[key] = user_val

                # 3. 检查是否有新增加的默认 key (user_prompts 中缺失的)
                for key in defaults:
                    if key not in user_prompts:
                        modified = True
                
                self._prompts = final_prompts
                
                if modified:
                    self._save_prompts(self._prompts)

            except Exception as e:
                logger.error(f"加载提示词文件失败: {e}，将使用默认提示词")
                self._prompts = defaults

    def _save_prompts(self, prompts: Dict[str, Dict[str, str]]):
        """保存提示词到文件"""
        # 确保目录存在
        if not DATA_DIR.exists():
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            
        with open(PROMPTS_FILE, "w", encoding="utf-8") as f:
            json.dump(prompts, f, ensure_ascii=False, indent=2)

    def get_prompt(self, key: str) -> Dict[str, str]:
        """获取指定 key 的提示词配置"""
        # 如果 key 不在 _prompts 中，尝试从 defaults 重新获取（作为兜底）
        # 但通常 _prompts 已经包含了 defaults
        return self._prompts.get(key, {})

    def get_system_prompt(self, key: str, **kwargs) -> str:
        """获取格式化后的 system prompt"""
        p = self.get_prompt(key)
        raw = p.get("system_prompt", "")
        if not raw:
            return ""
        try:
            return raw.format(**kwargs)
        except KeyError:
            # 如果缺少参数，返回原始字符串或部分格式化
            logger.warning(f"System prompt formatting missing keys for {key}")
            return raw

    def get_user_prompt(self, key: str, **kwargs) -> str:
        """获取格式化后的 user prompt"""
        p = self.get_prompt(key)
        raw = p.get("user_prompt", "")
        if not raw:
            return ""
        try:
            return raw.format(**kwargs)
        except KeyError:
            logger.warning(f"User prompt formatting missing keys for {key}")
            return raw

    def update_prompt(self, key: str, data: Dict[str, str]):
        """更新提示词配置"""
        if key in self._prompts:
            self._prompts[key].update(data)
            self._save_prompts(self._prompts)
        else:
            # 如果是新的 key，则添加
            self._prompts[key] = data
            self._save_prompts(self._prompts)

    def get_all_prompts(self) -> Dict[str, Dict[str, str]]:
        """获取所有提示词"""
        return self._prompts

# 全局单例
prompt_manager = PromptManager()
