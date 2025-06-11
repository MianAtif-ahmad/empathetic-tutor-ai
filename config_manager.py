# config_manager.py
import yaml
import os
from typing import Dict, Any

class ConfigManager:
    def __init__(self, config_file: str = "config.yaml"):
        self.config_file = config_file
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as file:
                    config = yaml.safe_load(file)
                    print(f"✅ Configuration loaded from {self.config_file}")
                    return config
            else:
                print(f"⚠️  Config file {self.config_file} not found, using defaults")
                return self._get_default_config()
        except Exception as e:
            print(f"❌ Error loading config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        return {
            'ai': {
                'provider': 'local',
                'fallback_provider': 'local',
                'enable_fallback': True,
                'openai': {'model': 'gpt-4', 'max_tokens': 200, 'temperature': 0.7},
                'anthropic': {'model': 'claude-3-sonnet-20240229', 'max_tokens': 200}
            },
            'system': {'debug': True, 'log_ai_responses': True, 'enhanced_empathy': False},
            'database': {'name': 'empathetic_tutor.db'},
            'frustration': {'low_threshold': 3.0, 'high_threshold': 7.0},
            'empathy': {'personalization_enabled': False}
        }
    
    def get_ai_provider(self) -> str:
        return self.config.get('ai', {}).get('provider', 'local')
    
    def get_fallback_provider(self) -> str:
        return self.config.get('ai', {}).get('fallback_provider', 'local')
    
    def is_fallback_enabled(self) -> bool:
        return self.config.get('ai', {}).get('enable_fallback', True)
    
    def get_openai_config(self) -> Dict[str, Any]:
        return self.config.get('ai', {}).get('openai', {})
    
    def get_anthropic_config(self) -> Dict[str, Any]:
        return self.config.get('ai', {}).get('anthropic', {})
    
    def is_enhanced_empathy_enabled(self) -> bool:
        return self.config.get('empathy', {}).get('personalization_enabled', False)
    
    def set_ai_provider(self, provider: str):
        valid_providers = ['openai', 'anthropic', 'local', 'disabled']
        if provider not in valid_providers:
            raise ValueError(f"Invalid provider. Must be one of: {valid_providers}")
        self.config['ai']['provider'] = provider
        print(f"✅ AI provider changed to: {provider}")
    
    def print_current_config(self):
        print("\n🔧 CURRENT CONFIGURATION:")
        print(f"   AI Provider: {self.get_ai_provider()}")
        print(f"   Fallback Enabled: {self.is_fallback_enabled()}")
        print(f"   Enhanced Empathy: {self.is_enhanced_empathy_enabled()}")

# Global configuration instance
config = ConfigManager()