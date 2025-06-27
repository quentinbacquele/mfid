import os
import json
import shutil
from pathlib import Path

class ConfigManager:
    """
    Manages user configurations and settings for the MFID application
    
    Features:
    - Save/load user configurations
    - Default profiles for different use cases
    - Import/export settings
    - Access to recent files and directories
    
    Example usage:
    ```
    config = ConfigManager()
    config.set('body_detection', 'confidence_threshold', 0.5)
    threshold = config.get('body_detection', 'confidence_threshold')
    ```
    """
    
    def __init__(self):
        """Initialize ConfigManager with default configuration"""
        self.config_dir = os.path.join(str(Path.home()), '.mfid')
        self.config_file = os.path.join(self.config_dir, 'config.json')
        self.profiles_dir = os.path.join(self.config_dir, 'profiles')
        
        # Create config directory if it doesn't exist
        os.makedirs(self.config_dir, exist_ok=True)
        os.makedirs(self.profiles_dir, exist_ok=True)
        
        # Default configuration
        self.default_config = {
            'general': {
                'recent_files': [],
                'recent_directories': [],
                'last_save_location': str(Path.home()),
                'theme': 'dark',
                'default_detection_output': os.path.join(str(Path.home()), 'mfid', 'output', 'detections'),
                'default_training_output': os.path.join(str(Path.home()), 'mfid', 'output', 'training_models')
            },
            'body_detection': {
                'model': 'm',
                'confidence_threshold': 0.5,
                'iou_threshold': 0.7,
                'preferred_device': 'cpu',
                'show_detections': True,
                'save_videos': True,
                'save_txt': False,
                'frame_skip': 0
            },
            'face_detection': {
                'confidence_threshold': 0.5,
                'iou_threshold': 0.45,
                'save_coordinates': True,
                'save_full_frames': True,
                'frame_skip': 5
            },
            'identity': {
                'model': 'best_m_id.pt',
                'confidence_threshold': 0.5
            },
            'training_settings': {
                'default_epochs': 100,
                'default_img_size': 640
            }
        }
        
        # Load or create configuration
        self.config = self.load_config()
    
    def load_config(self):
        """Load configuration from file or create with defaults if it doesn't exist"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                # Make sure all default sections and options exist
                for section, options in self.default_config.items():
                    if section not in config:
                        config[section] = options
                    else:
                        for option, value in options.items():
                            if option not in config[section]:
                                config[section][option] = value
                return config
            except (json.JSONDecodeError, IOError) as e:
                print(f"Error loading config: {e}. Using defaults.")
                return self.default_config.copy()
        else:
            # Create new config file with defaults
            config = self.default_config.copy()
            self.save_config(config)
            return config
    
    def save_config(self, config=None):
        """Save current configuration to file"""
        if config is None:
            config = self.config
        
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=4)
            return True
        except IOError as e:
            print(f"Error saving config: {e}")
            return False
    
    def get(self, section, option, default=None):
        """Get a configuration value"""
        if section in self.config and option in self.config[section]:
            return self.config[section][option]
        return default
    
    def set(self, section, option, value):
        """Set a configuration value and save the configuration"""
        if section not in self.config:
            self.config[section] = {}
        
        self.config[section][option] = value
        return self.save_config()
    
    def add_recent_file(self, file_path):
        """Add a file to the recent files list"""
        if not os.path.exists(file_path):
            return False
        
        recent_files = self.config['general']['recent_files']
        
        # Remove if exists and add to beginning
        if file_path in recent_files:
            recent_files.remove(file_path)
        
        # Add to the beginning and limit to 10 entries
        recent_files.insert(0, file_path)
        self.config['general']['recent_files'] = recent_files[:10]
        return self.save_config()
    
    def add_recent_directory(self, directory):
        """Add a directory to the recent directories list"""
        if not os.path.isdir(directory):
            return False
        
        recent_dirs = self.config['general']['recent_directories']
        
        # Remove if exists and add to beginning
        if directory in recent_dirs:
            recent_dirs.remove(directory)
        
        # Add to the beginning and limit to 10 entries
        recent_dirs.insert(0, directory)
        self.config['general']['recent_directories'] = recent_dirs[:10]
        return self.save_config()
    
    def get_recent_files(self):
        """Get list of recent files"""
        return self.config['general']['recent_files']
    
    def get_recent_directories(self):
        """Get list of recent directories"""
        return self.config['general']['recent_directories']
    
    def save_profile(self, profile_name):
        """Save current configuration as a named profile"""
        if not profile_name:
            return False
        
        profile_path = os.path.join(self.profiles_dir, f"{profile_name}.json")
        
        try:
            with open(profile_path, 'w') as f:
                json.dump(self.config, f, indent=4)
            return True
        except IOError as e:
            print(f"Error saving profile: {e}")
            return False
    
    def load_profile(self, profile_name):
        """Load a named profile"""
        if not profile_name:
            return False
        
        profile_path = os.path.join(self.profiles_dir, f"{profile_name}.json")
        
        if not os.path.exists(profile_path):
            return False
        
        try:
            with open(profile_path, 'r') as f:
                profile_config = json.load(f)
            
            # Update configuration
            self.config.update(profile_config)
            self.save_config()
            return True
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error loading profile: {e}")
            return False
    
    def get_profiles(self):
        """Get list of available profiles"""
        profiles = []
        for filename in os.listdir(self.profiles_dir):
            if filename.endswith('.json'):
                profiles.append(os.path.splitext(filename)[0])
        return profiles
    
    def export_config(self, export_path):
        """Export configuration to a file"""
        try:
            shutil.copy2(self.config_file, export_path)
            return True
        except IOError as e:
            print(f"Error exporting config: {e}")
            return False
    
    def import_config(self, import_path):
        """Import configuration from a file"""
        if not os.path.exists(import_path):
            return False
        
        try:
            with open(import_path, 'r') as f:
                new_config = json.load(f)
            
            # Validate that this is a valid config file
            for section in self.default_config:
                if section not in new_config:
                    print(f"Invalid config file: missing section {section}")
                    return False
            
            # Update configuration
            self.config = new_config
            self.save_config()
            return True
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error importing config: {e}")
            return False
    
    def reset_to_defaults(self):
        """Reset configuration to defaults"""
        self.config = self.default_config.copy()
        return self.save_config() 