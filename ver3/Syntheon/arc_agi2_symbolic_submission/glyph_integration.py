#!/usr/bin/env python3
# glyph_integration.py
"""
Integration script for switching to glyph-based rules in the Syntheon system.
This script modifies the main execution flow to use the glyph-based DSL
while maintaining compatibility with the existing codebase.
"""

import os
import sys
import re
import shutil
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Path configuration
MAIN_PY = "main.py"
SYNTHEON_ENGINE = "syntheon_engine.py"
GLYPH_INTERPRETER = "glyph_interpreter.py"
ORIGINAL_RULES = "syntheon_rules.xml"
GLYPH_RULES = "syntheon_rules_glyphs.xml"
BACKUP_DIR = "backups"

def create_backup(file_path):
    """Create a backup of the specified file."""
    if not os.path.exists(BACKUP_DIR):
        os.makedirs(BACKUP_DIR)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(BACKUP_DIR, f"{os.path.basename(file_path)}.{timestamp}")
    
    shutil.copy2(file_path, backup_path)
    logging.info(f"Created backup: {backup_path}")
    
    return backup_path

def modify_main_py():
    """Modify main.py to use the hybrid engine with glyph-based rules."""
    create_backup(MAIN_PY)
    
    with open(MAIN_PY, 'r') as f:
        content = f.read()
    
    # Add import for GlyphInterpreter
    import_pattern = r'from syntheon_engine import SyntheonEngine'
    glyph_import = 'from syntheon_engine import SyntheonEngine\nfrom glyph_interpreter import GlyphInterpreter\nfrom hybrid_engine_adapter import HybridEngine'
    content = re.sub(import_pattern, glyph_import, content)
    
    # Replace engine initialization
    engine_init_pattern = r'engine = SyntheonEngine\(\)\s+engine\.load_rules_from_xml\("syntheon_rules\.xml"\)'
    hybrid_engine_init = '''# Initialize hybrid engine with both rule sets
    engine = HybridEngine(
        rules_xml_path="syntheon_rules_glyphs.xml",
        glyph_rules_xml_path="syntheon_rules_glyphs.xml",
        default_mode="auto"  # auto, performance, or dsl
    )
    
    # Log engine initialization
    logging.info("Hybrid engine initialized with both rule sets")
    logging.info("Using adaptive rule selection for optimal accuracy")'''
    
    content = re.sub(engine_init_pattern, hybrid_engine_init, content)
    
    # Write modified content back to file
    with open(MAIN_PY, 'w') as f:
        f.write(content)
    
    logging.info("Modified main.py to use the hybrid engine")

def patch_syntheon_engine():
    """Add compatibility methods to SyntheonEngine for hybrid operation."""
    create_backup(SYNTHEON_ENGINE)
    
    with open(SYNTHEON_ENGINE, 'r') as f:
        content = f.read()
    
    # Add compatibility methods at the end of the class
    class_end_pattern = r'(\s+@staticmethod\s+def _color_swapping.*?\s+return out\n)'
    
    compatibility_methods = r'''\1
    # Compatibility methods for hybrid operation
    def use_glyph_dsl(self, enable=True):
        """Enable or disable glyph DSL mode."""
        self.use_dsl = enable
        return self
        
    def apply_rule_hybrid(self, name, grid, **kwargs):
        """Apply rule with hybrid mode selection."""
        # This would normally select between performance and DSL modes
        # but for now just forwards to the standard implementation
        return self.apply_rule(name, grid, **kwargs)
'''
    
    content = re.sub(class_end_pattern, compatibility_methods, content, flags=re.DOTALL)
    
    # Add DSL mode flag to __init__
    init_pattern = r'(\s+def __init__\(self\):\s+self\.rules_meta = \{\})'
    init_with_dsl = r'\1\n        self.use_dsl = False'
    
    content = re.sub(init_pattern, init_with_dsl, content)
    
    # Write modified content back to file
    with open(SYNTHEON_ENGINE, 'w') as f:
        f.write(content)
    
    logging.info("Patched syntheon_engine.py with compatibility methods")

def create_config_file():
    """Create a configuration file for glyph mode settings."""
    config_content = """# Glyph DSL Configuration
# Controls how the hybrid engine operates

[engine]
# Mode options: "auto", "performance", "dsl"
default_mode = "auto"

# Performance thresholds
dsl_slowdown_threshold = 1.5  # Maximum acceptable DSL slowdown factor
grid_size_threshold = 10      # Grid size above which performance mode is preferred

[rules]
# Rules file paths
performance_rules = "syntheon_rules.xml"
glyph_rules = "syntheon_rules_glyphs.xml"

[logging]
# Logging settings
log_mode_selection = true
log_performance_stats = true
"""
    
    with open("glyph_config.ini", 'w') as f:
        f.write(config_content)
    
    logging.info("Created glyph_config.ini")

def main():
    """Main execution function."""
    print("Syntheon Glyph DSL Integration")
    print("==============================")
    print("This script will modify the system to use glyph-based rules.")
    print("Backups of modified files will be created in the 'backups' directory.")
    
    # Check if required files exist
    required_files = [MAIN_PY, SYNTHEON_ENGINE, GLYPH_INTERPRETER, ORIGINAL_RULES, GLYPH_RULES]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"Error: The following required files are missing: {', '.join(missing_files)}")
        return 1
    
    # Perform integration steps
    try:
        # Create backup directory
        if not os.path.exists(BACKUP_DIR):
            os.makedirs(BACKUP_DIR)
        
        # Modify files
        modify_main_py()
        patch_syntheon_engine()
        create_config_file()
        
        print("\nIntegration completed successfully.")
        print("The system is now configured to use the hybrid engine with glyph-based rules.")
        print("\nRecommended next steps:")
        print("1. Run the system with 'python main.py' to verify accuracy")
        print("2. Adjust settings in glyph_config.ini as needed")
        print("3. To revert changes, restore files from the 'backups' directory")
        
        return 0
    
    except Exception as e:
        print(f"Error during integration: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
