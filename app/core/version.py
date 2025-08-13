"""
Version management for RAG Backend Lucy.
This file is automatically updated by git hooks during commits.
"""

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

# Version file path relative to project root
VERSION_FILE = Path(__file__).parent.parent.parent / "VERSION.json"

# Default version info
DEFAULT_VERSION = {
    "version": "0.1.0",
    "build_date": "",
    "git_commit": "",
    "git_branch": "",
    "build_number": 0,
    "environment": "development"
}

def get_version_info() -> Dict[str, Any]:
    """
    Get version information from VERSION.json file.
    Falls back to default if file doesn't exist.
    """
    try:
        if VERSION_FILE.exists():
            with open(VERSION_FILE, 'r') as f:
                version_data = json.load(f)
                # Update build date to current time
                version_data["build_date"] = datetime.now().isoformat()
                return version_data
    except (json.JSONDecodeError, IOError) as e:
        print(f"Warning: Could not read version file: {e}")
    
    return DEFAULT_VERSION.copy()

def get_version_string() -> str:
    """Get a formatted version string."""
    version_info = get_version_info()
    return f"v{version_info['version']}-{version_info['build_number']}"

def get_full_version_info() -> Dict[str, Any]:
    """Get complete version information."""
    version_info = get_version_info()
    
    # Add additional runtime information
    version_info.update({
        "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
        "platform": os.sys.platform,
        "build_date": datetime.now().isoformat()
    })
    
    return version_info

# Version constants for easy access
VERSION_INFO = get_version_info()
VERSION_STRING = get_version_string()

