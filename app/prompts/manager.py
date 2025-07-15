from typing import Dict, Any, Optional
import json
from pathlib import Path
import logging
from .base import PromptTemplate

logger = logging.getLogger(__name__)

class PromptManager:
    """Manager class for handling prompt templates."""
    
    def __init__(self, template_dir: str = "app/prompts/templates"):
        """
        Initialize the prompt manager.
        
        Args:
            template_dir (str): Directory containing template files
        """
        self.template_dir = Path(template_dir)
        self.templates: Dict[str, PromptTemplate] = {}
        self.load_templates()
    
    def load_templates(self) -> None:
        """Load all templates from the template directory."""
        if not self.template_dir.exists():
            logger.warning(f"Template directory {self.template_dir} does not exist")
            return
            
        for template_file in self.template_dir.glob("*.json"):
            try:
                with open(template_file) as f:
                    template_data = json.load(f)
                    template_name = template_data.get("name")
                    if not template_name:
                        logger.warning(f"Template in {template_file} has no name, skipping")
                        continue
                        
                    self.templates[template_name] = PromptTemplate.from_dict(template_data)
                    logger.info(f"Loaded template: {template_name}")
            except Exception as e:
                logger.error(f"Error loading template from {template_file}: {e}")
    
    def get_template(self, name: str) -> PromptTemplate:
        """
        Get a template by name.
        
        Args:
            name (str): Name of the template
            
        Returns:
            PromptTemplate: The requested template
            
        Raises:
            KeyError: If template not found
        """
        if name not in self.templates:
            raise KeyError(f"Template '{name}' not found")
        return self.templates[name]
    
    def render(self, template_name: str, **kwargs) -> str:
        """
        Render a template with the given variables.
        
        Args:
            template_name (str): Name of the template
            **kwargs: Variables to use in rendering
            
        Returns:
            str: The rendered template
            
        Raises:
            KeyError: If template not found
            ValueError: If required variables are missing
        """
        template = self.get_template(template_name)
        return template.render(**kwargs)
    
    def add_template(self, name: str, template: PromptTemplate) -> None:
        """
        Add a new template.
        
        Args:
            name (str): Name for the template
            template (PromptTemplate): The template to add
        """
        self.templates[name] = template
        logger.info(f"Added template: {name}")
    
    def save_template(self, name: str, file_name: Optional[str] = None) -> None:
        """
        Save a template to a JSON file.
        
        Args:
            name (str): Name of the template to save
            file_name (str, optional): Name of the file to save to. If not provided,
                                     uses the template name with .json extension
        """
        template = self.get_template(name)
        file_name = file_name or f"{name}.json"
        file_path = self.template_dir / file_name
        
        try:
            with open(file_path, 'w') as f:
                json.dump(template.to_dict(), f, indent=2)
            logger.info(f"Saved template {name} to {file_path}")
        except Exception as e:
            logger.error(f"Error saving template {name} to {file_path}: {e}")
            raise 