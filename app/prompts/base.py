from typing import Dict, Any, Optional
from jinja2 import Template, Environment, meta
import json
import logging

logger = logging.getLogger(__name__)

class PromptTemplate:
    """Base class for prompt templates with validation and rendering capabilities."""
    
    def __init__(self, template: str, variables: Optional[Dict[str, Any]] = None):
        """
        Initialize a prompt template.
        
        Args:
            template (str): The template string
            variables (Dict[str, Any], optional): Default variables for the template
        """
        self.template = template
        self.variables = variables or {}
        self._env = Environment()
        self._template = self._env.from_string(template)
        self._required_variables = self._get_required_variables()
        
    def _get_required_variables(self) -> set:
        """Get the set of required variables from the template."""
        ast = self._env.parse(self.template)
        return meta.find_undeclared_variables(ast)
    
    def validate_variables(self, variables: Dict[str, Any]) -> bool:
        """
        Validate that all required variables are provided.
        
        Args:
            variables (Dict[str, Any]): The variables to validate
            
        Returns:
            bool: True if all required variables are provided
        """
        provided_vars = set(variables.keys())
        missing_vars = self._required_variables - provided_vars
        if missing_vars:
            logger.warning(f"Missing required variables: {missing_vars}")
            return False
        return True
    
    def render(self, **kwargs) -> str:
        """
        Render the template with the given variables.
        
        Args:
            **kwargs: Variables to use in rendering
            
        Returns:
            str: The rendered template
            
        Raises:
            ValueError: If required variables are missing
        """
        variables = {**self.variables, **kwargs}
        if not self.validate_variables(variables):
            raise ValueError(f"Missing required variables: {self._required_variables - set(variables.keys())}")
        return self._template.render(**variables)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the template to a dictionary for storage."""
        return {
            "template": self.template,
            "variables": self.variables,
            "required_variables": list(self._required_variables)
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PromptTemplate':
        """Create a template from a dictionary."""
        return cls(
            template=data["template"],
            variables=data.get("variables", {})
        ) 