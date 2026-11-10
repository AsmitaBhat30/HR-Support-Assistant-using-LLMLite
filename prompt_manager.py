import json
from pathlib import Path
from typing import Dict, Any
from jinja2 import Environment, FileSystemLoader, select_autoescape

class PromptManager:
    def __init__(self, template_dir: str = "prompts"):
        self.env = Environment(
            loader=FileSystemLoader(template_dir),
            autoescape=select_autoescape(['xml']) # Security against injection in XML tags
        )

    def render_prompt(self, template_name: str, **kwargs) -> str:
        template = self.env.get_template(template_name)
        return template.render(**kwargs)

    def parse_response(self, response_text: str) -> Dict[str, Any]:
        """Safe JSON parsing helper with markdown fallback."""
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            cleaned = response_text.replace("```json", "").replace("```", "").strip()
            return json.loads(cleaned)