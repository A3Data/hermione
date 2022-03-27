import os
import json
from .templates import (
    build_barebones_template,
    build_starter_template,
    build_sagemaker_template,
)

TEMPLATES = [
    {"name": "starter", "factory": build_starter_template},
    {"name": "barebones", "factory": build_barebones_template},
    {"name": "sagemaker", "factory": build_sagemaker_template},
]


def create_project(
    project_path, project_name, template_name, context_data, output_blueprint=False
):
    template_option = next((x for x in TEMPLATES if x["name"] == template_name), None)
    if not template_name:
        raise ValueError(f"template {template_name} not found")
    template_factory = template_option["factory"]
    template = template_factory()
    template.create_project(project_path, project_name, context_data=context_data)

    if output_blueprint:
        blueprint = {
            "project_name": project_name,
            "template": template_name,
            "context_data": context_data,
        }
        with open(
            os.path.join(project_path, project_name, f"{project_name}_blueprint.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(blueprint, f, ensure_ascii=False, indent=4)
