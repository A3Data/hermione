import os
import json
from .templates import build_local_not_implemented_template,\
    build_local_implemented_template,\
    build_sagemaker_template

TEMPLATES = {
    "local_implemented": build_local_implemented_template,
    "local_not_implemented": build_local_not_implemented_template,
    "sagemaker": build_sagemaker_template,
}


def create_project(project_path, project_name, template_name, context_data, output_blueprint=False):
    template_builder = TEMPLATES[template_name]
    template = template_builder()
    template.create_project(project_path, project_name, context_data=context_data)

    if output_blueprint:
        blueprint = {
            "project_name": project_name,
            "template": template_name,
            "context_data": context_data
        }
        with open(os.path.join(project_path, f"{project_name}_blueprint.json"), 'w', encoding='utf-8') as f:
            json.dump(blueprint, f, ensure_ascii=False, indent=4)
