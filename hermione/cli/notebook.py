import os
import sys
import json
import click
import hermione
import uuid
import shutil

HERMIONE_DIR = hermione.__path__[0]
NOTEBOOK_TEMPLATES_DIR = os.path.normpath(
    os.path.join(HERMIONE_DIR, "data", "notebook_templates")
)


@click.group()
def notebook():
    pass


@notebook.command()
@click.option("--name", "notebook_name")
@click.option("--template", "template")
def new(notebook_name, template):
    """
    Create a new jupyter notebook from template
    """
    with open(os.path.join(NOTEBOOK_TEMPLATES_DIR, "mapping.json"), "r") as file:
        template_mapping = json.load(file)
    if template not in template_mapping:
        click.echo(
            f"Error: Cannot create new notebook. No registered template with nane {template}"
        )
        sys.exit(-1)
    template_path = os.path.join(NOTEBOOK_TEMPLATES_DIR, template_mapping[template])
    if not notebook_name.endswith(".ipynb"):
        notebook_name += ".ipynb"
    if os.path.exists(notebook_name):
        click.echo(
            f"Error: Cannot create new notebook. Theres already a notebook named {notebook_name}"
        )
        sys.exit(-1)
    shutil.copy(template_path, notebook_name)


@notebook.command()
@click.argument(
    "template_file", type=click.Path(exists=True, dir_okay=False, readable=True)
)
@click.argument("template_name")
def add_template(template_file, template_name):
    """
    Register a new notebook template
    """
    with open(os.path.join(NOTEBOOK_TEMPLATES_DIR, "mapping.json"), "r") as file:
        template_mapping = json.load(file)
    if template_name in template_mapping:
        click.echo(
            f"Error: Cannot create new template. Theres is already a template named {template_name}"
        )
        sys.exit(-1)
    if not template_file.endswith(".ipynb"):
        click.echo(
            f"Error: Cannot create new template. Template file must have end with .ipynb"
        )
        sys.exit(-1)

    template_filename = f"{uuid.uuid4().hex}_{os.path.basename(template_file)}"
    template_path = os.path.join(NOTEBOOK_TEMPLATES_DIR, template_filename)

    shutil.copy(template_file, template_path)
    template_mapping[template_name] = template_filename
    with open(os.path.join(NOTEBOOK_TEMPLATES_DIR, "mapping.json"), "w") as file:
        json.dump(template_mapping, file, indent=4)


@notebook.command()
def list_templates():
    """
    List all templates
    """
    with open(os.path.join(NOTEBOOK_TEMPLATES_DIR, "mapping.json"), "r") as file:
        template_mapping = json.load(file)
    print("\n".join(template_mapping.keys()))
