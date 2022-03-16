import click
import os
import sys
import json
from datetime import datetime
from hermione.templating import (
    create_project,
    TEMPLATES,
    ProjectDirAlreadyExistsException,
)

TEMPLATE_OPTIONS = [template_option["name"] for template_option in TEMPLATES]

LOCAL_PATH = os.getcwd()


@click.group()
def project():
    pass


@project.command()
@click.argument("project_name")
@click.option("--template", type=click.Choice(TEMPLATE_OPTIONS, case_sensitive=False))
def new(project_name, template):
    """
    Create a new hermione project
    """
    while not template:
        selection_prompt_str = "Please select one of the following templates \n"
        for i, template_name in enumerate(TEMPLATE_OPTIONS):
            selection_prompt_str += f"\t({i}) {template_name} \n"
        selection_prompt_str += "Option"
        selected_option = click.prompt(selection_prompt_str, type=int, default=0)
        if selected_option >= len(TEMPLATE_OPTIONS):
            click.echo(f"Error: {selected_option} its not a valid option")
        else:
            template = TEMPLATE_OPTIONS[selected_option]
    try:
        create_project(
            LOCAL_PATH,
            project_name,
            template.lower(),
            context_data={"project_start_date": datetime.now().strftime("%B %d, %Y")},
        )
    except ProjectDirAlreadyExistsException:
        click.echo(
            f"Error: Cannot create new project. Folder {project_name} already exists"
        )
        sys.exit(-1)


@project.command()
@click.argument(
    "blueprint_file_path", type=click.Path(exists=True, dir_okay=False, readable=True)
)
def replay(blueprint_file_path):
    """
    Create a new hermione project from blueprint
    """
    with open(blueprint_file_path) as json_file:
        blueprint = json.load(json_file)
        try:
            create_project(
                LOCAL_PATH,
                blueprint["project_name"],
                blueprint["template"],
                blueprint["context_data"],
            )
        except ProjectDirAlreadyExistsException:
            click.echo(
                f'Error: Cannot create new project. Folder {blueprint["project_name"]} already exists'
            )
            sys.exit(-1)
