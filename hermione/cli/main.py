import click
import os
import sys
import json
from datetime import datetime
from hermione.templating import create_project, ProjectDirAlreadyExistsException
from hermione._version import __version__ as version

LOCAL_PATH = os.getcwd()

# Correct LOCAL_PATH in case of empty spaces #21

logo = r"""
 _                         _                  
| |__   ___ _ __ _ __ ___ (_) ___  _ __   ___ 
| '_ \ / _ \ '__| '_ ` _ \| |/ _ \| '_ \ / _ \
| | | |  __/ |  | | | | | | | (_) | | | |  __/
|_| |_|\___|_|  |_| |_| |_|_|\___/|_| |_|\___|
v{}
""".format(version)


@click.group()
def cli():
    pass

@cli.command()
def info():
    """
    Checks that hermione is correctly installed
    """
    click.echo(logo)


def start_new_project(project_name, template, context_data):
    try:
        create_project(
            LOCAL_PATH,
            project_name,
            template,
            context_data=context_data,
            output_blueprint=True
        )
    except ProjectDirAlreadyExistsException:
        click.echo(
            f'Error: Cannot create new project. Folder {project_name} already exists'
        )
        sys.exit(-1)

    print(f'Creating virtual environment {project_name}_env')
    os.chdir(project_name)
    env_name = f"{project_name}_env"
    os.system(f"python -m venv {env_name}")
    if template in ["local_implemented", "local_not_implemented"]:
        os.system(f"python -m venv {env_name}")
        os.system(f"{project_name}_env/bin/python -m pip install -e .")

    # Create git repo
    os.system('git init')
    print("A git repository was created. You should add your files and make your first commit.\n")


@cli.command()
@click.argument('project_name')
@click.option('-imp', '--implemented', 'implemented', prompt='Do you want to start with an implemented example (recommended) [y/n]?', 
            default='y', show_default=True)
def new(project_name, implemented):
    """
    Create a new hermione project
    """
    if implemented in ['yes', 'ye', 'y', 'Yes', 'YES', 'Y']:
        is_imp = True
    else:
        is_imp = False

    if is_imp:
        option = click.prompt('Do you want to start with: \n\t(1) Sagemaker \n\t(2) Local version \n', type=int, default=2)
        if option == 1:
            template = "sagemaker"
        else:
            template = "local_implemented"
    else:
        template = "local_not_implemented"

    start_new_project(
        project_name,
        template,
        context_data={
            "project_start_date": datetime.now().strftime("%B %d, %Y")
        }
    )


@cli.command()
@click.argument('blueprint_file_path', type=click.Path(exists=True, dir_okay=False, readable=True))
def from_blueprint(blueprint_file_path):
    """
    Create a new hermione project from blueprint
    """
    with open(blueprint_file_path) as json_file:
        blueprint = json.load(json_file)
        start_new_project(
            blueprint["project_name"],
            blueprint["template"],
            blueprint["context_data"]
        )



@cli.command()
def train():
    """
    Execute the script in train.py. One should be at src directory
    """
    if not os.path.exists('./scripts/train.py'):
        click.echo("You gotta have an scripts/train.py file")
    else:
        os.system('python ./scripts/train.py')
        print("\nModel trained. For MLFlow logging control, type:\nmlflow ui\nand visit http://localhost:5000/")


@cli.command()
def predict():
    """
    Execute the script in predict.py to make batch predictions. 
    One should be at src directory
    """
    if not os.path.exists('scripts/predict.py'):
        click.echo("You gotta have an scripts/predict.py file. You must be at the project's root folder.")
    else:
        print("Making predictions: ")
        os.system('python ./scripts/predict.py')


@click.argument('image_name')
@click.option('-t', '--tag', 'tag', default='latest', show_default=True)
@cli.command()
def build(image_name, tag):
    """
    Build a docker image with given image_name. Only run if you have docker installed.
    One should be at the root directory.
    """
    if not os.path.exists('Dockerfile'):
        click.echo("You gotta have an Dockerfile file. You must be at the project's root folder.")
    else:
        os.system(f'docker build -f Dockerfile -t {image_name}:{tag} .')


@click.argument('image_name')
@click.option('-t', '--tag', 'tag', default='latest', show_default=True)
@cli.command()
def run(image_name, tag):
    """
    Run a container with given image_name. 
    Only run if you have docker installed.
    """
    if not os.path.exists('Dockerfile'):
        click.echo("You gotta have an Dockerfile file. You must be at the project's root folder.")
    else:
        os.system(f'docker run --rm -p 5000:5000 {image_name}:{tag}')
