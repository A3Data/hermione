import click
import os
import sys
from datetime import datetime
from hermione.templating import build_local_not_implemented_template,\
    build_local_implemented_template,\
    build_sagemaker_template
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

    if os.path.exists(project_name):
        click.echo(
            f'Error: Cannot create new project. Folder {project_name} already exists'
        )
        sys.exit(-1)
    click.echo(f"Creating project {project_name}")

    if is_imp:
        option = click.prompt('Do you want to start with: \n\t(1) Sagemaker \n\t(2) Local version \n', type=int, default=2)
        if option == 1:
            template = build_sagemaker_template()
        else:
            template = build_local_implemented_template()
    else:
        template = build_local_not_implemented_template()

    template.create_project(
        LOCAL_PATH, project_name,
        context_data={
            "project_start_date": datetime.now().strftime("%B %d, %Y")
        })

    print(f'Creating virtual environment {project_name}_env')
    os.chdir(project_name)
    env_name = f"{project_name}_env"
    os.system(f"python -m venv {env_name}")
    if template.name in ["local_implemented_template", "local_not_implemented_template"]:
        os.system(f"python -m venv {env_name}")
        os.system(f"{project_name}_env/bin/python -m pip install -e .")

    # Create git repo
    os.system('git init')
    print("A git repository was created. You should add your files and make your first commit.\n")
    

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
