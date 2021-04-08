import click
import os
import re
import sys
from .writer import *
from .module_writer import modules_autocomplete, write_module
from .__init__ import __version__ as version

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
    
    click.echo(f"Creating project {project_name}")


    custom_inputs = {
        'project_name':project_name, 
        "project_start_date": datetime.today().strftime("%B %d, %Y")
        }
    os.makedirs(os.path.join(LOCAL_PATH, project_name))
    if is_imp:
        write_module(os.path.join(LOCAL_PATH, project_name), '__IMPLEMENTED_BASE__', True, custom_inputs)
    else:
        write_module(os.path.join(LOCAL_PATH, project_name), '__NOT_IMPLEMENTED_BASE__', True, custom_inputs)

    print(f'Creating virtual environment {project_name}_env')
    os.chdir(project_name)
    env_name = f"{project_name}_env"
    os.system(f"python -m venv {env_name}")

    # Create git repo
    os.system('git init')
    print("A git repository was created. You should add your files and make your first commit.\n")
    


@cli.command()
def train():
    """
    Execute the script in train.py. One should be at src directory
    """
    if not os.path.exists('./train.py'):
        click.echo("You gotta have an src/train.py file")
    else:
        os.system('python ./train.py')
        print("\nModel trained. For MLFlow logging control, type:\nmlflow ui\nand visit http://localhost:5000/")


@cli.command()
def predict():
    """
    Execute the script in predict.py to make batch predictions. 
    One should be at src directory
    """
    if not os.path.exists('./predict.py'):
        click.echo("You gotta have an src/predict.py file")
    else:
        print("Making predictions: ")
        os.system('python ./predict.py')


@click.argument('image_name')
@click.option('-t', '--tag', 'tag', default='latest', show_default=True)
@cli.command()
def build(image_name, tag):
    """
    Build a docker image with given image_name. Only run if you have docker installed.
    One should be at the root directory.
    """
    if not os.path.exists('src/Dockerfile'):
        click.echo("You gotta have an src/Dockerfile file. You must be at the project's root folder.")
    else:
        os.system(f'docker build -f src/Dockerfile -t {image_name}:{tag} .')


@click.argument('image_name')
@click.option('-t', '--tag', 'tag', default='latest', show_default=True)
@cli.command()
def run(image_name, tag):
    """
    Run a container with given image_name. 
    Only run if you have docker installed.
    """
    if not os.path.exists('src/Dockerfile'):
        click.echo("You gotta have an src/Dockerfile file. You must be at the project's root folder.")
    else:
        os.system(f'docker run --rm -p 5000:5000 {image_name}:{tag}')


@click.argument("module_name", type = click.STRING, autocompletion=modules_autocomplete)
@cli.command()
@click.option('-y','--autoconfirm', is_flag=True)
def add_module(module_name, autoconfirm):
    write_module(LOCAL_PATH, module_name, autoconfirm)