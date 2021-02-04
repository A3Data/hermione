import click
import os
import re
import sys
from .writer import *
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
        file_source = 'file_text'
    elif implemented in ['no', 'n', 'No', 'NO', 'N']:
        file_source = 'not_implemented_file_text'
    
    click.echo(f"Creating project {project_name}")
    # ML Folder
    os.makedirs(os.path.join(LOCAL_PATH, project_name, 'src/ml/model'))
    os.makedirs(os.path.join(LOCAL_PATH, project_name, 'src/ml/preprocessing'))
    os.makedirs(os.path.join(LOCAL_PATH, project_name, 'src/ml/visualization'))
    os.makedirs(os.path.join(LOCAL_PATH, project_name, 'src/ml/notebooks'))
    #os.makedirs(os.path.join(LOCAL_PATH, project_name, 'src/ml/analysis'))
    os.makedirs(os.path.join(LOCAL_PATH, project_name, 'src/ml/data_source'))
    # Config folder
    os.makedirs(os.path.join(LOCAL_PATH, project_name, 'src/config'))
    # API folder
    os.makedirs(os.path.join(LOCAL_PATH, project_name, 'src/api'))
    # Tests folder
    os.makedirs(os.path.join(LOCAL_PATH, project_name, 'src/tests'))
    # Output 
    os.makedirs(os.path.join(LOCAL_PATH, project_name, 'output'))
    # Data folder
    os.makedirs(os.path.join(LOCAL_PATH, project_name, 'data/raw'))

    # Write config file
    write_config_file(LOCAL_PATH, project_name)
    #write_logging_file(LOCAL_PATH, project_name)
    write_requirements_txt(LOCAL_PATH, project_name, file_source)
    write_gitignore(LOCAL_PATH, project_name, file_source)
    write_readme_file(LOCAL_PATH, project_name, file_source)
    #write_application_config(LOCAL_PATH, project_name)
    write_src_util_file(LOCAL_PATH, project_name, file_source)
    write_wsgi_file(LOCAL_PATH, project_name, file_source)
    write_app_file(LOCAL_PATH, project_name, file_source)
    write_visualization_file(LOCAL_PATH, project_name, file_source)
    write_visualization_streamlit_file(LOCAL_PATH, project_name, file_source)
    write_normalization_file(LOCAL_PATH, project_name, file_source)
    write_preprocessing_file(LOCAL_PATH, project_name, file_source)
    write_text_vectorizer_file(LOCAL_PATH, project_name, file_source)
    write_metrics_file(LOCAL_PATH, project_name, file_source)
    write_trainer_file(LOCAL_PATH, project_name, file_source)
    write_wrapper_file(LOCAL_PATH, project_name, file_source)
    write_data_source_base_file(LOCAL_PATH, project_name, file_source)
    write_database_file(LOCAL_PATH, project_name, file_source)
    write_spreadsheet_file(LOCAL_PATH, project_name, file_source)
    #write_cluster_analysis_file(LOCAL_PATH, project_name)
    #write_vif_file(LOCAL_PATH, project_name)
    write_example_notebook_file(LOCAL_PATH, project_name, file_source)
    write_train_dot_py(LOCAL_PATH, project_name, file_source)
    write_predict(LOCAL_PATH, project_name, file_source)
    write_dockerfile(LOCAL_PATH, project_name, file_source)

    if implemented in ['yes', 'ye', 'y', 'Yes', 'YES', 'Y']:
        write_titanic_data(LOCAL_PATH, project_name, file_source)
        write_myrequests_file(LOCAL_PATH, project_name, file_source)
        

    write_test_file(LOCAL_PATH, project_name, file_source)
    write_test_readme(LOCAL_PATH, project_name, file_source)

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
