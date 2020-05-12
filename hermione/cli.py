import click
import os
from .writer import *
from .__init__ import __version__ as version

LOCAL_PATH = os.getcwd()

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
@click.option('-p', '--python-version', 'python_version', default='3.7', show_default=True)
def new(project_name, python_version):
    """
    Create a new hermione project
    """
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
    #write_endpoints_file(LOCAL_PATH, project_name)
    write_requirements_txt(LOCAL_PATH, project_name)
    write_gitignore(LOCAL_PATH, project_name)
    write_readme_file(LOCAL_PATH, project_name)
    #write_application_file(LOCAL_PATH, project_name)
    #write_application_config(LOCAL_PATH, project_name)
    write_src_util_file(LOCAL_PATH, project_name)
    #write_wsgi_file(LOCAL_PATH, project_name)
    write_visualization_file(LOCAL_PATH, project_name)
    write_normalization_file(LOCAL_PATH, project_name)
    write_preprocessing_file(LOCAL_PATH, project_name)
    write_text_vectorizer_file(LOCAL_PATH, project_name)
    write_metrics_file(LOCAL_PATH, project_name)
    write_trainer_file(LOCAL_PATH, project_name)
    write_wrapper_file(LOCAL_PATH, project_name)
    write_data_source_base_file(LOCAL_PATH, project_name)
    write_database_file(LOCAL_PATH, project_name)
    write_spreadsheet_file(LOCAL_PATH, project_name)
    #write_cluster_analysis_file(LOCAL_PATH, project_name)
    #write_vif_file(LOCAL_PATH, project_name)
    write_example_notebook_file(LOCAL_PATH, project_name)
    write_titanic_data(LOCAL_PATH, project_name)
    write_train_dot_py(LOCAL_PATH, project_name)

    write_test_file(LOCAL_PATH, project_name)

    print(f'Creating conda virtual environment {project_name}')
    os.system(f"conda create -y --prefix {os.path.join(LOCAL_PATH, project_name)}/{project_name}_env  python={python_version}")

    # Create git repo
    os.chdir(project_name)
    os.system('git init')
    print("A git repository was created. You should add your files and make your first commit.\n")


@cli.command()
def train():
    """
    Execute the script in train.py. One should be at src directory
    """
    os.system('python ./train.py')
    print("\nNew results logged on mlflow. Access typing:\nmlflow ui\nand visit http://localhost:5000/")
