import click
import os
import sys
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
@click.option('-imp', '--implemented', 'implemented', prompt='Do you want to start with an implemented example? [y/n] Default:', 
            default='n', show_default=True)
def new(project_name, python_version, implemented):
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
    #write_endpoints_file(LOCAL_PATH, project_name)
    write_requirements_txt(LOCAL_PATH, project_name, file_source)
    write_gitignore(LOCAL_PATH, project_name, file_source)
    write_readme_file(LOCAL_PATH, project_name, file_source)
    #write_application_file(LOCAL_PATH, project_name)
    #write_application_config(LOCAL_PATH, project_name)
    write_src_util_file(LOCAL_PATH, project_name, file_source)
    #write_wsgi_file(LOCAL_PATH, project_name)
    write_visualization_file(LOCAL_PATH, project_name, file_source)
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

    if implemented in ['yes', 'ye', 'y', 'Yes', 'YES', 'Y']:
        write_titanic_data(LOCAL_PATH, project_name, file_source)
        

    write_test_file(LOCAL_PATH, project_name, file_source)
    write_test_readme(LOCAL_PATH, project_name, file_source)

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
    if not os.path.exists('./train.py'):
        click.echo("You gotta have an src/train.py file")
    else:
        os.system('python ./train.py')
        print("\nModel trained. For MLFlow logging control, type:\nmlflow ui\nand visit http://localhost:5000/")
    
