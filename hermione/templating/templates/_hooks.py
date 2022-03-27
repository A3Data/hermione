import os
from ..epoximise.templating import Hook
from functools import partial


def create_virtual_env():
    project_name = os.path.basename(os.getcwd())
    env_name = f"{project_name}_env"
    print(f"Creating virtual environment {env_name}")
    os.system(f"python -m venv {env_name}")


def install_project_locally():
    print(f"Installing project locally")
    project_name = os.path.basename(os.getcwd())
    env_name = f"{project_name}_env"
    executable_path_folder = (
        "Scripts" if os.path.exists(os.path.join(env_name, "Scripts")) else "bin"
    )
    executable_path = os.path.join(env_name, executable_path_folder, "python")
    os.system(f"{executable_path} -m pip install -e .")


def initialize_git_repository():
    os.system("git init")
    print(
        "A git repository was created. You should add your files and make your first commit.\n"
    )


CreateVirtualEnv = partial(Hook, hook_callable=create_virtual_env)
InstallProjectLocally = partial(Hook, hook_callable=install_project_locally)
InitializeGitRepository = partial(Hook, hook_callable=initialize_git_repository)
