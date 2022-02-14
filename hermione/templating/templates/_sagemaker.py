from ..epoximise.templating import *
from ._project_artifacts import get_artefact_full_path
from ._hooks import CreateVirtualEnv, InitializeGitRepository


def build_sagemaker_template():
    with ProjectTemplate("sagemaker_template") as template:
        for file_name in ["build_and_push.sh", "README.md", "requirements.txt"]:
            CopyFile(get_artefact_full_path("sagemaker", file_name))
        for module in ["data", "inference", "processor", "src", "train"]:
            CopyDir(get_artefact_full_path("sagemaker", module))
        CreateVirtualEnv()
        InitializeGitRepository()
    return template
