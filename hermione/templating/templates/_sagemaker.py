from ..epoximise.templating import *
from ._utils import ArtifactsPathLoader


def build_sagemaker_template():
    artifacts = ArtifactsPathLoader()
    with ProjectTemplate("sagemaker_template") as template:
        RenderTemplateFiles(
            artifacts.get_paths("sagemaker", ["build_and_push.sh", "README.md", "requirements.txt"])
        )
        for module in ["data", "inference", "processor", "src", "train"]:
            RenderTemplateDir(artifacts.get_path("sagemaker", module))
    return template

