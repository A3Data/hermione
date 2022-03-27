from ..epoximise.templating import *
from ._project_artifacts import get_artefact_full_path, get_hermione_src_dir
from ._hooks import CreateVirtualEnv, InstallProjectLocally, InitializeGitRepository


def build_starter_template():
    with ProjectTemplate("starter_template") as template:
        for file_name in [
            ".gitignore",
            "README.tpl.md",
            "requirements.txt",
            "setup.tpl.py",
            "Dockerfile",
        ]:
            CopyFile(get_artefact_full_path("shared", file_name))
        with CreateDir("config"):
            CopyFile(get_artefact_full_path("shared", "config/config.tpl.json"))
        for module in ["dashboard", "data", "pyspark", "tests"]:
            CopyDir(get_artefact_full_path("shared", module))
        for module in ["api", "notebooks", "scripts"]:
            CopyDir(get_artefact_full_path("starter", module))
        with CopyDir(get_hermione_src_dir(), "src"):
            CopyDir(get_artefact_full_path("shared", "src/data_source"))
            CopyDir(
                get_artefact_full_path("starter", "src/data_source"),
                merge_if_exists=True,
            )
        CreateVirtualEnv()
        InstallProjectLocally()
        InitializeGitRepository()
    return template
