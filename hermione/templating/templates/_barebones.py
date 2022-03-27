from ..epoximise.templating import *
from ._project_artifacts import get_artefact_full_path, get_hermione_src_dir
from ._hooks import CreateVirtualEnv, InstallProjectLocally, InitializeGitRepository


def build_barebones_template():
    with ProjectTemplate("barebones_template") as template:
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
        CopyDir(get_artefact_full_path("shared", "tests"))
        with CreateDir("data"):
            for data_dir in ["raw", "processed", "output"]:
                CreateDir(data_dir)
        for module in ["api", "notebooks", "scripts"]:
            CopyDir(get_artefact_full_path("barebones", module))
        with CopyDir(get_hermione_src_dir(), "src"):
            CopyDir(get_artefact_full_path("shared", "src/data_source"))
            CopyDir(
                get_artefact_full_path("barebones", "src/data_source"),
                merge_if_exists=True,
            )
        CreateVirtualEnv()
        InstallProjectLocally()
        InitializeGitRepository()
    return template
