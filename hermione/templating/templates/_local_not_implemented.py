from ..epoximise.templating import *
from ._utils import ArtifactsPathLoader


def build_local_not_implemented_template():
    artifacts = ArtifactsPathLoader()
    with ProjectTemplate("local_not_implemented_template") as template:
        RenderTemplateFiles(
            artifacts.get_paths("shared", [".gitignore", "README.tpl.md", "requirements.txt", "setup.tpl.py"])
        )
        with CreateDir("config"):
            RenderTemplateFile(artifacts.get_path("shared", "config/config.tpl.json"))
        with RenderTemplateDir(artifacts.get_path("shared", "tests")):
            RenderTemplateFile(artifacts.get_path("local_not_implemented", "tests/test_project.py"))
        with CreateDir("data"):
            for data_dir in ["raw", "processed", "output"]:
                with CreateDir(data_dir):
                    CreateFile(".gitkeep")
        with RenderTemplateDir(artifacts.get_path("local_not_implemented", "api")):
            RenderTemplateFile(artifacts.get_path("shared", "api/wsgi.py"))
        for module in ["notebooks", "scripts"]:
            RenderTemplateDir(artifacts.get_path("local_not_implemented", module))
        with RenderTemplateDir(artifacts.get_hermione_src_dir(), "src"):
            RenderTemplateDir(artifacts.get_path("shared", "src/data_source"))
            RenderTemplateDir(artifacts.get_path("local_not_implemented", "src/data_source"), merge_if_exists=True)
        RenderTemplateFile(artifacts.get_path("shared", "Dockerfile"))
    return template

