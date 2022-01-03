from ..epoximise.templating import *
from ._utils import ArtifactsPathLoader


def build_local_implemented_template():
    artifacts = ArtifactsPathLoader()
    with ProjectTemplate("local_implemented_template") as template:
        RenderTemplateFiles(
            artifacts.get_paths("shared", [".gitignore", "README.tpl.md", "requirements.txt", "setup.tpl.py"])
        )
        with CreateDir("config"):
            RenderTemplateFile(artifacts.get_path("shared", "config/config.tpl.json"))
        for module in ["dashboard", "data", "pyspark"]:
            RenderTemplateDir(artifacts.get_path("shared", module))
        for module in ["api", "notebooks", "scripts"]:
            RenderTemplateDir(artifacts.get_path("local_implemented", module))
        with RenderTemplateDir(artifacts.get_path("shared", "tests")):
            RenderTemplateFile(artifacts.get_path("local_implemented", "tests/test_project.py"))
        with RenderTemplateDir(artifacts.get_hermione_src_dir(), "src"):
            RenderTemplateDir(artifacts.get_path("shared", "src/data_source"))
            RenderTemplateDir(artifacts.get_path("local_implemented", "src/data_source"), merge_if_exists=True)
        RenderTemplateFile(artifacts.get_path("shared", "Dockerfile"))
    return template
