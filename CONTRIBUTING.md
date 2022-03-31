# Contributing to Hermione

## Getting started

### Forking repository

The first step in contributing to Herminone is to fork the repository and clone the fork to your machine.

### Remove any Hermione installations

```bash
pip uninstall hermione-ml
```

### "Install" Hermione in development mode

Inside your fork directory, use the following command:

```bash
python setup.py develop 
```

This command will "install" Hermione in [development mode](https://setuptools.readthedocs.io/en/latest/userguide/development_mode.html).
That way, the Hermione cli command will point directly to the source code. Any changes you make to the code will be immediately available.

## Adding new modules

All Hermione module templates are defined as code in the ["templating"](hermione/templating/) module. To add a new module, you need to add a new script in the ["TEMPLATES"](hermione/templating/projects.py) dictionary. Build scripts can be created using the ["epoximise"](hermione/templating/epoximise/) module, an internal library for defining template-as-code. An example of a build script can be seen below:

```python
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
```

### Using jinja template files

Herminone can process [jinja](https://jinja.palletsprojects.com/en/2.11.x/) templates. To let Herminone know the file needs to be interpreted as a jinja template, you just need to add `.tpl` before the file extension (`sample.py` would become `sample.tpl.py`).

All the user inputs are passed to the template as keys of `inputs` variable. For example:

#### **`README.tpl.md`**
``` jinja
# {{ inputs['project_name'] }}

Project started in {{ inputs['project_start_date'] }}.


**Please, complete here information on using and testing this project.**
```

After the template above is processed, it will become:

#### **`README.md`**
``` md
# My Project

Project started in 01/01/21.


**Please, complete here information on using and testing this project.**
```

### Empty Folders and .gitignore files

Empty folders added to the templates, would not be added to Hermione's git repository.

To deal with that a empty file named `.hermioneignore` needs to be added to empty folders. Those files will be ignored by Hermione and will not be added to the user's project

Similarly `.gitignore` files inside the templates would be processed by git and other files could be mistakenly ignored by git. To deal with that the `.gitignore` can be renamed to `.tpl.gitignore`. They will be ignored by git, but will be renamed by Hermione when populating the user's project.