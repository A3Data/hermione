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

All Hermione module templates are kept in ("hermione/module_templates/")[hermione/moule_templates] folder.
To add a new module, you just need to add a folder named after the module and add all the files and folders from the module to it.

The files and folders need to respect the exact same folder structure they will have on the user's project.

When a user install your module, all the files will be copied to the project's directory following the same structure of the "module_template" folder.
If any file already exists on user's project, its content will be appended with the template content.

### Module template config file

Optionally, you can add a json file containing some information about the module, and any inputs to be asked to the user:

```json
{
    "info": "Base files with implemented example",
    "input_info": [
        ["project_name", "My Project", "Enter your project name"],
        ["project_start_date", "01/01/21", "Enter the date your project started"]
    ]
}
```

The config file needs to have the same name as the module template folder, with the `.json` extension.

The `info` field of the json is used as a description of the module, when user uses autocompletion.

The `input_info` field of the json contains a list of lists. Each list contains information about one of the module's user inputs:

- 1st element is the name of the input
- 2nd element is the default value
- 3rd element is the text the user will be prompted when being asked for the input


### Using jinja template files

Optionally, you can let Herminone process the template files as [jinja](https://jinja.palletsprojects.com/en/2.11.x/) templates.

To let Herminoe know the file needs to be interpreted as a jinja template, you just need to add `.tpl`before the file extension (`sample.py`would become `sample.tpl.py`).

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
