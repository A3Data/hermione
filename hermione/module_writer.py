import click
import os
import shutil
from datetime import datetime
import json
from jinja2 import Template
import hermione

import importlib
import pkgutil

plugins_dirs = [
    importlib.import_module(name).__path__[0]
    for finder, name, ispkg
    in pkgutil.iter_modules()
    if name.startswith('hermione_ml_')
]


def get_modules():

    def get_module_info(module_name, parent_dir):
        config_file = os.path.join( parent_dir, f'{module_name}.json')
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                data = json.load(f)
            return module_name, data['info']
        else:
            return module_name, ''

    parent_dirs = [hermione.__path__[0]] + plugins_dirs
    module_paths = {}
    module_info = {}

    for parent_dir in reversed(parent_dirs):
        for module_name in next(os.walk(os.path.join(parent_dir, 'module_templates')))[1]:
            module_paths[module_name] = os.path.join(parent_dir, 'module_templates', module_name)
            module_info[module_name] = get_module_info(module_name, parent_dir)
    
    return module_paths, module_info

def modules_autocomplete(ctx, args, incomplete):
    """Get list of modules available for installation

    Args:
        ctx:
        args:
        incomplete:
    """

    _, module_info = get_modules()    
    return [module_info[key] for key in module_info.keys() if (incomplete in key) and key[:2] != '__']


def write_module(LOCAL_PATH, module_name, autoconfirm = False , custom_inputs  = {}, ):
    """Copy files to project

    Args:
        LOCAL_PATH (str): Local Path
        project_name (str): Project Name
    """

    def get_inputs(input_info, autoconfirm = False):
        input_data = {}
        for info in input_info:
            if autoconfirm:
                input_data[info[0]] = info[1]
            else:
                input_data[info[0]] = click.prompt(info[2], default=info[1])
        return input_data

    module_paths, _ = get_modules()
    module_path = module_paths[module_name]
    config_file = os.path.join( os.path.dirname(module_path), f'{module_name}.json')

    # Load config file
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            data = json.load(f)
    else:
        data = {'input_info':[]}

    # Request input data
    data['inputs'] = get_inputs(data['input_info'], autoconfirm)
    data['inputs'].update(custom_inputs)

    # Process files

    files_to_create = {}
    files_to_append = {}
    dirs_to_create = []

    for dir_path, dirs, files in os.walk(module_path):

        rel_path = os.path.relpath(dir_path, module_path)
        local_dir_path = os.path.join(LOCAL_PATH, rel_path)

        for d in dirs:
            local_d_path = os.path.join(local_dir_path, d)
            if not os.path.exists(local_d_path):
                dirs_to_create.append(local_d_path)

        for f in files:

            if f == '.hermioneignore':
                continue

            outputText = open(os.path.join(dir_path, f), 'r', encoding='latin-1').read()
            if '.tpl.' in f:
                template = Template(outputText)
                outputText = template.render(**data)

            local_f_path = os.path.join(local_dir_path, f.replace('.tpl.','.'))
            if os.path.exists(local_f_path):
                files_to_append[local_f_path] = outputText
            else:
                files_to_create[local_f_path] = outputText

    # Show changes to user and ask for permission

    if (not autoconfirm):
        click.secho('DIRECTORIES TO BE CREATED:', fg='green', bold = True)
        for key in dirs_to_create:
            click.secho(key, fg='green')

        click.secho('FILES TO BE CREATED:', fg='green', bold = True)
        for key, text in files_to_create.items():
            click.secho(key, fg='green')
            click.secho(
                '\t+  '.join(('\n'+text).splitlines(True)) + '\n'
            )

        click.secho('FILES TO BE APPENDED:', fg='yellow', bold = True)
        for key, text in files_to_append.items():
            click.secho(key, fg='yellow')
            click.secho(
                '\t+  '.join(('\n'+text).splitlines(True)) + '\n'
            )

    if (not autoconfirm) and (not click.confirm('Do you want to continue?', default = True, abort=True)):
        return

    # Create and append files and dirs

    for directory in dirs_to_create:
        if not os.path.exists(directory):
            os.makedirs(directory)

    for key, text in files_to_create.items():
        directory = os.path.dirname(key)
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(key,'wb') as f:
            f.write(text.encode('latin-1'))

    for key, text in files_to_append.items():
        with open(key,'ab') as f:
            f.write(('\n\n'+text).encode('latin-1'))
