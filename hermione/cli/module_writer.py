import click
import os
import shutil
from datetime import datetime
import json
from jinja2 import Template
import hermione

import importlib
import pkgutil
import shutil

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
    #os.makedirs(LOCAL_PATH)
    template_folder = os.path.join(hermione.__path__[0], "project_template")
    print(template_folder)
    shutil.copytree(template_folder, LOCAL_PATH, dirs_exist_ok=True)
    src_folder = os.path.join(hermione.__path__[0], "core/")
    shutil.copytree(src_folder, os.path.join(LOCAL_PATH, "src"), dirs_exist_ok=True)

