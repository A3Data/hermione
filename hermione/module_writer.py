import click
import os
import shutil
from datetime import datetime
import json
import jinja2
import rony

def get_modules(ctx, args, incomplete):
    """Get list of modules available for installation

    Args:
        ctx:
        args:
        incomplete:
    """

    def get_module_info(module_folder):
        config_file = os.path.join(rony.__path__[0], 'module_templates', module_folder,'config.json')
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                data = json.load(f)
            return module_folder, data['info']
        else:
            return module_folder, ''
    
    module_folders = next(os.walk(os.path.join(rony.__path__[0], 'module_templates')))[1]
    return [get_module_info(m) for m in module_folders if incomplete in m]


def write_module(LOCAL_PATH, module_name, autoconfirm = False ):
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

    module_path = os.path.join(rony.__path__[0], 'module_templates', module_name)
    config_file = os.path.join(module_path, 'config.json')

    # Load config file
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            data = json.load(f)
    else:
        data = {'input_info':[]}

    # Request input data
    data['inputs'] = get_inputs(data['input_info'], autoconfirm)

    # Process files
    templateLoader = jinja2.FileSystemLoader(searchpath=module_path)
    templateEnv = jinja2.Environment(loader=templateLoader)
    files_to_create = {}
    files_to_append = {}

    for dir_path,_,files in os.walk(module_path):

        rel_path = os.path.relpath(dir_path, module_path)
        local_dir_path = os.path.join(LOCAL_PATH, rel_path)

        for f in files:
            if f=='config.json':
                continue

            if os.path.splitext(f)[1] == '.rtpl':
                template = templateEnv.get_template(os.path.join(rel_path, f))
                outputText = template.render(**data)
            else:
                outputText = open(os.path.join(dir_path, f), 'r').read()

            local_f_path = os.path.join(local_dir_path, f.replace('.rtpl',''))
            if os.path.exists(local_f_path):
                files_to_append[local_f_path] = outputText
            else:
                files_to_create[local_f_path] = outputText

    # Show changes to user and ask for permission

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

    # Create and append files
    for key, text in files_to_create.items():
        directory = os.path.dirname(key)
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(key,'w') as f:
            f.write(text)

    for key, text in files_to_append.items():
        with open(key,'a') as f:
            f.write('\n\n'+text)