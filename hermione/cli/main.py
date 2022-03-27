import click
from hermione._version import __version__ as version
from .run import run
from .project import project
from .notebook import notebook

logo = r"""
 _                         _                  
| |__   ___ _ __ _ __ ___ (_) ___  _ __   ___ 
| '_ \ / _ \ '__| '_ ` _ \| |/ _ \| '_ \ / _ \
| | | |  __/ |  | | | | | | | (_) | | | |  __/
|_| |_|\___|_|  |_| |_| |_|_|\___/|_| |_|\___|
v{}
""".format(
    version
)


@click.group()
def cli():
    pass


@cli.command()
def info():
    """
    Checks if hermione is correctly installed
    """
    click.echo(logo)


cli.add_command(run)
cli.add_command(project)
cli.add_command(notebook)
