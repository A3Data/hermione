import os
import click


@click.group()
def run():
    pass


@run.command()
def train():
    """
    Execute the script in train.py. One should be at src directory
    """
    if not os.path.exists("./scripts/train.py"):
        click.echo("You gotta have an scripts/train.py file")
    else:
        os.system("python ./scripts/train.py")
        print(
            "\nModel trained. For MLFlow logging control, type:\nmlflow ui\nand visit http://localhost:5000/"
        )


@run.command()
def predict():
    """
    Execute the script in predict.py to make batch predictions.
    One should be at src directory
    """
    if not os.path.exists("scripts/predict.py"):
        click.echo(
            "You gotta have an scripts/predict.py file. You must be at the project's root folder."
        )
    else:
        print("Making predictions: ")
        os.system("python ./scripts/predict.py")


@click.argument("image_name")
@click.option("-t", "--tag", "tag", default="latest", show_default=True)
@run.command()
def build(image_name, tag):
    """
    Build a docker image with given image_name. Only run if you have docker installed.
    One should be at the root directory.
    """
    if not os.path.exists("Dockerfile"):
        click.echo(
            "You gotta have an Dockerfile file. You must be at the project's root folder."
        )
    else:
        os.system(f"docker build -f Dockerfile -t {image_name}:{tag} .")


@click.argument("image_name")
@click.option("-t", "--tag", "tag", default="latest", show_default=True)
@run.command()
def container(image_name, tag):
    """
    Run a container with given image_name.
    Only run if you have docker installed.
    """
    if not os.path.exists("Dockerfile"):
        click.echo(
            "You gotta have an Dockerfile file. You must be at the project's root folder."
        )
    else:
        os.system(f"docker run --rm -p 5000:5000 {image_name}:{tag}")
