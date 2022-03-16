import pytest
import os
import shutil
import tempfile
from dataclasses import dataclass
from typing import List, Dict
import json
from datetime import datetime
from hermione.templating import create_project
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent


def read_json(file):
    with open(os.path.join(CURRENT_DIR, file)) as f:
        data = json.load(f)
    return data


def get_all_file_paths(folder):
    return list(str(path) for path in Path(folder).rglob("*"))


@pytest.fixture()
def cleandir():
    old_cwd = os.getcwd()
    newpath = tempfile.mkdtemp()
    os.chdir(newpath)
    yield
    os.chdir(old_cwd)
    shutil.rmtree(newpath)


@dataclass
class ProjectTestCase:
    template: str
    context_data: Dict
    expected_project_name: str
    expected_files: List[str]


def project_test_cases():
    context_data = {"project_start_date": datetime.now().strftime("%B %d, %Y")}
    return [
        ProjectTestCase(
            template="barebones",
            expected_project_name="barebones_project",
            expected_files=read_json("barebones_project_files.json"),
            context_data=context_data,
        ),
        ProjectTestCase(
            template="starter",
            expected_project_name="starter_project",
            expected_files=read_json("starter_project_files.json"),
            context_data=context_data,
        ),
        ProjectTestCase(
            template="sagemaker",
            expected_project_name="sagemaker_project",
            expected_files=read_json("sagemaker_project_files.json"),
            context_data=context_data,
        ),
    ]


@pytest.mark.usefixtures("cleandir")
@pytest.mark.parametrize("test_case", project_test_cases())
def test_project_creation(test_case: ProjectTestCase):
    create_project(
        os.getcwd(),
        test_case.expected_project_name,
        test_case.template,
        test_case.context_data,
        output_blueprint=True,
    )
    project_path = os.path.join(os.getcwd(), test_case.expected_project_name)
    assert os.path.exists(project_path)
    for file in test_case.expected_files:
        assert os.path.exists(file)
