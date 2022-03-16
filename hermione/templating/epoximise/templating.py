import shutil
import os
import errno
from typing import Optional, List
from ._context import context
import sys
from contextlib import contextmanager
from abc import ABCMeta, abstractmethod
from jinja2 import Template
from .exceptions import ProjectDirAlreadyExistsException


def copy_folder(src, dst, dirs_exist_ok):
    shutil.copytree(src, dst, dirs_exist_ok=dirs_exist_ok)


def copy_file(src, dst, context_data=None):
    if ".tpl." in src:
        new_dst = dst.replace(".tpl", "")
        with open(src, "r", encoding="latin-1") as src_file:
            src_content = src_file.read()
        template = Template(src_content)
        output = template.render(context=context_data).encode("latin-1")
        with open(new_dst, "wb") as f:
            f.write(output)
    else:
        shutil.copyfile(src, dst)
        shutil.copymode(src, dst)


def create_folder(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            return False
    return True


class Node(metaclass=ABCMeta):
    @abstractmethod
    def eval(self, **kwargs):
        raise NotImplementedError()


class ChildNode(Node, metaclass=ABCMeta):
    def __init__(self, parent: Optional[Node] = None):
        self.parent = parent


class BranchNode(ChildNode, metaclass=ABCMeta):
    def __init__(self, parent: Optional["BranchNode"] = None):
        super().__init__(parent=parent)
        self.children: List[Node] = []

    def add_child(self, child: Node):
        child.parent = self
        self.children.append(child)


class TerminalNode(ChildNode, metaclass=ABCMeta):
    def __init__(self, parent: Optional[BranchNode] = None):
        super().__init__(parent=parent)


class ChangeDirectory(BranchNode, metaclass=ABCMeta):
    @contextmanager
    def _set_as_active_workspace(self):
        with context(active_workspace=self):
            yield self

    @abstractmethod
    def _eval_self(self):
        raise NotImplementedError()

    def eval(self, **kwargs):
        curdir = os.getcwd()
        new_work_dir = self._eval_self()
        new_work_space_path = os.path.normpath(os.path.join(curdir, new_work_dir))
        os.chdir(new_work_space_path)
        for child in self.children:
            child.eval()
        os.chdir(curdir)

    def __enter__(self):
        self._ctx = self._set_as_active_workspace()
        return self._ctx.__enter__()

    def __exit__(self, exc_type, exc_value, traceback) -> None:  # type: ignore
        self._ctx.__exit__(exc_type, exc_value, traceback)
        del self._ctx


class ProjectTemplate(ChangeDirectory):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.project_name = None

    def create_project(self, project_path, project_name, context_data=None):
        if os.path.exists(os.path.join(project_path, project_name)):
            raise ProjectDirAlreadyExistsException
        with context(**context_data, project_name=project_name):
            self.project_name = project_name
            os.chdir(project_path)
            create_folder(project_name)
            self.eval()

    def _eval_self(self):
        return self.project_name


def active_workspace_as_parent(cls: ChildNode):
    class_init = cls.__dict__.get("__init__")

    def __init__(self, *args, **kwargs):
        if class_init:
            class_init(self, *args, **kwargs)
        current_workspace = context.get("active_workspace", None)
        if current_workspace and not self.parent:
            self.parent = current_workspace
            if isinstance(current_workspace, BranchNode):
                current_workspace.add_child(self)

    setattr(cls, "__init__", __init__)
    return cls


@active_workspace_as_parent
class CreateDir(ChangeDirectory):
    def __init__(self, dirname):
        super().__init__()
        self.dirname = dirname

    def _eval_self(self):
        create_folder(self.dirname)
        return self.dirname


@active_workspace_as_parent
class CreateFile(TerminalNode):
    def __init__(self, filename):
        super().__init__(parent=parent)
        self.filename = filename

    def eval(self, **kwargs):
        open(self.filename, "a").close()


@active_workspace_as_parent
class CopyDir(ChangeDirectory):
    def __init__(self, src, dst=None, merge_if_exists=False):
        super().__init__()
        self.merge_if_exists = merge_if_exists
        self.src = src
        self.dst = dst if dst else os.path.basename(os.path.normpath(src))

    def _eval_self(self):
        copy_folder(self.src, self.dst, dirs_exist_ok=self.merge_if_exists)
        return self.dst


@active_workspace_as_parent
class CopyFile(TerminalNode):
    def __init__(self, src, dst=None):
        super().__init__()
        self.src = src
        self.dst = dst if dst else os.path.basename(os.path.normpath(src))

    def eval(self, **kwargs):
        copy_file(self.src, self.dst, context_data=context)


@active_workspace_as_parent
class Hook(TerminalNode):
    def __init__(self, hook_callable, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.hook_callable = hook_callable

    def eval(self, **kwargs):
        self.hook_callable(**self.kwargs)
