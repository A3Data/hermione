import shutil
import os
import errno
from ._context import context
import sys
from contextlib import contextmanager
from abc import ABCMeta, abstractmethod
from jinja2 import Template
from .exceptions import ProjectDirAlreadyExistsException

def copy_folder(src, dst, dirs_exist_ok):
    shutil.copytree(src, dst, dirs_exist_ok=dirs_exist_ok)


def copy_file(src, dst, context_data=None):
    if '.tpl.' in src:
        new_dst = dst.replace(".tpl", "")
        with open(src, 'r', encoding='latin-1') as src_file:
            src_content = src_file.read()
        template = Template(src_content)
        output = template.render(context=context_data).encode('latin-1')
        with open(new_dst, 'wb') as f:
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


class Renderer(metaclass=ABCMeta):
    def __init__(self):
        self.root_renderer = None

    @abstractmethod
    def _render(self, **kwargs):
        pass


class RendererContext(Renderer, metaclass=ABCMeta):
    def __init__(self):
        super(RendererContext, self).__init__()
        self.children = []

    def add_new_renderer(self, child):
        child.root_renderer = self
        self.children.append(child)

    @contextmanager
    def _set_as_current_workspace(self):
        with context(current_workspace=self):
            yield self

    @abstractmethod
    def _render_context(self):
        pass

    def _render(self, **kwargs):
        curdir = os.getcwd()
        new_work_dir = self._render_context()
        new_work_space_path = os.path.normpath(os.path.join(curdir, new_work_dir))
        os.chdir(new_work_space_path)
        for child in self.children:
            child._render()
        os.chdir(curdir)

    def __enter__(self):
        self._ctx = self._set_as_current_workspace()
        return self._ctx.__enter__()

    def __exit__(self, exc_type, exc_value, traceback) -> None:  # type: ignore
        self._ctx.__exit__(exc_type, exc_value, traceback)
        del self._ctx


class ProjectTemplate(RendererContext):
    def __init__(self, name):
        self.name = name
        self.project_name = None
        super(ProjectTemplate, self).__init__()

    def create_project(self, project_path, project_name, context_data=None):
        if os.path.exists(os.path.join(project_path, project_name)):
            raise ProjectDirAlreadyExistsException
        with context(**context_data, project_name=project_name):
            self.project_name = project_name
            os.chdir(project_path)
            create_folder(project_name)
            self._render()

    def _render_context(self):
        return self.project_name


def context_renderer(cls):
    class_init = cls.__dict__.get('__init__')

    def __init__(self, *args, **kwargs):
        current_workspace = context.get("current_workspace", None)
        if current_workspace:
            current_workspace.add_new_renderer(self)
        if class_init:
            class_init(self, *args, **kwargs)

    setattr(cls, '__init__', __init__)
    return cls


@context_renderer
class CreateDir(RendererContext):
    def __init__(self, dirname):
        super(CreateDir, self).__init__()
        self.dirname = dirname

    def _render_context(self):
        create_folder(self.dirname)
        return self.dirname


@context_renderer
class CreateFile(Renderer):
    def __init__(self, filename):
        super(CreateFile, self).__init__()
        self.filename = filename

    def _render(self, **kwargs):
        open(self.filename, 'a').close()


@context_renderer
class RenderTemplateDir(RendererContext):
    def __init__(self, src, dst=None, merge_if_exists=False):
        super(RenderTemplateDir, self).__init__()
        self.merge_if_exists = merge_if_exists
        self.src = src
        self.dst = dst if dst else os.path.basename(os.path.normpath(src))

    def _render_context(self):
        copy_folder(self.src, self.dst, dirs_exist_ok=self.merge_if_exists)
        return self.dst


@context_renderer
class RenderTemplateFile(Renderer):
    def __init__(self, src, dst=None):
        super(RenderTemplateFile, self).__init__()
        self.src = src
        self.dst = dst if dst else os.path.basename(os.path.normpath(src))

    def _render(self, **kwargs):
        copy_file(self.src, self.dst, context_data=context)


@context_renderer
class RenderTemplateFiles:
    def __init__(self, targets):
        self.renderers = [RenderTemplateFile(target) for target in targets]

    def _render(self, **kwargs):
        pass
