import os
from pathlib import Path
import hermione

HERMIONE_DIR = hermione.__path__[0]


class ArtifactsPathLoader(object):
    def __init__(self):
        current_dir = os.path.normpath(os.path.dirname(os.path.abspath(__file__)))
        self.artifacts_dir = os.path.normpath(os.path.join(current_dir, 'project_artifacts'))

    def get_path(self, artifact_type, artifact):
        return os.path.normpath(os.path.join(self.artifacts_dir, artifact_type, artifact))

    def get_paths(self, artifact_type, artifacts):
        return [self.get_path(artifact_type, artifact) for artifact in artifacts]

    def get_hermione_src_dir(self):
        return os.path.normpath(os.path.join(HERMIONE_DIR, "core"))
