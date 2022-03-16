import os
import hermione

HERMIONE_DIR = hermione.__path__[0]

ARTIFACTS_DIR = os.path.normpath(
    os.path.join(HERMIONE_DIR, "data", "project_artifacts")
)


def get_artefact_full_path(artifact_type, artifact):
    return os.path.normpath(os.path.join(ARTIFACTS_DIR, artifact_type, artifact))


def get_artefact_full_paths(artifact_type, artifacts):
    return [get_full_path(artifact_type, artifact) for artifact in artifacts]


def get_hermione_src_dir():
    return os.path.normpath(os.path.join(HERMIONE_DIR, "core"))
