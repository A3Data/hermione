class TemplatingException(Exception):
    """
    Base exception class.

    All template-specific exceptions are a subclass of this class.
    """


class ProjectDirAlreadyExistsException(TemplatingException):
    """
    Exception for when a project's dir already exists.
    """
