# coding: utf8
import os
from datetime import datetime
import codecs
import hermione

def write_config_file(LOCAL_PATH, project_name):
    with open(os.path.join(LOCAL_PATH, project_name, 'src/config', 'config.json'), 'w+') as file:
            file.write("{\n")
            file.write(f""""project_name": "{project_name}",
            "env_path": "{project_name}/{project_name}_env",
            "files_path": "../data/raw/",
            "key": "<<<<key>>>>",
            "user": "<<<<user>>>>"
            """
            )
            file.write("}")

def write_logging_file(LOCAL_PATH, project_name):
    with open(os.path.join(LOCAL_PATH, project_name, 'src/config', 'loggin.json'), 'w+') as file:
        file.write(r"""{
    "version": 1.0,
    "disable_existing_loggers": false,
    "formatters": {
        "basic": {
            "format": "%(message)s"
        },
        "detailed": {
            "format": "\n%(asctime)s:%(levelname)s:%(filename)s:%(funcName)s:%(lineno)d:\n%(message)s"
        }
    },

    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "DEBUG",
            "formatter": "basic",
            "stream": "ext://sys.stdout"
        },

        "info_file_handler": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "INFO",
            "formatter": "basic",
            "filename": "logs/info.log",
            "maxBytes": 10485760,
            "backupCount": 10,
            "encoding": "utf8"
        },

        "error_file_handler": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "ERROR",
            "formatter": "detailed",
            "filename": "logs/errors.log",
            "maxBytes": 10485760,
            "backupCount": 10,
            "encoding": "utf8"
        }
    },

    "loggers": {
        "ml_logger": {
            "level": "DEBUG",
            "handlers": ["console", "info_file_handler", "error_file_handler"],
            "propagate": false
        }
    }
}
""")

def write_requirements_txt(LOCAL_PATH, project_name, file_source):
    with open(os.path.join(hermione.__path__[0], file_source, 'requirements.txt'), 'r') as infile:
        arquivo = infile.readlines()
        with open(os.path.join(LOCAL_PATH, project_name, 'requirements.txt'), 'w+') as outfile:
            outfile.writelines(arquivo)

def write_gitignore(LOCAL_PATH, project_name, file_source):
    with open(os.path.join(LOCAL_PATH, project_name, '.gitignore'), 'w+') as outfile:
        outfile.writelines(".ipynb_checkpoints \nmlruns/ \n__pycache__/ \n.vscode/ \ncatboost_info/ \n.metaflow \ndata/ \n*_env/")

def write_readme_file(LOCAL_PATH, project_name, file_source):
    with codecs.open(os.path.join(LOCAL_PATH, project_name, 'README.md'), 'w+', "utf-8-sig") as outfile:
        outfile.write(f"""# {project_name}

Project started in {datetime.today().strftime("%B %d, %Y")}.


**Please, complete here information on using and testing this project.**
""")

def write_application_config(LOCAL_PATH, project_name, file_source):
    with open(os.path.join(hermione.__path__[0], file_source, 'app_config.txt'), 'r') as infile:
        arquivo = infile.readlines()
        with open(os.path.join(LOCAL_PATH, project_name, 'src','config.py'), 'w+') as outfile:
            outfile.writelines(arquivo)

def write_src_util_file(LOCAL_PATH, project_name, file_source):
    with codecs.open(os.path.join(hermione.__path__[0], file_source, 'src_util.txt'), 'r', 'utf-8-sig') as infile:
        arquivo = infile.readlines()
        with codecs.open(os.path.join(LOCAL_PATH, project_name, 'src','util.py'), 'w+', 'utf-8-sig') as outfile:
            outfile.writelines(arquivo)

def write_wsgi_file(LOCAL_PATH, project_name, file_source):
    with open(os.path.join(hermione.__path__[0], file_source, 'wsgi.txt'), 'r') as infile:
        arquivo = infile.readlines()
        with open(os.path.join(LOCAL_PATH, project_name, 'src', 'api', 'wsgi.py'), 'w+') as outfile:
            outfile.writelines(arquivo)

def write_app_file(LOCAL_PATH, project_name, file_source):
    with open(os.path.join(hermione.__path__[0], file_source, 'app.txt'), 'r') as infile:
        arquivo = infile.readlines()
        with open(os.path.join(LOCAL_PATH, project_name, 'src', 'api', 'app.py'), 'w+') as outfile:
            outfile.writelines(arquivo)

def write_myrequests_file(LOCAL_PATH, project_name, file_source):
    with open(os.path.join(hermione.__path__[0], file_source, 'myrequests.txt'), 'r') as infile:
        arquivo = infile.readlines()
        with open(os.path.join(LOCAL_PATH, project_name, 'src', 'api', 'myrequests.py'), 'w+') as outfile:
            outfile.writelines(arquivo)




def write_visualization_file(LOCAL_PATH, project_name, file_source):
    with open(os.path.join(hermione.__path__[0], file_source, 'visualization.txt'), 'r') as infile:
        arquivo = infile.readlines()
        with open(os.path.join(LOCAL_PATH, project_name, 'src', 'ml', 'visualization', 'visualization.py'), 'w+') as outfile:
            outfile.writelines(arquivo)

def write_visualization_streamlit_file(LOCAL_PATH, project_name, file_source):
    with open(os.path.join(hermione.__path__[0], file_source, 'app-streamlit-titanict.txt'), 'r') as infile:
        arquivo = infile.readlines()
        with open(os.path.join(LOCAL_PATH, project_name, 'src', 'ml', 'visualization', 'app-streamlit-titanict.py'), 'w+') as outfile:
            outfile.writelines(arquivo)

def write_normalization_file(LOCAL_PATH, project_name, file_source):
    with open(os.path.join(hermione.__path__[0], file_source, 'normalization.txt'), 'r') as infile:
        arquivo = infile.readlines()
        with open(os.path.join(LOCAL_PATH, project_name, 'src', 'ml', 'preprocessing', 'normalization.py'), 'w+') as outfile:
            outfile.writelines(arquivo)

def write_preprocessing_file(LOCAL_PATH, project_name, file_source):
    with open(os.path.join(hermione.__path__[0], file_source, 'preprocessing.txt'), 'r') as infile:
        arquivo = infile.readlines()
        with open(os.path.join(LOCAL_PATH, project_name, 'src', 'ml', 'preprocessing', 'preprocessing.py'), 'w+') as outfile:
            outfile.writelines(arquivo)

def write_text_vectorizer_file(LOCAL_PATH, project_name, file_source):
    with open(os.path.join(hermione.__path__[0], file_source, 'text_vectorizer.txt'), 'r') as infile:
        arquivo = infile.readlines()
        with open(os.path.join(LOCAL_PATH, project_name, 'src', 'ml', 'preprocessing', 'text_vectorizer.py'), 'w+') as outfile:
            outfile.writelines(arquivo)

def write_feature_selection_file(LOCAL_PATH, project_name, file_source):
    with open(os.path.join(hermione.__path__[0], file_source, 'feature_selection.txt'), 'r') as infile:
        arquivo = infile.readlines()
        with open(os.path.join(LOCAL_PATH, project_name, 'src', 'ml', 'analysis', 'feature_selection.py'), 'w+') as outfile:
            outfile.writelines(arquivo)

def write_pca_file(LOCAL_PATH, project_name, file_source):
    with open(os.path.join(hermione.__path__[0], file_source, 'pca.txt'), 'r') as infile:
        arquivo = infile.readlines()
        with open(os.path.join(LOCAL_PATH, project_name, 'src', 'ml', 'analysis', 'pca.py'), 'w+') as outfile:
            outfile.writelines(arquivo)

def write_vif_file(LOCAL_PATH, project_name, file_source):
    with open(os.path.join(hermione.__path__[0], file_source, 'vif.txt'), 'r') as infile:
        arquivo = infile.readlines()
        with open(os.path.join(LOCAL_PATH, project_name, 'src', 'ml', 'analysis', 'vif.py'), 'w+') as outfile:
            outfile.writelines(arquivo)

def write_cluster_file(LOCAL_PATH, project_name, file_source):
    with open(os.path.join(hermione.__path__[0], file_source, 'cluster.txt'), 'r') as infile:
        arquivo = infile.readlines()
        with open(os.path.join(LOCAL_PATH, project_name, 'src', 'ml', 'analysis', 'cluster.py'), 'w+') as outfile:
            outfile.writelines(arquivo)

def write_metrics_file(LOCAL_PATH, project_name, file_source):
    with open(os.path.join(hermione.__path__[0], file_source, 'metrics.txt'), 'r') as infile:
        arquivo = infile.readlines()
        with open(os.path.join(LOCAL_PATH, project_name, 'src', 'ml', 'model', 'metrics.py'), 'w+') as outfile:
            outfile.writelines(arquivo)

def write_trainer_file(LOCAL_PATH, project_name, file_source):
    with open(os.path.join(hermione.__path__[0], file_source, 'trainer.txt'), 'r') as infile:
        arquivo = infile.readlines()
        with open(os.path.join(LOCAL_PATH, project_name, 'src', 'ml', 'model', 'trainer.py'), 'w+') as outfile:
            outfile.writelines(arquivo)

def write_wrapper_file(LOCAL_PATH, project_name, file_source):
    with open(os.path.join(hermione.__path__[0], file_source, 'wrapper.txt'), 'r') as infile:
        arquivo = infile.readlines()
        with open(os.path.join(LOCAL_PATH, project_name, 'src', 'ml', 'model', 'wrapper.py'), 'w+') as outfile:
            outfile.writelines(arquivo)

def write_data_source_base_file(LOCAL_PATH, project_name, file_source):
    with open(os.path.join(hermione.__path__[0], file_source, 'data_source_base.txt'), 'r') as infile:
        arquivo = infile.readlines()
        with open(os.path.join(LOCAL_PATH, project_name, 'src', 'ml', 'data_source', 'base.py'), 'w+') as outfile:
            outfile.writelines(arquivo)

def write_database_file(LOCAL_PATH, project_name, file_source):
    with open(os.path.join(hermione.__path__[0], file_source, 'database.txt'), 'r') as infile:
        arquivo = infile.readlines()
        with open(os.path.join(LOCAL_PATH, project_name, 'src', 'ml', 'data_source', 'database.py'), 'w+') as outfile:
            outfile.writelines(arquivo)

def write_spreadsheet_file(LOCAL_PATH, project_name, file_source):
    with open(os.path.join(hermione.__path__[0], file_source, 'spreadsheet.txt'), 'r') as infile:
        arquivo = infile.readlines()
        with open(os.path.join(LOCAL_PATH, project_name, 'src', 'ml', 'data_source', 'spreadsheet.py'), 'w+') as outfile:
            outfile.writelines(arquivo)

def write_example_notebook_file(LOCAL_PATH, project_name, file_source):
    with open(os.path.join(hermione.__path__[0], file_source, 'example_notebook.txt'), 'r') as infile:
        arquivo = infile.readlines()
        with open(os.path.join(LOCAL_PATH, project_name, 'src', 'ml', 'notebooks', 'example_notebook.ipynb'), 'w+') as outfile:
            outfile.writelines(arquivo)

def write_train_dot_py(LOCAL_PATH, project_name, file_source):
    with open(os.path.join(hermione.__path__[0], file_source, 'train_dot_py.txt'), 'r') as infile:
        arquivo = infile.readlines()
        with open(os.path.join(LOCAL_PATH, project_name, 'src', 'train.py'), 'w+') as outfile:
            outfile.writelines(arquivo)

def write_titanic_data(LOCAL_PATH, project_name, file_source):
    with codecs.open(os.path.join(hermione.__path__[0], file_source, 'train.csv'), 'r', 'utf-8-sig') as infile:
        arquivo = infile.readlines()
        with codecs.open(os.path.join(LOCAL_PATH, project_name, 'data', 'raw','train.csv'), 'w+', 'utf-8-sig') as outfile:
            outfile.writelines(arquivo)

def write_test_file(LOCAL_PATH, project_name, file_source):
    with codecs.open(os.path.join(hermione.__path__[0], file_source, 'test_project.txt'), 'r', 'utf-8-sig') as infile:
        arquivo = infile.readlines()
        with codecs.open(os.path.join(LOCAL_PATH, project_name, 'src', 'tests','test_project.py'), 'w+', 'utf-8-sig') as outfile:
            outfile.writelines(arquivo)

def write_test_readme(LOCAL_PATH, project_name, file_source):
    with codecs.open(os.path.join(hermione.__path__[0], file_source, 'test_readme.txt'), 'r', 'utf-8-sig') as infile:
        arquivo = infile.readlines()
        with codecs.open(os.path.join(LOCAL_PATH, project_name, 'src', 'tests','README.md'), 'w+', 'utf-8-sig') as outfile:
            outfile.writelines(arquivo)

def write_predict(LOCAL_PATH, project_name, file_source):
    with open(os.path.join(hermione.__path__[0], file_source, 'predict.txt'), 'r') as infile:
        arquivo = infile.readlines()
        with open(os.path.join(LOCAL_PATH, project_name, 'src', 'predict.py'), 'w+') as outfile:
            outfile.writelines(arquivo)

def write_dockerfile(LOCAL_PATH, project_name, file_source):
    with open(os.path.join(hermione.__path__[0], file_source, 'dockerfile.txt'), 'r') as infile:
        arquivo = infile.readlines()
        with open(os.path.join(LOCAL_PATH, project_name, 'src', 'Dockerfile'), 'w+') as outfile:
            outfile.writelines(arquivo)
