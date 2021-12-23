# Hermione test files

In this folder, you can develop unit tests for your Data Science project.

Unit testing is a regular process in software development but, unfortunately, not so common in Data Science projects. To ensure your code quality and that the project is running flawless at all times, it is extremely important that you code unit tests, specially if you are not working alone but in a Data Science team.

The tests you have in the implemented example project test, for instance, if the project has its minimum directory structure, if your dataset is correctly imported, if the dataset has no missing values and that some columns that should be there are there indeed after preprocessing.

There are no "written in stone" rules to good testing in Data Science. You just have to figure out what tests are best for you.

## How to run the tests

When working locally, you should run your tests before pushing to a remote repository or sharing your code to others. To do that, **ensure that you are inside `tests` folder**.

```bash
cd src/tests
```

Then, run the `pytest` command.

```bash
pytest
```

If you want to have a coverage report, do so:

```bash
coverage run -m pytest
coverage report -m
```

Both `coverage` and `pytest` libraries are already in the `requirements.txt` file.

## Include tests on CI/CD files

If you are working with a remote repository, it is a great practice to code a CI/CD `.yml` file. For more information, visit

- [CI/CD for Machine Learning](https://www.infoq.com/presentations/ci-cd-ml/)
- [CI/CD for Machine Learning & AI](https://blog.paperspace.com/ci-cd-for-machine-learning-ai/)
- [Accelerate MLOps: using CI/CD with machine learning models
](https://algorithmia.com/blog/accelerate-mlops-using-ci-cd-with-machine-learning-models)
