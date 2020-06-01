![hermione](images/vertical_logo.png)


![Hermione](https://github.com/A3Data/hermione/workflows/hermione/badge.svg)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![GitHub issues](https://img.shields.io/github/issues/a3data/hermione.svg)](https://GitHub.com/a3data/hermione/issues/)
[![GitHub issues-closed](https://img.shields.io/github/issues-closed/a3data/hermione.svg)](https://GitHub.com/a3data/hermione/issues?q=is%3Aissue+is%3Aclosed)


[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)

A Data Science Project struture in cookiecutter style.

Developed with ❤️ by [A3Data](http://www.a3data.com.br/)

  

## What is Hermione?

  

Hermione is the newest **open source** library that will help Data Scientists on setting up more organized codes, in a quicker and simpler way. Besides, there are some classes in Hermione which assist with daily tasks such as: column normalization and denormalization, data view, text vectoring, etc. Using Hermione, all you need is to execute a method and the rest is up to her, just like magic.

### Why Hermione?
To bring in a little of **A3Data** experience, we work in Data Science teams inside several client companies and it’s undeniable the excellence of notebooks as a data exploration tool. Nevertheless, when it comes to data science products and their context, when the models needs to be consumed, monitored and have periodic maintenance, putting it into production inside a Jupyter Notebook is not the best choice (we are not even mentioning memory and CPU performance yet). And that’s why **Hermione comes in**!
We have been inspired by this brilliant, empowered and awesome witch of The Harry Potter saga to name this framework!

This is also our way of reinforcing our position that women should be taking more leading roles in the technology field. **#CodeLikeAGirl**

## Installing


### Dependences

- Python (>= 3.6)
  

### Install

```python

pip install hermione

```
## How do I use Hermione?
After installed Hermione:
1.  Create you new project:

 ![](https://cdn-images-1.medium.com/max/800/1*7Ju0Tq2DP1pE5bfGPguh2w.png)

2. Enter “y” if you want to start with an example code

![](https://cdn-images-1.medium.com/max/800/1*TJoFVA-Nio2O3XvxBN4MUQ.png)

3. Hermione already creates a conda virtual environment for the project. Activate it

![](https://cdn-images-1.medium.com/max/800/1*38yp-E_AUxM7lIw9PCo0rw.png)

4. After activating, you should install some libraries. There are a few suggestions in “requirements.txt” file:

![](https://cdn-images-1.medium.com/max/800/1*rpXdiYmPKHNbVoKFZIHrlQ.png)

5. Now we will train some models from the example, using MLflow ❤. To do so, inside *src* directory, just type: _hermione train_. The “hermione train” command will search for a train.py file and execute it. In the example, models and metrics are already controlled via MLflow.

![](https://cdn-images-1.medium.com/max/800/1*MmVcmAYspxWdzbd5r00W5g.png)

6. After that, a mlflow experiment is created. To verify the experiment in mlflow, type: mlflow ui. The application will go up.

![](https://cdn-images-1.medium.com/max/800/1*DReyAtL9eJ0fiwxaVo3Yfw.png)

7. To access the experiment, just enter the path previously provided in your preferred browser. Then it is possible to check the trained models and their metrics.

![](https://cdn-images-1.medium.com/max/800/1*c_rDEqERZR6r8JVI3TMTcQ.png)

8. In the Titanic example, we also provide a step by step notebook. To view it, just type jupyter notebook inside directory `/src/notebooks/`.

![](https://cdn-images-1.medium.com/max/800/1*U3ToR5jDjQJihT9EnxeDdg.png)

Do you want to create your **project from scratch**? There click [here](tutorial_base.md) to check a tutorial.


## Documentation
This is the class structure diagram that Hermione relies on:

![](images/class_diagram.png)

Here we describe briefly what each class is doing:

### Data Source
-   **DataBase** - should be used when data recovery requires a connection to a database. Contains methods for opening and closing a connection.
-   **Spreadsheet**  - should be used when data recovery is in spreadsheets/text files. All aggregation of the bases to generate a "flat table" should be performed in this class.
-   **DataSource**  - abstract class which DataBase and Spreadsheet inherit from.


### Preprocessing

-   **Preprocessing**  - concentrates all preprocessing steps that must be performed on the data before the model is trained.
-   **Normalization** - applies normalization and denormalization to reported columns. This class contains the following normalization algorithms already implemented: StandardScaler e MinMaxScaler.
-   **TextVectorizer**  - transforms text into vector. Implemented methods: Bag of words, TF_IDF, Embedding: mean, median e indexing.

### Visualization

-   **Visualization** - methods for data visualization. There are methods to make static and interactive plots.

### Model

-   **Trainer**  - module that centralizes training algorithms classes. Algorithms from `scikit-learn` library can be easily used with the TrainerSklearn implemented class.
-   **Wrapper** - centralizes the trained model with its metrics. This class has built-in integration with MLFlow.
-   **Metrics** - it contains key metrics that are calculated when models are trained. Classification, regression and clustering metrics are already implemented.

### Tests
-   **test_project** - module for unit testing.
  

## Contributing

  Make a pull request with your implementation.

For suggestions, contact us: hermione@a3data.com.br

## Licence
Hermione is open source and has Apache 2.0 License: [![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)