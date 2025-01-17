# LLM Project

## Project Task
Create a sentiment analysis tool that can classify a text as positive,negative or neutral


## Dataset
The dataset that we will use is taken from Kaggle website and can be downloaded here:

Amazon Musical Instruments Reviews

There are two formats available of the dataset: JSON and CSV. We will use the CSV one in this project.

Overall, the dataset talks about the feedback received after the customers purchased musical instruments from Amazon.

## Pre-trained Model
We are using K-Fold Cross Validation on our early dataset (before resampling) because the CV itself is not affected by the imbalanced dataset as it splits the dataset and takes into account every validations

## Performance Metrics
The accuracy of the model will be calculated as the number of correct predictions divided by the total predictions.

## Hyperparameters
Param = {"C": np.logspace(-4, 4, 50), "penalty": ['l1', 'l2']}
grid_search = GridSearchCV(estimator = LogisticRegression(random_state = 42), param_grid = Param, scoring = "accuracy", cv = 10, verbose = 0, n_jobs = -1)

grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_

