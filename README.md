Boosted Trees (GradientBoosting) Classifier in SciKitLearn Using Tf-IDF preprocessing for Text Classification - Base problem category as per Ready Tensor specifications.

* sklearn
* python
* pandas
* numpy
* scikit-optimize
* flask
* nginx
* uvicorn
* docker
* text classification

This is a Text Classifier that uses a Boosted Trees model implemented through SciKitLearn.

This boosted tree model works through learning from the errors of an ensemble of weaker decision trees and creates stronger decision trees from this. 

The data preprocessing step includes tokenizing the input text, applying a tf-idf vectorizer to the tokenized text, and applying Singular Value Decomposition (SVD) to find the optimal factors coming from the original matrix. In regards to processing the labels, a label encoder is used to turn the string representation of a class into a numerical representation.

Hyperparameter Tuning (HPT) is conducted by finding the optimal number of boosting stages to perform, number of samples required to be at a leaf node, number of samples required to split an internal node, learning rate that determines the contribution of each tree, and depth of the individual regression estimators.

During the model development process, the algorithm was trained and evaluated on a variety of datasets such as email spam detection, customer churn, credit card fraud detection, cancer diagnosis, and titanic passanger survivor prediction.

This Binary Classifier is written using Python as its programming language. Scikitlearn is used to implement the main algorithm, preprocess the data, and evaluate the model. NLTK, numpy, and pandas are used for the data preprocessing steps. SciKit-Optimize was used to handle the HPT. Flask + Nginx + gunicorn are used to provide web service which includes two endpoints- /ping for health check and /infer for predictions in real time.


