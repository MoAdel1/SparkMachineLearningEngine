# SparkMachineLearningEngine

This repo is a work in progress for a machine learning engine that expands the search space per each model or model family to make sure we have the optimimum configuraion/hyper-parameters for the selected model. It can also be expanded to support feature engineering stage as well.

It is developed on [Databricks](https://databricks.com/) which is an industry-leading, cloud-based data engineering tool used for processing and transforming massive quantities of data and exploring/understanding the data through machine learning models.

This engine is mainly a template that can be expanded for any number of models that can follow the same architecture. The major advantage over traditional search engines for machine learning models is that this engine can be scaled over any numbers of nodes in a cluster and the search algorithm can be changed at any time for a more efficient search.

# Components
- **ml_engine/main.py :** the main import location for needed modules and other submodules in this engine.
- **ml_engine/models.py :** contains the current supported machine learning models. 
- **ml_engine/processing.py :** contains all the common preprocessing and helper functions that can be applied to the data before/after passing it to the model.

# References
- [Databricks documentation](https://docs.databricks.com/)
- [Core module for searching the space](https://github.com/hyperopt/hyperopt)