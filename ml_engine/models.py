# Databricks notebook source
# Hybrid logistic regression

def pred_logistic_linear_regression(logistic_model, linear_reg_model, cut_off, x_values):
  # run the classifier
  output = logistic_model.predict(x_values)
  output = output.reshape(-1, 1)
  # run the regressor
  predections = linear_reg_model.predict(x_values[(output == 1).reshape(-1), :])
  output[(output == 1).reshape(-1), :] = predections
  # apply post processing
  output[output <= cut_off] = 0
  return output

def logistic_linear_regression(params):  
  # extract parameters
  class_weight = params['class_weight']
  cut_off = round(params['cut_off'], 2)
  C = params['C']
  # set the tags values
  log_tags(params)
  # log the artifacts
  log_pre_processing(params)
  # load datasets
  x_train, y_train, x_validate, y_validate, x_test, y_test, _ = load_datasets(params['dataset_path'])
  # generate class labels
  y_train_class = generate_class(y_train, 0, 1, 0)
  # train logistic regression model
  logistic_model = LogisticRegression(n_jobs=-1, max_iter=10000, 
                                      C=C, class_weight=class_weight).fit(x_train, y_train_class)
  # train c_1 (has value) linear regression model
  linear_reg_model = LinearRegression(n_jobs=-1).fit(x_train[(y_train_class == 1).reshape(-1), :], 
                                                     y_train[(y_train_class == 1).reshape(-1), :])
  # calculate predection
  pred_validate = pred_logistic_linear_regression(logistic_model, linear_reg_model, cut_off, x_validate)
  pred_test = pred_logistic_linear_regression(logistic_model, linear_reg_model, cut_off, x_test)
  # calculate and log metrics for validation set
  validation_metrics = metrics(pred_validate, y_validate, suffix='validate')
  # calculate and log metrics for test set
  test_metrics = metrics(pred_test, y_test, suffix='test')
  # calculate and log confidence between test and validation sets
  conf_dict = confidence(validation_metrics, test_metrics)
  # save predictions
  save_results(y_validate, y_test, pred_validate, pred_test)
  # register model
  mlflow.sklearn.log_model(logistic_model, 'logistic_model')
  mlflow.sklearn.log_model(linear_reg_model, 'linear_reg_model')
  # function return
  return {'loss': -1*(validation_metrics['f_c_1_validate'] + validation_metrics['f_c_0_validate']), 'status': STATUS_OK} 

def train_logistic_linear_regression(experiment_path, params_space, max_evals):
  # experiment configs
  algo=tpe.suggest # other option is rand.suggest 
  spark_trials = SparkTrials(parallelism=20, timeout=7200)
  mlflow.set_experiment(experiment_path)
  # calculat sample weight and cut off range
  max_importance = 0.6
  sample_weight_limit = min(int((max_importance*size_class_no_value)/(size_class_value*(1-max_importance))), 35)
  cuf_off_limit = min(mean_iap_value, 20)
  # start the training process
  with mlflow.start_run():
    best_result = fmin(fn=logistic_linear_regression, space=params_space, 
                       algo=algo, max_evals=max_evals, trials=spark_trials)

# COMMAND ----------

# Logistic regression

def pred_lr(model, threshold, x_values):
  class_preds = model.predict_proba(x_values)
  output = np.copy(class_preds[:, list(model.classes_).index(1)])
  output[output >= threshold] = 1
  output[output < threshold] = 0
  return output, class_preds


def lr(params):  
  # extract parameters
  class_weight = params['class_weight']
  C = params['C']
  threshold = params['threshold']
  loss = params['loss']
  # set the tags values
  log_tags(params)
  # log the artifacts
  log_pre_processing(params)
  # load datasets
  x_train, y_train, x_validate, y_validate, x_test, y_test, data_columns  = load_datasets(params['dataset_path'])
  features_columns = data_columns[:-1].tolist()
  mlflow.set_tag('features_columns', features_columns)
  # train the model 
  model = LogisticRegression(n_jobs=-1, max_iter=10000, 
                             C=C, class_weight=class_weight).fit(x_train, y_train)
  # calculate predection
  pred_validate, _ = pred_lr(model, threshold, x_validate)
  pred_test, _ = pred_lr(model, threshold, x_test)
  # calculate and log metrics for validation set
  validation_metrics = metrics(pred_validate, y_validate, suffix='validate')
  # calculate and log metrics for test set
  test_metrics = metrics(pred_test, y_test, suffix='test')
  # calculate and log confidence between test and validation sets
  conf_dict = confidence(validation_metrics, test_metrics)
  # save predictions
  save_results(y_validate, y_test, pred_validate, pred_test)
  # register model
  mlflow.sklearn.log_model(model, 'lr_model')
  # function return
  if loss=='normal':
    return {'loss': -1*(validation_metrics['f_c_1_validate'] + validation_metrics['f_c_0_validate']), 'status': STATUS_OK}
  else:
    return {'loss': -1*(0.7*validation_metrics['r_c_1_validate'] + 0.3*validation_metrics['p_c_1_validate']), 'status': STATUS_OK}

def train_lr(experiment_path, params_space, max_evals):
  # experiment configs
  algo=tpe.suggest # other option is rand.suggest 
  spark_trials = SparkTrials(parallelism=5, timeout=7200)
  mlflow.set_experiment(experiment_path)
  # params configs
  '''
  params_space = {'class_weight': {0: 1, 1: hp.quniform('class_weight', 30, 40, 5)},   
                  'threshold': hp.uniform('threshold', 0.2, 0.5),
                  'C': 1,
                  'dataset_path': dataset_path,
                  'model_type': model_type,
                  'model_name': model_name,
                  'training_log_path': training_log_path,
                  'register_model': register_model, 
                  'dataset_size': dataset_size,
                  'data_query': data_query,
                  'pre_processing': pre_processing,
                  'loss': 'fishing_net'}
  '''
  # start the training process
  with mlflow.start_run():
    best_result = fmin(fn=lr, space=params_space, 
                       algo=algo, max_evals=max_evals, trials=spark_trials)

# COMMAND ----------

# Random forest

def log_rf_analysis(model, features):
  importance = model.feature_importances_
  importance_map = dict()
  for f,i in zip(features, importance):
    importance_map[f] = i
  mlflow.set_tag('importance', importance_map)

def pred_rf(model, threshold, x_values):
  class_preds = model.predict_proba(x_values)
  output = np.copy(class_preds[:, list(model.classes_).index(1)])
  output[output >= threshold] = 1
  output[output < threshold] = 0
  return output, class_preds

def rf(params):  
  # extract parameters
  class_weight = params['class_weight']
  threshold = params['threshold']
  max_depth = params['max_depth']
  n_estimators = params['n_estimators']
  loss = params['loss']
  # set the tags values
  log_tags(params)
  # log the artifacts
  log_pre_processing(params)
  # load datasets
  x_train, y_train, x_validate, y_validate, x_test, y_test, data_columns  = load_datasets(params['dataset_path'])
  features_columns = data_columns[:-1].tolist()
  mlflow.set_tag('features_columns', features_columns)
  # train the model
  model = RandomForestClassifier(n_jobs=-1, 
                                 n_estimators=n_estimators, 
                                 max_depth=max_depth, 
                                 class_weight=class_weight).fit(x_train, y_train)
  # log analysis
  log_rf_analysis(model, features_columns)
  # calculate predection
  pred_validate, _ = pred_rf(model, threshold, x_validate)
  pred_test, _ = pred_rf(model, threshold, x_test)
  # calculate and log metrics for validation set
  validation_metrics = metrics(pred_validate, y_validate, suffix='validate')
  # calculate and log metrics for test set
  test_metrics = metrics(pred_test, y_test, suffix='test')
  # calculate and log confidence between test and validation sets
  conf_dict = confidence(validation_metrics, test_metrics)
  # save predictions
  save_results(y_validate, y_test, pred_validate, pred_test)
  # register model
  mlflow.sklearn.log_model(model, 'rf_model')
  # function return
  if loss=='normal':
    return {'loss': -1*(validation_metrics['f_c_1_validate'] + validation_metrics['f_c_0_validate']), 'status': STATUS_OK}
  else:
    return {'loss': -1*(0.7*validation_metrics['r_c_1_validate'] + 0.3*validation_metrics['p_c_1_validate']), 'status': STATUS_OK} 

def train_rf(experiment_path, params_space, max_evals):
  # experiment configs
  algo=tpe.suggest # other option is rand.suggest 
  spark_trials = SparkTrials(parallelism=3)
  mlflow.set_experiment(experiment_path)
  # params configs
  '''
  params_space = {'class_weight': {0: 1, 1: hp.quniform('class_weight', 30, 55, 5)},
                  'threshold': hp.uniform('threshold', 0.4, 0.5),
                  'max_depth': hp.choice('max_depth', [None, 5, 10]),
                  'n_estimators': hp.choice('n_estimators', [20, 50, 100]),
                  'dataset_path': dataset_path,
                  'model_type': model_type,
                  'model_name': model_name,
                  'training_log_path': training_log_path,
                  'register_model': register_model, 
                  'dataset_size': dataset_size,
                  'data_query': data_query,
                  'pre_processing': pre_processing,
                  'loss': 'fishing_net'}
  '''
  # start the training process
  with mlflow.start_run():
    best_result = fmin(fn=rf, space=params_space, 
                       algo=algo, max_evals=max_evals, trials=spark_trials)

# COMMAND ----------

# Xgboost

def log_xgboost_analysis(model, features):
  # save F score graph
  ax = plot_importance(model)
  plt.rcParams['figure.figsize'] = [20, 20]
  ax.figure.savefig('Importance.png')
  mlflow.log_artifact('Importance.png')
  # save the F score dict
  f_scores = model.get_booster().get_fscore()
  f_score_map = dict()
  for i in range(len(features)):
    target = f'f{i}'
    value = f_scores.get(target)
    f_score_map[features[i]] = (target, value if value != None else 0)
  mlflow.set_tag('f_score_map', f_score_map)

def pred_xgboost(model, threshold, x_values):
  class_preds = model.predict_proba(x_values)
  probs = np.copy(class_preds[:, list(model.classes_).index(1)])
  output = np.copy(class_preds[:, list(model.classes_).index(1)])
  output[output >= threshold] = 1
  output[output < threshold] = 0
  return output, probs


def xgboost(params):  
  # extract parameters
  sample_weight = params['sample_weight']
  threshold = params['threshold']
  max_depth = params['max_depth']
  n_estimators = params['n_estimators']
  loss = params['loss']
  gamma = params['gamma']
  learning_rate = params['learning_rate']
  subsample = params['subsample']
  colsample_bytree = params['colsample_bytree']
  min_child_weight = params['min_child_weight']
  # set the tags values
  log_tags(params)
  # log the artifacts
  log_pre_processing(params)
  # load datasets
  x_train, y_train, x_validate, y_validate, x_test, y_test, data_columns  = load_datasets(params['dataset_path'])
  features_columns = data_columns[:-1].tolist()
  mlflow.set_tag('features_columns', features_columns)
  # train the model
  model = XGBClassifier(n_jobs=-1,
                        max_depth=max_depth,
                        n_estimators=n_estimators,
                        learning_rate=learning_rate,
                        gamma=gamma,
                        subsample=subsample,
                        colsample_bytree=colsample_bytree,
                        min_child_weight=min_child_weight,
                        objective="binary:logistic",
                        scale_pos_weight=sample_weight)
  model.fit(X=x_train, y=y_train, eval_set=[(x_validate, y_validate)], 
            early_stopping_rounds=20, verbose=0)
  # log analysis
  log_xgboost_analysis(model, features_columns)
  # calculate predection
  pred_validate, _ = pred_xgboost(model, threshold, x_validate)
  pred_test, _ = pred_xgboost(model, threshold, x_test)
  # calculate and log metrics for validation set
  validation_metrics = metrics(pred_validate, y_validate, suffix='validate')
  # calculate and log metrics for test set
  test_metrics = metrics(pred_test, y_test, suffix='test')
  # calculate and log confidence between test and validation sets
  conf_dict = confidence(validation_metrics, test_metrics)
  # save predictions
  save_results(y_validate, y_test, pred_validate, pred_test)
  # register model
  mlflow.sklearn.log_model(model, 'xgboost_model')
  # function return
  if loss=='normal':
    return {'loss': -1*(validation_metrics['f_c_1_validate'] + validation_metrics['f_c_0_validate']), 'status': STATUS_OK}
  else:
    return {'loss': -1*(0.7*validation_metrics['r_c_1_validate'] + 0.3*validation_metrics['p_c_1_validate']), 'status': STATUS_OK} 

def train_xgboost(experiment_path, params_space, max_evals):
  # experiment configs
  algo=tpe.suggest # other option is rand.suggest 
  spark_trials = SparkTrials(parallelism=3)
  mlflow.set_experiment(experiment_path)
  # start the training process
  with mlflow.start_run():
    best_result = fmin(fn=xgboost, space=params_space, 
                       algo=algo, max_evals=max_evals, trials=spark_trials)

# COMMAND ----------

# Shallow neural nets 

def pred_nn(model, threshold, x_values):
  class_preds = model.predict_proba(x_values)
  output = np.copy(class_preds[:, list(model.classes_).index(1)])
  output[output >= threshold] = 1
  output[output < threshold] = 0
  return output, class_preds


def nn(params):  
  # extract parameters
  loss = params['loss']
  threshold = params['threshold']
  # set the tags values
  log_tags(params)
  # log the artifacts
  log_pre_processing(params)
  # load datasets
  x_train, y_train, x_validate, y_validate, x_test, y_test, data_columns  = load_datasets(params['dataset_path'])
  features_columns = data_columns[:-1].tolist()
  mlflow.set_tag('features_columns', features_columns)
  # train the model 
  model = MLPClassifier(hidden_layer_sizes=(10, 10,)).fit(x_train, y_trian)
  # calculate predection
  pred_validate, _ = pred_nn(model, threshold, x_validate)
  pred_test, _ = pred_nn(model, threshold, x_test)
  # calculate and log metrics for validation set
  validation_metrics = metrics(pred_validate, y_validate, suffix='validate')
  # calculate and log metrics for test set
  test_metrics = metrics(pred_test, y_test, suffix='test')
  # calculate and log confidence between test and validation sets
  conf_dict = confidence(validation_metrics, test_metrics)
  # save predictions
  save_results(y_validate, y_test, pred_validate, pred_test)
  # register model
  mlflow.sklearn.log_model(model, 'nn_model')
  # function return
  if loss=='normal':
    return {'loss': -1*(validation_metrics['f_c_1_validate'] + validation_metrics['f_c_0_validate']), 'status': STATUS_OK}
  else:
    return {'loss': -1*(0.7*validation_metrics['r_c_1_validate'] + 0.3*validation_metrics['p_c_1_validate']), 'status': STATUS_OK}
  
def train_nn(experiment_path, params_space, max_evals):
  # experiment configs
  algo=tpe.suggest # other option is rand.suggest 
  spark_trials = SparkTrials(parallelism=3)
  mlflow.set_experiment(experiment_path)
  # params configs
  '''
  params_space = {'threshold': hp.uniform('threshold', 0.2, 0.5),
                  'dataset_path': dataset_path,
                  'model_type': model_type,
                  'model_name': model_name,
                  'training_log_path': training_log_path,
                  'register_model': register_model, 
                  'dataset_size': dataset_size,
                  'data_query': data_query,
                  'pre_processing': pre_processing,
                  'loss': 'fishing_net'}
  '''
  # start the training process
  with mlflow.start_run():
    best_result = fmin(fn=nn, space=params_space, 
                       algo=algo, max_evals=max_evals, trials=spark_trials)