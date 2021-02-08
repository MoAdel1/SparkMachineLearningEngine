# Databricks notebook source
# processing functions

def categorical_indexing(df, column, indexer=None):
  # string index
  if (indexer == None):
    stringIndexer = StringIndexer(inputCol=column, outputCol='{}_INDEXED'.format(column))
    indexer = stringIndexer.fit(df)
  indexed = indexer.transform(df)
  return indexed, indexer

def categorical_hotencoding(df, column, categories=None):
  if(categories == None):
    categories = df.select(column).distinct().rdd.flatMap(lambda x : x).collect()
    categories.sort()
  for category in categories:
    function = udf(lambda item: 1 if item == category else 0, IntegerType())
    new_column_name = column+'_'+category.replace('"', '').upper()
    df = df.withColumn(new_column_name, function(col(column)))
  df = df.drop(column)
  return df, categories

def numerical_standardize(df, column, mean_value=None, sttdev_value=None):
  if(mean_value == None and sttdev_value == None):
    mean_value, sttdev_value = df.select(mean(column), stddev(column)).first()
  df = df.withColumn('{}_STANDARD'.format(column), (col(column) - mean_value) / sttdev_value)
  df = df.drop(column)
  return df, mean_value, sttdev_value

def classify_target(df, column):
  function = udf(lambda item: 1 if item > 0 else 0, IntegerType())
  df = df.withColumn('{}_CLASS'.format(column), function(col(column)))
  return df

def split_data_classification(df, train_percent, test_percent, validate_percent, random_state):
  # normalize percentage
  train_percent = train_percent/100
  test_percent = test_percent/100
  validate_percent = validate_percent/100
  # convert data into numpy 
  data = np.array(df.collect(), dtype=np.float64)
  # get number of records
  records = data.shape[0]
  validate_size = int(validate_percent * records)
  test_size = int(test_percent * records)
  train_size = records - (validate_size + test_size)
  # split data
  data_train, data_remains = train_test_split(data, train_size=train_size, random_state=random_state, stratify=data[:, -1:])
  if validate_size == 0:
    data_test = data_remains
    data_validate = None
  else:
    data_validate, data_test = train_test_split(data_remains, train_size=validate_size, random_state=random_state, stratify=data_remains[:, -1:])
  # get x and y arrays
  x_train = data_train[:, :-1]
  y_train = data_train[:, -1:]
  x_validate = data_validate[:, :-1] if data_validate is not None else None
  y_validate = data_validate[:, -1:] if data_validate is not None else None
  x_test = data_test[:, :-1]
  y_test = data_test[:, -1:]
  # function return 
  return x_train, y_train, x_validate, y_validate, x_test, y_test

def split_data_regression(df, train_percent, test_percent, validate_percent, random_state):
  # normalize percentage
  train_percent = train_percent/100
  test_percent = test_percent/100
  validate_percent = validate_percent/100
  # convert data into numpy 
  data = np.array(df.collect(), dtype=np.float64)
  # get number of records
  records = data.shape[0]
  validate_size = int(validate_percent * records)
  test_size = int(test_percent * records)
  train_size = records - (validate_size + test_size)
  # split data
  data_train, data_remains = train_test_split(data, train_size=train_size, random_state=random_state, stratify=data[:, -1:])
  if validate_size == 0:
    data_test = data_remains
    data_validate = None
  else:
    data_validate, data_test = train_test_split(data_remains, train_size=validate_size, random_state=random_state, stratify=data_remains[:, -1:])
  # get x and y arrays
  x_train = data_train[:, :-2]
  y_train = data_train[:, -2:-1]
  x_validate = data_validate[:, :-2] if data_validate is not None else None
  y_validate = data_validate[:, -2:-1] if data_validate is not None else None
  x_test = data_test[:, :-2]
  y_test = data_test[:, -2:-1]
  # function return 
  return x_train, y_train, x_validate, y_validate, x_test, y_test

# COMMAND ----------

# dbfs data functions

def save_datasets(dataset_name, x_train, y_train, x_validate, y_validate, x_test, y_test, data_columns):
  # Save data to a local file first.
  data_filename = '{}.npz'.format(dataset_name)
  local_data_dir = tempfile.mkdtemp()
  local_data_path = os.path.join(local_data_dir, data_filename)
  np.savez(local_data_path, 
           x_train=x_train, 
           y_train=y_train, 
           x_validate=x_validate, 
           y_validate=y_validate, 
           x_test=x_test, 
           y_test=y_test,
           data_columns=data_columns)
  # Move it to DBFS, which is shared among cluster nodes.
  dbfs_tmp_dir = '/dbfs/ml/tmp/hyperopt'
  os.makedirs(dbfs_tmp_dir, exist_ok=True)
  dbfs_data_dir = tempfile.mkdtemp(dir=dbfs_tmp_dir)  
  dbfs_data_path = os.path.join(dbfs_data_dir, data_filename)  
  shutil.move(local_data_path, dbfs_data_path)
  return dbfs_data_path


def load_datasets(path):
  dataset = np.load(path, allow_pickle=True)
  x_train = dataset['x_train'] 
  y_train = dataset['y_train']
  x_validate = dataset['x_validate']
  y_validate = dataset['y_validate'] 
  x_test = dataset['x_test']
  y_test = dataset['y_test']
  data_columns = dataset['data_columns']
  return x_train, y_train, x_validate, y_validate, x_test, y_test, data_columns

# COMMAND ----------

# general ML training functions

def generate_weights(vector, threshold, value_1, value_2):
  output = np.copy(vector)
  output[output == threshold] = value_1
  output[output != threshold] = value_2
  return output

def generate_class(vector, threshold, value_1, value_2):
  output = np.copy(vector)
  output[output > threshold] = value_1
  output[output <= threshold] = value_2
  return output

def save_results(y_validate, y_test, pred_validate, pred_test):
  df_validate = pd.DataFrame({'Target': y_validate.reshape(-1), 'Predictions': pred_validate.reshape(-1)})
  df_test = pd.DataFrame({'Target': y_test.reshape(-1), 'Predictions': pred_test.reshape(-1)})
  df_validate.to_csv('validation_sample.csv', index=False)
  df_test.to_csv('test_sample.csv', index=False)
  mlflow.log_artifact('validation_sample.csv')
  mlflow.log_artifact('test_sample.csv')

  
def log_tags(params):
  mlflow.set_tag('model_type', params['model_type'])
  mlflow.set_tag('model_name', params['model_name'])
  mlflow.set_tag('training_log_path', params['training_log_path'])
  mlflow.set_tag('dataset_size', params['dataset_size'])
  mlflow.set_tag('data_query', params['data_query'])

  
def log_pre_processing(params):
  with open('pre_processing.pickle', 'wb') as file:
      pickle.dump(params['pre_processing'], file)
  mlflow.log_artifact('pre_processing.pickle')

  
def metrics(pred, actual, suffix=None, monitoring = False):
  # define class of predections
  pred_class = generate_class(pred, 0, 1, 0)
  actual_class = generate_class(actual, 0, 1, 0)
  # calculate metrics
  rmse = mean_squared_error(actual, pred, squared=False)
  mae = mean_absolute_error(actual, pred)
  r2 = r2_score(actual, pred)
  p_c_1, r_c_1, f_c_1, _ = precision_recall_fscore_support(actual_class, pred_class, average='binary', pos_label=1, warn_for = tuple())
  p_c_0, r_c_0, f_c_0, _ = precision_recall_fscore_support(actual_class, pred_class, average='binary', pos_label=0, warn_for = tuple())
  output =  {'rmse' if suffix == None else 'rmse_{}'.format(suffix): rmse,
             'mae' if suffix == None else 'mae_{}'.format(suffix): rmse,
             'r2' if suffix == None else 'r2_{}'.format(suffix): r2,
             'p_c_1' if suffix == None else 'p_c_1_{}'.format(suffix): p_c_1,
             'r_c_1' if suffix == None else 'r_c_1_{}'.format(suffix): r_c_1,
             'f_c_1' if suffix == None else 'f_c_1_{}'.format(suffix): f_c_1,
             'p_c_0' if suffix == None else 'p_c_0_{}'.format(suffix): p_c_0,
             'r_c_0' if suffix == None else 'r_c_0_{}'.format(suffix): r_c_0,
             'f_c_0' if suffix == None else 'f_c_0_{}'.format(suffix): f_c_0}
  # log metrics
  if monitoring == False:
    for key in output:
      mlflow.log_metric(key, output[key])
  return output


def confidence(v_metrics, t_metrics):
  output = dict()
  output['confidence_rmse'] = 100 - abs(100 * ((t_metrics['rmse_test'] - v_metrics['rmse_validate']) / v_metrics['rmse_validate']))
  output['confidence_rmse'] = 100 - abs(100 * ((t_metrics['mae_test'] - v_metrics['mae_validate']) / v_metrics['mae_validate']))
  output['confidence_r2'] = 100 - abs(100 * ((t_metrics['r2_test'] - v_metrics['r2_validate']) / v_metrics['r2_validate']))
  output['confidence_p_c_1'] = 100 - abs(100 * ((t_metrics['p_c_1_test'] - v_metrics['p_c_1_validate']) / v_metrics['p_c_1_validate']))
  output['confidence_p_c_0'] = 100 - abs(100 * ((t_metrics['p_c_0_test'] - v_metrics['p_c_0_validate']) / v_metrics['p_c_0_validate']))
  output['confidence_r_c_1'] = 100 - abs(100 * ((t_metrics['r_c_1_test'] - v_metrics['r_c_1_validate']) / v_metrics['r_c_1_validate']))
  output['confidence_r_c_0'] = 100 - abs(100 * ((t_metrics['r_c_0_test'] - v_metrics['r_c_0_validate']) / v_metrics['r_c_0_validate']))
  output['confidence_f_c_1'] = 100 - abs(100 * ((t_metrics['f_c_1_test'] - v_metrics['f_c_1_validate']) / v_metrics['f_c_1_validate']))
  output['confidence_f_c_0'] = 100 - abs(100 * ((t_metrics['f_c_0_test'] - v_metrics['f_c_0_validate']) / v_metrics['f_c_0_validate']))
  # log metrics
  for key in output:
    mlflow.log_metric(key, output[key])
  return output