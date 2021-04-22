##### SALARY PREDICTION
# Use Kaggle' Job Salary Prediction Dataset

import time, datetime, sys
import json
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

import h2o
from h2o.estimators.word2vec import H2OWord2vecEstimator
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.random_forest import H2ORandomForestEstimator
from h2o.estimators.deeplearning import H2ODeepLearningEstimator
from h2o.grid import H2OGridSearch
from h2o.estimators import H2OXGBoostEstimator

##### CONSTANTS
QUICK_ROWS = 0  # quick test. 0: all, >0: just use head with QUICK_ROWS (should above 3000)
USE_SAVED_FRAME = True
USE_SAVED_W2V_MODEL = True
USE_SAVED_MODEL = False

DATA_FILE = "/Documents/MLProject/job-salary-prediction/Train_rev1.csv"
RESPONSE_NAME = "SalaryNormalized"
STOPPING_METRIC = "mae"

INOUT_DIR = "Documents/MLProject/job-salary-prediction/inout"
RESTART_H2O = True  # To release all resources of H2O after heavy jobs
SHUTDOWN_H2O = False

WORD_VEC_SIZE = 200
EPOCHS = 5
GOOD_DISTANCE_PERCENTAGE = 30
SEED = 12345678901
VERBAL = False

FULLDESC_W2V_MODEL_NAME = "fulldesc_w2v_model"
TITLE_W2V_MODEL_NAME = "title_w2v_model"
VEC_JOB_DF_NAME = "vec_job_df"
VECCAT_JOB_DF_NAME = "veccat_job_df"
BEST_SAL_MODEL_NAME = "best_sal_model"

STOP_WORDS = ["ax", "i", "you", "edu", "s", "t", "m", "subject", "can", "lines", "re", "what",
              "there", "all", "we", "one", "the", "a", "an", "of", "or", "in", "for", "by", "on",
              "but", "is", "in", "a", "not", "with", "as", "was", "if", "they", "are", "this", "and", "it", "have",
              "from", "at", "my", "be", "by", "not", "that", "to", "from", "com", "org", "like", "likes", "so"
    , "our", "will", "looking", "both"
    , "vacancies", "candidate", "up", "usd", "exp", "welcome", "urgent", "hot", "good", "new", "attractive", "___",
              "years", "year", "salary", "need", "needed", "very"
    , "bị", "bởi", "cả", "các", "cái", "cần", "càng", "chỉ", "chiếc", "cho", "chứ", "chưa", "chuyện", "có", "có_thể",
              "cứ", "của", "cùng", "cũng", "đã", "đang", "đây", "để", "đến_nỗi", "đều", "điều", "do", "đó", "được",
              "dưới", "gì", "khi", "không", "là", "lại", "lên", "lúc", "mà", "mỗi", "một_cách", "này", "nên", "nếu",
              "ngay", "nhiều", "như", "nhưng", "những", "nơi", "nữa", "phải", "qua", "ra", "rằng", "rằng", "rất", "rất",
              "rồi", "sau", "sẽ", "so", "sự", "tại", "theo", "thì", "trên", "trước", "từ", "từng", "và", "vẫn", "vào",
              "vậy", "vì", "việc", "với", "vừa"
              ]


##### FUNCTIONS
def niceTime():
    return datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')


def StartOfFunc(funcname=""):
    print(niceTime(), '- START ', funcname)


def EndOfFunc(funcname=""):
    print(niceTime(), '- END ', funcname)


def tokenize(sentences, stop_word=STOP_WORDS):
    StartOfFunc(sys._getframe().f_code.co_name)
    tokenized = sentences.tokenize("\\W+")
    if VERBAL: print(niceTime(), "tokenized")

    tokenized_lower = tokenized.tolower()
    tokenized_filtered = tokenized_lower[(tokenized_lower.nchar() >= 2) | (tokenized_lower.isna()), :]
    if VERBAL: print(niceTime(), "filtered")

    tokenized_words = tokenized_filtered[tokenized_filtered.grep("[0-9]", invert=True, output_logical=True), :]
    tokenized_words = tokenized_words[(tokenized_words.isna()) | (~ tokenized_words.isin(STOP_WORDS)), :]
    if VERBAL: print("tokenized_words:\n", tokenized_words.describe())

    EndOfFunc(sys._getframe().f_code.co_name)
    return tokenized_words


def load_data(data_file):
    StartOfFunc(sys._getframe().f_code.co_name)

    vjobs_df = h2o.import_file(data_file, destination_frame="vjob",
                               col_names=["Id", "Title", "FullDescription", "LocationRaw", "LocationNormalized",
                                          "ContractType",
                                          "ContractTime", "Company", "Category", "SalaryRaw", "SalaryNormalized",
                                          "SourceName"],
                               col_types=["int", "string", "string", "enum", "enum", "enum",
                                          "enum", "enum", "enum", "string", "int", "string"], header=1)

    vjobs_df = vjobs_df.drop(['Id', 'SalaryRaw', 'LocationRaw'])  # use ignore_columns not work
    if (QUICK_ROWS > 0): vjobs_df = vjobs_df.head(QUICK_ROWS)

    print(niceTime(), "Data description")
    if VERBAL: vjobs_df.describe()

    #     topCats=['IT Jobs'] #,'Engineering Jobs', 'Accounting & Finance Jobs'
    #     data_df=vjobs_df[vjobs_df['Category'].isin(topCats)]
    #     if VERBAL: data_df.describe()
    data_df = vjobs_df

    EndOfFunc(sys._getframe().f_code.co_name)
    return data_df


def split(df):
    StartOfFunc(sys._getframe().f_code.co_name)

    [train_df, valid_df, test_df] = df.split_frame(ratios=[0.7, 0.15], seed=SEED)
    train_df.frame_id = df.frame_id + "_train"
    valid_df.frame_id = df.frame_id + "_valid"
    test_df.frame_id = df.frame_id + "_test"

    EndOfFunc(sys._getframe().f_code.co_name)
    return [train_df, valid_df, test_df]


def find_synonyms(word, w2v_model):
    print(w2v_model.find_synonyms(word, count=5))


def vectorize_title(job_df, use_saved_model=False):
    StartOfFunc(sys._getframe().f_code.co_name)

    print("Break Title into sequence of words")
    words = tokenize(job_df["Title"])

    if use_saved_model:
        print("Load w2v model from " + INOUT_DIR + "/" + TITLE_W2V_MODEL_NAME)
        w2v_model = h2o.load_model(INOUT_DIR + "/" + TITLE_W2V_MODEL_NAME)
    else:
        print("Build word2vec model")
        w2v_model = H2OWord2vecEstimator(sent_sample_rate=0.0, epochs=EPOCHS, vec_size=WORD_VEC_SIZE)
        w2v_model.train(training_frame=words)
        w2v_model.model_id = TITLE_W2V_MODEL_NAME

        model_path = h2o.save_model(model=w2v_model, path=INOUT_DIR, force=True)
        print("w2v_model saved to: ", model_path)

    print("Calculate a vector for each job title")
    job_vecs = w2v_model.transform(words, aggregate_method="AVERAGE")
    if VERBAL: print("job_vecs.describe:\n", job_vecs.describe())
    vec_job_df = job_df.cbind(job_vecs)
    vec_job_df.frame_id = "vec_job_df"
    vec_job_df = vec_job_df.drop('Title')

    EndOfFunc(sys._getframe().f_code.co_name)

    return vec_job_df


def preprocess(job_df, use_saved_model=False):
    ''' Return 2 dataframes
        1. Vectorize both Title and FullDescription, vec_job_df
        2. Categorize Title and vectorize FullDescription, veccat_job_df
    '''
    StartOfFunc(sys._getframe().f_code.co_name)

    vec_job_df = vectorize_title(job_df, use_saved_model=use_saved_model)
    if VERBAL: print(vec_job_df.describe())

    print(niceTime(), "Convert Title to category data type")
    veccat_job_df = job_df
    veccat_job_df['Title'] = veccat_job_df['Title'].asfactor()

    print(niceTime(), "Tokenize")
    words = tokenize(job_df["FullDescription"])
    if VERBAL: words.describe()

    if use_saved_model:
        print("Load w2v model from " + INOUT_DIR + "/" + FULLDESC_W2V_MODEL_NAME)
        w2v_model = h2o.load_model(INOUT_DIR + "/" + FULLDESC_W2V_MODEL_NAME)
    else:
        print(niceTime(), "Train with H2OWord2vecEstimator")
        w2v_model = H2OWord2vecEstimator(sent_sample_rate=0.0001, epochs=EPOCHS, vec_size=WORD_VEC_SIZE)
        w2v_model.train(training_frame=words)
        w2v_model.model_id = FULLDESC_W2V_MODEL_NAME

        model_path = h2o.save_model(model=w2v_model, path=INOUT_DIR, force=True)
        print("w2v_model saved to: ", model_path)

    print(niceTime(), "Calculate a vector for each FullDescription")
    vecs = w2v_model.transform(words, aggregate_method="AVERAGE")
    print(niceTime(), "Prepare training&validation data (keep only FullDescription made of known words)")
    valid_data = ~ vecs["C1"].isna()
    if VERBAL: valid_data.describe()

    print(niceTime(), "Combine vec column")
    vec_job_df = vec_job_df[valid_data, :].cbind(vecs[valid_data, :])
    vec_job_df.frame_id = "vec_job_df"
    vec_job_df = vec_job_df.drop('FullDescription')

    veccat_job_df = veccat_job_df[valid_data, :].cbind(vecs[valid_data, :])
    veccat_job_df.frame_id = "veccat_job_df"
    veccat_job_df = veccat_job_df.drop('FullDescription')

    EndOfFunc(sys._getframe().f_code.co_name)

    return [vec_job_df, veccat_job_df]


def trainGBM(train_df, valid_df, col_sample_rate=1):
    StartOfFunc(sys._getframe().f_code.co_name)

    print("Build a GBM model")
    gbm_model = H2OGradientBoostingEstimator(nfolds=5, ignored_columns=['Id', 'SalaryRaw', 'LocationRaw'], ntrees=150,
                                             seed=SEED, learn_rate=0.5, stopping_tolerance=0.05,
                                             col_sample_rate=col_sample_rate)

    gbm_model.train(x=train_df.names,
                    y=RESPONSE_NAME,
                    training_frame=train_df,
                    validation_frame=valid_df)

    print(gbm_model.model_performance(valid=True))
    EndOfFunc(sys._getframe().f_code.co_name)

    return gbm_model


"""def trainRF(train_df, valid_df, col_sample_rate=1):
    StartOfFunc(sys._getframe().f_code.co_name)

    model = H2OGradientBoostingEstimator(nfolds=5, ignored_columns=['Id', 'SalaryRaw', 'LocationRaw'], ntrees=150,
                                         seed=SEED, learn_rate=0.5, stopping_tolerance=0.05,
                                         col_sample_rate=col_sample_rate)

    model.train(x=train_df.names,
                y=RESPONSE_NAME,
                training_frame=train_df,
                validation_frame=valid_df)

    print(gbm_model.model_performance(valid=True))
    EndOfFunc(sys._getframe().f_code.co_name)

    return gbm_model"""


def trainDL(train_df, valid_df):
    print("Build a DL model")
    model = H2ODeepLearningEstimator(nfolds=5, seed=SEED, stopping_metric=STOPPING_METRIC
                                     , hidden=[200, 200]
                                     , epochs=8
                                     , rate=0.005
                                     # Learning rate (higher => less stable, lower => slower convergence). Default 0.005. With 0.003 I see unstable mae (up/down deviance a lot)
                                     , sparse=True
                                     # Sparse data handling (more efficient for data with lots of 0 values). Default False.
                                     , stopping_tolerance=0.1
                                     # The relative tolerance for the metric-based stopping to stop training if the improvement is less than this value.
                                     , activation='rectifier_with_dropout'
                                     , input_dropout_ratio=0.25
                                     , hidden_dropout_ratios=[0.6, 0.6]
                                     # Hidden layer dropout ratios (can improve generalization), specify one value per hidden layer, defaults to 0.5.
                                     , missing_values_handling='mean_imputation'
                                     # ``"mean_imputation"``, ``"skip"``  (default: ``"mean_imputation"``).
                                     )

    model.train(x=train_df.names,
                y=RESPONSE_NAME,
                training_frame=train_df,
                validation_frame=valid_df)

    print(model.model_performance(valid=True))
    EndOfFunc(sys._getframe().f_code.co_name)

    return model


def grid_init_GBM():
    if (QUICK_ROWS > 0):
        l_ntrees = 3
    else:
        l_ntrees = 1000

    param = {
        "ignored_columns": ['Id', 'SalaryRaw', 'LocationRaw']
        , "seed": SEED  # potentially result in overfitting to a particular random sample selected
        , "nfolds": 4  # default 0, 5-10 is good but 10 will take more time
        #         , 'tree_method': 'hist'
        #         , 'grow_policy': 'lossguide'
        #         , 'max_bins':1
        #         , 'max_leaves':1
        #         , 'min_sum_hessian_in_leaf': 0.1
        #         , 'min_data_in_leaf':10.0
        #         , "stopping_rounds": 3
        #         , "min_rows": 16
        #         #     , "col_sample_rate_per_tree" : 0.9
        #         #     , "min_rows" : 5
        #         #     , "score_tree_interval": 100
    }
    hyper_parameters = {
        'ntrees': l_ntrees
        # default 50 and CV not change, 10 CV(folds) still have save ntrees 50). this should be tuned using CV for a particular learning rate??
        , 'max_depth': [10]
        # default 5 and CV not change (Should be tuned using CV??) Control over-fitting, higher may overfitting
        , 'col_sample_rate': [0.8]
        , "sample_rate": [0.85]
        , "histogram_type": 'quantiles_global'
        , 'learn_rate': [0.1]
        , 'min_rows': [60]  # default 10, higher for preventing overfitting.
        #           'min_split_improvement': [0.0005]
        , 'nbins_cats': [800]  # default 1024, higher values can lead to more overfitting.
        #         , 'stopping_rounds': [5]
        #         , 'stopping_tolerance': [0.001]
    }

    print("hyper_parameters: ", hyper_parameters)

    grid_search = H2OGridSearch(H2OGradientBoostingEstimator(**param)
                                , hyper_params=hyper_parameters)

    return grid_search


def grid_init_RF():
    '''
       Too slow: grid search with trees [50,100], max depth [20, 40] H2O est. 5 hours to run!
      try trees [50,100] max depth [15,30]: 4 hours
    '''
    if (QUICK_ROWS > 0):
        l_ntrees = 3
    else:
        l_ntrees = 500

    param = {
        "ignored_columns": ['Id', 'SalaryRaw', 'LocationRaw']
        , "seed": SEED  # potentially result in overfitting to a particular random sample selected
        , "nfolds": 4  # default 0, 5-10 is good but 10 will take more time
        #         , 'tree_method': 'hist'
        #         , 'grow_policy': 'lossguide'
        #         , 'max_bins':1
        #         , 'max_leaves':1
        #         , 'min_sum_hessian_in_leaf': 0.1
        #         , 'min_data_in_leaf':10.0
        #         , "stopping_rounds": 3
        #         , "min_rows": 16
        #         #     , "col_sample_rate_per_tree" : 0.9
        #         #     , "min_rows" : 5
        #         #     , "score_tree_interval": 100
    }
    hyper_parameters = {
        'ntrees': l_ntrees
        , 'max_depth': 10
        #         , 'col_sample_rate': [0.9]
        #         , 'learn_rate': learn_rate
        #           'min_split_improvement': [0.0005]
        #         , 'nbins_cats': [800]
        #         , 'stopping_rounds': [5]
        #         , 'stopping_tolerance': [0.001]
    }

    print("hyper_parameters: ", hyper_parameters)

    grid_search = H2OGridSearch(H2ORandomForestEstimator(**param)
                                , hyper_params=hyper_parameters)

    return grid_search


def grid_init_XGBoost():
    if (QUICK_ROWS > 0):
        l_ntrees = [2, 3]
    else:
        l_ntrees = [1400]

    # http://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html
    # http://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/xgboost.html
    hyper_parameters = {
        'ntrees': l_ntrees
        , 'col_sample_rate': [0.6]
        , 'learn_rate': [0.03]  # defaults 0.3
        , 'max_leaves': [900]
        , 'max_bins': [63]
        # default 256 # the smaller the faster but almost same accuracy, read more: http://lightgbm.readthedocs.io/en/latest/GPU-Performance.html
        #           'min_split_improvement': [0.0005]
        #         , 'nbins_cats': [800]
        #         , 'stopping_rounds': [3]
        #         , 'subsample': 0.9
        #         , 'colsample_bytree': 0.75
        #         , 'reg_alpha': [0.1]
        #         , 'reg_lambda': [0.1]
        #         , 'stopping_tolerance': [0.001]
    }

    param = {
        "seed": SEED  # potentially result in overfitting to a particular random sample selected
        , 'stopping_metric': STOPPING_METRIC
        , "nfolds": 3  # default 0, 5-10 is good but 10 will take more time
        , 'tree_method': 'hist'
        , 'grow_policy': 'lossguide'
        , 'max_depth': 0
        #             , 'min_sum_hessian_in_leaf': 0.1
        #             , 'min_data_in_leaf':3.0
        , "stopping_rounds": 3
        #         , "min_rows": 16
        #         #     , "col_sample_rate_per_tree" : 0.9
        #         #     , "min_rows" : 5
        #         #     , "score_tree_interval": 100
        #       , "ignored_columns": ['Id', 'SalaryRaw', 'LocationRaw']  # not works

    }

    print("hyper_parameters: ", hyper_parameters)
    print("parameters: ", param)

    grid_search = H2OGridSearch(H2OXGBoostEstimator(**param)
                                , hyper_params=hyper_parameters)

    return grid_search


def grid_train(train_df, valid_df, algorithm=""):
    StartOfFunc(sys._getframe().f_code.co_name)
    '''
    Grid search to find optimum parameters


    Note:
    + I tried categorical_encoding='one_hot_explicit', histogram_type='quantiles_global' for GBM 
    with only [20,50] trees, max depth 7. H2O shows 46 hours to run! 
    When try only one_hot_explicit, even with col_sample_rate 0.3, H2O still estimates 6 hours.
    So I don't use one_hot_explicit any more in this project, will study more when have time.

    '''

    if (algorithm == "" or algorithm == H2OGradientBoostingEstimator.__name__):
        grid_search = grid_init_GBM()

    elif (algorithm == H2OXGBoostEstimator.__name__):
        grid_search = grid_init_XGBoost()

    elif (algorithm == H2ORandomForestEstimator.__name__):
        grid_search = grid_init_RF()

    grid_search.train(x=train_df.names,
                      y=RESPONSE_NAME,
                      training_frame=train_df,
                      validation_frame=valid_df)

    grid = h2o.get_grid(grid_search.grid_id)
    grid.show()

    EndOfFunc(sys._getframe().f_code.co_name)

    return grid


def show_score_his(model, interateType="Trees"):
    StartOfFunc(sys._getframe().f_code.co_name)

    score_history = model.scoring_history()
    plt.title('Scoring history')
    plt.xlabel(interateType)
    plt.ylabel(STOPPING_METRIC)

    x = 0
    if (interateType == "Trees"):
        x = score_history.number_of_trees
    elif (interateType == "Epochs"):
        x = score_history.epochs

    y1 = score_history.training_mae
    plt.plot(x, y1, '-b', label='Train')

    y2 = score_history.validation_mae
    plt.plot(x, y2, '-r', label='Valid')

    plt.legend(loc='upper right')

    EndOfFunc(sys._getframe().f_code.co_name)


def evaluate(model, test_df):
    StartOfFunc(sys._getframe().f_code.co_name)

    pred_df = model.predict(test_df)
    perf = model.model_performance(test_data=test_df)

    print("perf.mae:", perf.mae)

    combined_df = pred_df.cbind(test_df)
    # combined_df.describe()
    diff_df = test_df[RESPONSE_NAME] - pred_df
    diff_df = diff_df.abs()
    diff_df.set_name(0, 'pred_distance')
    # diff_df.describe()
    # diff_df.hist()

    diff_percentage_df = diff_df * 100 / test_df[RESPONSE_NAME]
    diff_percentage_df.set_name(0, 'pred_distance_percentage')
    good_pred_count = len(
        diff_percentage_df[diff_percentage_df['pred_distance_percentage'] <= GOOD_DISTANCE_PERCENTAGE])
    total_count = len(diff_percentage_df)
    good_pred_rate = np.round(good_pred_count * 100 / total_count)
    print("Good pred %: ", good_pred_rate, " (", good_pred_count, "/", total_count, ")")
    diff_percentage_df.hist()

    EndOfFunc(sys._getframe().f_code.co_name)
    return [good_pred_rate, perf.mae]


def train_evaluate(data_df):
    '''
    Train and evaluate models
    '''
    StartOfFunc(sys._getframe().f_code.co_name)

    print(data_df.frame_id)
    [train_df, valid_df, test_df] = split(data_df)
    model = grid_train(train_df, valid_df, algorithm=H2OXGBoostEstimator.__name__)[0]
    #     model= grid_train(train_df, valid_df, algorithm = H2ORandomForestEstimator.__name__ )[0]
    #     model= grid_train(train_df, valid_df, algorithm = H2OGradientBoostingEstimator.__name__ )[0]
    #     model = trainDL(train_df, valid_df)

    [good_pred_rate, mae] = evaluate(model, test_df)

    show_score_his(model)

    try:
        model.varimp_plot()
    except Exception:
        print("Error with model.varimp_plot()")

    model_param_file_name = INOUT_DIR + "/" + model.model_id + "_params.json"
    print(niceTime(), "Save model params to ", model_param_file_name)
    with open(model_param_file_name, 'w') as outfile:
        json.dump(model.get_params(), outfile)

    EndOfFunc(sys._getframe().f_code.co_name)
    return [good_pred_rate, mae, model]


############################### MAIN ####################################
def main():
    start_time = niceTime()
    if (RESTART_H2O and h2o.cluster()):
        h2o.cluster().shutdown()
        time.sleep(2)

    h2o.init(max_mem_size="4G")
    niceTime()
    if USE_SAVED_FRAME:
        print("USE SAVED FRAME IN ", INOUT_DIR)
        vec_job_df = h2o.import_file(path=INOUT_DIR + '/' + VEC_JOB_DF_NAME)
        #     veccat_job_df = h2o.import_file(path=INOUT_DIR + '/' + VECCAT_JOB_DF_NAME)
        if (QUICK_ROWS > 0): vec_job_df = vec_job_df.head(QUICK_ROWS)

    else:
        print("LOAD DATA FILE ", DATA_FILE)
        job_df = load_data(DATA_FILE)
        job_df.frame_id = "job_df"
        [vec_job_df, veccat_job_df] = preprocess(job_df, use_saved_model=USE_SAVED_W2V_MODEL)
        vec_job_df.frame_id = VEC_JOB_DF_NAME
        veccat_job_df.frame_id = VECCAT_JOB_DF_NAME

        h2o.export_file(frame=vec_job_df, path=INOUT_DIR + '/' + VEC_JOB_DF_NAME, force=True)
        h2o.export_file(frame=veccat_job_df, path=INOUT_DIR + '/' + VECCAT_JOB_DF_NAME, force=True)

    vec_job_df = vec_job_df.drop(['Id'])

    vec_job_df.describe()
    # veccat_job_df.describe()
    if USE_SAVED_MODEL:
        print("USE SAVED MODEL IN ", INOUT_DIR)
        best_model = h2o.load_model(INOUT_DIR + '/' + BEST_SAL_MODEL_NAME)
        best_model
        # Later ...     best_pred_rate = evaluate(best_model, veccat_job_df)
    else:
        print("")
        #     if (USE_SAVED_FRAME == False): train_evaluate(vec_job_df)
        [vec_good_pred_rate, vec_mae, vec_model] = train_evaluate(vec_job_df)

        # Too SLOW, up to 80 hours (estimated) to run so not use it:
        if False:
            [veccat_good_pred_rate, veccat_mae, veccat_model] = train_evaluate(veccat_job_df)
            if veccat_good_pred_rate > vec_good_pred_rate:
                best_model = veccat_model
                best_pred_rate = veccat_good_pred_rate
                best_mae = veccat_mae

        best_model = vec_model
        best_pred_rate = vec_good_pred_rate
        best_mae = vec_mae

        best_model.model_id = BEST_SAL_MODEL_NAME
        print("Save best model to ", INOUT_DIR)
        model_path = h2o.save_model(model=best_model, path=INOUT_DIR, force=True)

        print("\n\n**************** BEST PREDICTION RATE: ", best_pred_rate, " with MAE: ", best_mae, "***************")

    model = h2o.get_model('best_sal_model')

    model.scoring_history()


main()

