# %%
#%pip install -r requirements.txt

# %%
#data work
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %%
#cleaning and reporting
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_selector as selector
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import MissingIndicator
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import classification_report

from boruta import BorutaPy
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold, cross_validate
from sklearn.metrics import average_precision_score, precision_recall_curve, f1_score, roc_auc_score, recall_score, precision_score, mean_absolute_error, mean_absolute_percentage_error

import cleanlab

#reporting
import mlflow


# %%
#models
import xgboost as xgb
from xgboost import XGBRegressor  
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import optuna
import joblib

#calibration
from nonconformist.base import RegressorAdapter
from nonconformist.icp import IcpRegressor
from nonconformist.nc import RegressorNc, AbsErrorErrFunc

# %%
df = pd.read_excel("SampleDonorData2.xlsx")
df.head()

# %%
#pulling out the targets and features to avoid data leakage later
target_b = df["TARGET_B"].copy().astype(int) #binary if donated or not
target_d = df["TARGET_D6"].copy().astype(int) #continuous - $'s donated
features =df.drop(columns=["TARGET_B", "TARGET_D6", "TARGET_D12", "TARGET_D18", "TARGET_D24"])

# %%
target_d.value_counts() #quick look at the breakdown

# %%
target_d[target_d <300].hist(bins=20) #visualizing frequency distributions

# %% [markdown]
# So as we would expect, heavy weight to 0 with a long right tail. We could think about taking the log of this value to bring in the tail some

# %%



#d split (dollar gift as a target)
X_train, X_test, y_train_d, y_test_d = train_test_split(
    features, target_d, test_size=0.05, random_state=42, stratify=target_b
)

#b split (boolean did/not they not donate)
y_test_b = target_b[y_test_d.index]
y_train_b = target_b[y_train_d.index]


#creating a validation dataset to help us choose the best hyperparamets/model and so test dataset can be final measure of estimated performance held out of sample
X_train, X_val, y_train_d, y_val_d = train_test_split(X_train, y_train_d, test_size=0.5, stratify=y_train_b)
y_val_b = y_train_b[y_val_d.index]
y_train_b = y_train_b[y_train_d.index]


X_train.reset_index(drop=True, inplace=True)
X_test.reset_index(drop=True, inplace=True)
y_train_d.reset_index(drop=True, inplace=True)
y_val_d.reset_index(drop=True, inplace=True)
y_test_d.reset_index(drop=True, inplace=True)
y_test_b.reset_index(drop=True, inplace=True)
y_val_b.reset_index(drop=True, inplace=True)
y_train_b.reset_index(drop=True, inplace=True)



print(features.shape)
features.head()

# %%
#These are our object data types we will need to one hot encode
print(X_train.select_dtypes(include = "object").head().shape)
X_train.select_dtypes(include = "object").head()


# %%
#These are our numeric data types
print(X_train.select_dtypes(include = "number").head().shape)
X_train.select_dtypes(include = "number").head()


# %%
X_train.describe()

# %% [markdown]
# all 24 columns are accounted for as either number or object, so let's solidify these into a list of columns

# %%
#selecting numeric cols and categorical columns into a list now as this may come in handy later
num_cols = X_train.select_dtypes(include = "number").head().columns.tolist()
cat_cols = X_train.select_dtypes(include = "object").head().columns.tolist()
num_cols

# %% [markdown]
# # Data Exploration
# 1. handle missing values & setup pipeline to enable early exploration
# 2. value distributions
# 3. correlation - gauge top performing columns to prepare for selection
# 4. noisy data instances/outliers - cleanlab

# %% [markdown]
# #### Step 1

# %%
#Before we setup the pipeline, let's check missing values
import pylab
def plot_missing_values(df):
    """ For each column with missing values plot proportion that is missing."""
    data = [(col, df[col].isnull().sum() / len(df)) 
            for col in df.columns if df[col].isnull().sum() > 0]
    col_names = ['column', 'percent_missing']
    missing_df = pd.DataFrame(data, columns=col_names).sort_values('percent_missing')
    pylab.rcParams['figure.figsize'] = (15, 8)
    missing_df.plot(kind='barh', x='column', y='percent_missing'); 
    plt.title('Percent of missing values in colummns')

plot_missing_values(X_train)

# %% [markdown]
# So a significant portion of values are missing for the top 3 variables in particular, we do not want to just drop this data/column, likely is a signal here

# %%
#creating a pipeline to do some preliminary cleaning so we can explore the data a bit easier with numerical methods
#using minmaxscaler to keep some sense of understanding about the intuition
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
])
from sklearn.preprocessing import FunctionTransformer
numerical_missing_onehot_transformer = Pipeline(steps=[
    ('missing_indicator', MissingIndicator()),
    
])

numerical_transformer = Pipeline(steps=[
    #('missing_indicator', MissingIndicator()),
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', MinMaxScaler()),
    
])

column_trans = ColumnTransformer(transformers=
        [('num_onehot_missing', numerical_missing_onehot_transformer, num_cols),
        ('num', numerical_transformer, num_cols), #selector(dtype_exclude="numeric")
        ('cat', categorical_transformer, cat_cols)], #selector(dtype_include="object")
    remainder='passthrough')
clean_data_pipeline = Pipeline(steps=[('preprocessor', column_trans)])

# %%
clean_data_pipeline.fit(X_train)

# %%
column_names = clean_data_pipeline.named_steps['preprocessor'].get_feature_names_out()
clean_df = pd.DataFrame(clean_data_pipeline.transform(X_train), columns=column_names)
clean_df_val = pd.DataFrame(clean_data_pipeline.transform(X_val), columns=column_names)
clean_df.head()

# %%
print(clean_df.shape)
clean_df.describe()

# %% [markdown]
# I'm not seeing any issues pop up with the min or max values here

# %% [markdown]
# #### Step 2 - value distributions
# 
# Let's take a look at the value distributions for our features to see if anything needs to be addressed

# %%
def viz_data_distribution(df):
    #viz
    plt.figure(figsize=(10,50))
    for i in range(len(df.columns)):
        try:
            plt.subplot(17, 1, i+1)
            sns.kdeplot(df[df.columns[i]])
            plt.title(df.columns[i])
        except ValueError as e:
            print(e)
            pass

    plt.tight_layout()
viz_data_distribution(clean_df)

# %% [markdown]
# We see some bell curves here which is good, nothing immediate it looks like hints at an issue

# %% [markdown]
# #### Step 3 - correlation/feature importance

# %%
#correlation with dollar target
clean_df.corrwith(y_train_d).sort_values(ascending=False)[0:15]

# %%
#correlations with boolean target
clean_df.corrwith(y_train_b).sort_values(ascending=False)[0:15]


# %% [markdown]
# looks like num_onehot_missing__missingindicator_CLUSTER_CODE and cat__URBANICITY_? are perfectly collinear so we may consider removing one of those. Interesting that missing donor age is highly correlated with both targets. Would be interesting to dive into what other features are correlated with those missing indicators to potentially uncover significant information troves to feed our modeling through feature engineering. 

# %%
#TODO: drop two columns above or use a feature selection capability in our pipeline

# %% [markdown]
# #### Step 4 - outliers/noisy data

# %% [markdown]
# let's take a quick look at outliers. We will rely on cleanlab to help us identify highly influential points

# %%
value_counts_d = y_train_d.value_counts().sort_index()
value_counts_d[value_counts_d.index > 200] # 
print(value_counts_d)
#value_counts_d[value_counts_d.index[value_counts_d.index < 2000]].hist(bins=20)

graph_value_counts_df = value_counts_d[value_counts_d.index[value_counts_d.index < 2000]]
value_counts_d.reset_index().plot.scatter(x="TARGET_D6", y= "count")


# %%
graph_value_counts_df.reset_index().plot.scatter(x="TARGET_D6", y= "count")

# %% [markdown]
# looks like values becomes fairly sparse after 1500. could look at grouping all data points above that to be 1500

# %%
#TODO: cutoff outliers with TARGET_D6 above 1500

# %% [markdown]
# looks look at cleanlab now for classification (target_b)

# %%
#below will help us weight to overcome class imbalance
sum_wpos = y_train_b.value_counts()[1]
sum_wneg = y_train_b.value_counts()[0]


rf = xgb.XGBClassifier(eval_metric = 'auc', use_label_encoder=True, scale_pos_weight = sum_wneg/sum_wpos) #ExtraTreesClassifier(class_weight="balanced")
cl = cleanlab.classification.CleanLearning(rf)
cl.fit(clean_df, y_train_b)
label_issues_b = cl.get_label_issues()

cl.fit(clean_df_val, y_val_b)
label_issues_b_val = cl.get_label_issues() #we will use this later since we are doing our continuous modeling on the validation dataset to better fit the
#continuous model to the output that stage 1 classification model hasn't overfit more to because it trained on it
#preds = cl.predict(clean_df)

# %%
#let's look at the overall breakdown here of value counts
print(label_issues_b['given_label'].value_counts())
label_issues_b['predicted_label'].value_counts()

# %% [markdown]
# so we are slightly overclassifying people as going to donate now, likely can be solved during optimization later as the weighting might be too high on positive cases

# %%
print(label_issues_b['is_label_issue'].sum())


def view_datapoint(index, label_issues, X_train_raw):
    """combines the original uncleaed feature column to show the lowest quality labels"""
    given_labels = label_issues["given_label"]
    predicted_labels = label_issues["predicted_label"].round(1)
    return pd.concat(
        [X_train_raw, given_labels, predicted_labels], axis=1
    ).iloc[index]

lowest_quality_labels = label_issues_b["label_quality"].argsort()[:200].to_numpy()
low_label_quality = view_datapoint(lowest_quality_labels, label_issues_b, X_train.reset_index(drop=True))
#print(low_label_quality['given_label'].value_counts())
#print(low_label_quality['predicted_label'].value_counts())
low_label_quality

# %%
#look at classification report to see confusion matrix
print(classification_report(low_label_quality['given_label'], low_label_quality['predicted_label']))

# %% [markdown]
# So we are seeing that for the noisiest labels, most of them are from failing to recognize that someone donated when they actually did. But noisy just equates to we are getting them all wrong here. 

# %%
#let's see distribution of label quality by label, with low label quality being bad
ax = sns.histplot(data=label_issues_b, x='label_quality', hue="given_label")
ax.set(ylim=(0, 300))
#ax.set_xticks(bins)

# %% [markdown]
# pretty much just there is a group at the bottom that have the worst scores but they are all for the label being 1. Let's see how this group is different from the rest of the group with label = 1

# %%
def compare_standardized_differences_across_slices(df, slice_indices, numeric_columns):
    """
    Compares the averages of rows in slice_indices with the overall DataFrame averages for numeric columns,
    standardizing the differences by the standard deviation of each column.

    Args:
        df: The DataFrame to analyze.
        slice_indices: A list of integers representing the rows to include in the comparison.
        numeric_columns: The list of numeric columns to compare averages for.

    Returns:
        A dictionary containing the standardized differences in averages.
    """

    slice_df = df.iloc[slice_indices]
    differences = {}
    for col in numeric_columns:
        slice_avg = slice_df[col].mean()
        overall_avg = df[col].mean()
        difference = slice_avg - overall_avg
        std_dev = df[col].std()
        standardized_difference = difference / std_dev
        differences[col] = standardized_difference
    return differences

def compare_averages_across_slices(df, slice_indices, numeric_columns):
    """
    Compares the averages of rows in slice_indices with the overall DataFrame averages for numeric columns.

    Args:
        df: The DataFrame to analyze.
        slice_indices: A list of integers representing the rows to include in the comparison.
        numeric_columns: The list of numeric columns to compare averages for.

    Returns:
        A dictionary containing the differences in averages.
    """

    slice_df = df.iloc[slice_indices]
    differences = {}
    for col in numeric_columns:
        slice_avg = slice_df[col].mean()
        overall_avg = df[col].mean()
        differences[col] = slice_avg - overall_avg
    return differences

# %%
#here I am going to find the feature rows corresponding to each boolean target
neg_label_indices = y_train_b[y_train_b == 0].index
pos_label_indices = y_train_b[y_train_b == 1].index

lowest_quality_labels = label_issues_b["label_quality"].argsort()[:200].to_numpy() #just a fun to have

#then grab the selection of rows for each boolean target and where those rows are also identified as noisy so we can compare
#noisiest y=0 labels
noisy_neg_label_indices = label_issues_b.loc[neg_label_indices, :]["label_quality"].argsort()[:100].to_numpy()
#noisiest y=1 labels
noisy_pos_label_indices = label_issues_b.loc[pos_label_indices, :]["label_quality"].argsort()[:100].to_numpy()

# %%
#comparing noisy group against overall group for all no donation received in 6 months time group
compare_standardized_differences_across_slices(X_train.loc[neg_label_indices, :], noisy_neg_label_indices, num_cols ) #low quality avg - overall avg
#more negative means smaller number than overall avg

# %% [markdown]
# so for noisiest negative labels (predicting innacurately that they are all 1) we are seeing donor age is higher than avg, higher home value, and have gifted more recent than avg

# %%
#comparing noisy group against overall group for all do receive donation within 6 months time group
compare_standardized_differences_across_slices(X_train.loc[pos_label_indices, :], noisy_pos_label_indices, num_cols ) #low quality avg - overall avg
#more negative means smaller number than overall avg

# %% [markdown]
# for the noisiest positive labels (where we are missing people that actually donated) they are less likely to have achieved pep star donar status, gave a higher gift amount last time, and a lower cluster code than avg.
# 
# We could decide to create some features here to help separate out some of this features with different hard cutoffs. We will use Boruta to help us with this

# %%
#TODO: test out cutting off between 0.0 and 0.1 from label_issues_b, but \
#I don't think we are seeing too much noise here on average with those in the lowest quality scores

# %% [markdown]
# # Modeling - Create Model Architecture for Model Selection + Optimization
# 
# - using sklearn pipelines
# - cleaning pipelines used above
# - Boruta for feature selection
# - interaction terms for feature generation (to hopefully add information to now better capture samples explored in noisy data section)
# 
# 1. Classification Section
# - random forest, SVC, XGBoost to try and classification if someone will donate in next 6 months of not
# 
# 2. Continuous Section
# - given that the classification model in section 1 said someone would donate, how good are we are predicting exactly how much they will donate? We may still need to predict $0 here as almost a boosting strategy
# 
# 
# ### Pipeline Setup - useful function declaration

# %%


# %%
def gen_cleaning_pipeline():
    "the basic data cleaning pipeline from earlier we used"
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ])
    numerical_missing_onehot_transformer = Pipeline(steps=[
        ('missing_indicator', MissingIndicator()),
        
    ])

    numerical_transformer = Pipeline(steps=[
        #('missing_indicator', MissingIndicator()),
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', MinMaxScaler()),
        
    ])

    preprocessing_pipeline = ColumnTransformer(transformers=
            [('num_onehot_missing', numerical_missing_onehot_transformer, num_cols),
            ('num', numerical_transformer, num_cols), #selector(dtype_exclude="numeric")
            ('cat', categorical_transformer, cat_cols)], #selector(dtype_include="object")
        remainder='drop')
    #preprocessing_pipeline = #Pipeline(steps=[('preprocessor', column_trans)])
    return preprocessing_pipeline

def gen_feature_selection_pipeline():
    """uses Boruta to run feature selection in the pipeline for us"""
    fs_model = RandomForestClassifier(class_weight='balanced', n_estimators=100, max_depth=5, random_state=42) #feature selection model
    boruta_features = BorutaPy(
        verbose=0,
        estimator=fs_model,
        n_estimators='auto',
        max_iter=10,  
        random_state=42,
    )
    return boruta_features

def gen_iteraction_features_pipeline():
    """will iteract all features together to create additional information for the model"""
    interaction_generator = PolynomialFeatures(degree = 1, \
                interaction_only=False, include_bias=False)
    return interaction_generator

def gen_full_pipeline(model, params = {}):
    """pulls together all the pipeline pieces to reduce data leakage"""
    data_cleaner = gen_cleaning_pipeline()
    interaction_generator = gen_iteraction_features_pipeline()
    boruta_features = gen_feature_selection_pipeline()
    
    pipeline = Pipeline([
        ('data_cleaner', data_cleaner),
        ('interaction_gen', interaction_generator),
        #('feature_selection', boruta_features), 
        ('model', model)
    ])

    pipeline_best_params = {key:val for key,val in params.items() if key in pipeline.get_params()} #filter to make sure best params are in pipeline

    pipeline.set_params(**pipeline_best_params)

    return pipeline

def my_custom_scorer(estimator, X, y):
    """takes in an estimator (pipeline/model) and makes predictions to compare \
    against true targets to create the metrics listed. I prefer precision/recall \
    and their harmonic mean as they are less sensitive to outliers and easier to \
    interpret I find. """
    y_pred = estimator.predict(X)
    precision = precision_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    recall = recall_score(y, y_pred)
    return {'precision': precision, 'f1': f1, 'recall': recall}

def evaluate_model(model, X, y) -> dict:
    """Does the cross validation evaluation approach with KFold. takes in the model and runs it through a Cross Validation for the test metrics from custom scorer"""
    

    cv = RepeatedStratifiedKFold(n_splits=3, 
                                 n_repeats=2, 
                                 random_state=1)
    scores = cross_validate(model, X, y, 
                             scoring=['precision', 'f1', 'recall'], #my_custom_scorer, #'roc_auc', 
                             cv=cv, n_jobs=-1)
    print(scores.keys())
    averaged_scores = {"test_precision" : scores["test_precision"].mean(),
                       "test_f1" : scores["test_f1"].mean(),
                       "test_recall" : scores["test_recall"].mean()}
    return averaged_scores


# %%
def combine_indices(*indices_lists):
    """takes any number of indices and combines them into a set"""
    combined_indices = set()
    for indices in indices_lists:
        combined_indices.update(indices)
    return list(combined_indices)

def get_remove_indices_training_data(targets, target_filter_param, label_issues_df, label_issues_filter_param):
    """
    Takes in the training data, target values, and filter parameters, and returns indices to remove based on quality and outlier criteria.

    Args:
        targets (pd.Series or np.ndarray): The target values for the training data.
        target_filter_param (float): The threshold for outlier filtering based on target values.
        label_issues_df (pd.DataFrame): DataFrame containing label quality information.
        label_issues_filter_param (float): The threshold for filtering based on label quality.

    Returns:
        list: List of indices to remove from the training data.
    """
    label_issues_df = label_issues_df.reset_index(drop=True)
    targets = targets.reset_index(drop=True)

    indices_to_remove = label_issues_df[label_issues_df['label_quality'] < label_issues_filter_param].index  # Remove low-quality labels
    indices_to_remove2 = targets[targets > target_filter_param].index  # Remove target outliers
    all_indices_to_remove = combine_indices(indices_to_remove, indices_to_remove2)  # Combine indices, removing duplicates

    return all_indices_to_remove

# %%
#what will actually be used to run the experiments
optuna.logging.set_verbosity(optuna.logging.ERROR)


def run_study(X_train, y_train, X_val, y_val, objective, model):
    """will actually kickoff the optimization and 
    _____returns:
    grid : optuna object = holds various optuna statistics such as best_params and best_value (aka score)
    pipeline: sklearn pipeline object = pipeline object fitted with the best params from optuna study and ready for pipeline.fit(data, targets)"""
    pipeline = None
    
    study = optuna.create_study(**{
        'direction': 'maximize',
        'sampler': optuna.samplers.TPESampler(seed=37),
        'pruner': optuna.pruners.MedianPruner(n_warmup_steps=10),
    })
    #study.pruning_trigger = optuna.pruning.HyperbandPruning(n_epochs=10) 

    #mlflow.start_run():
        #mlflow_callback = MLflowCallback(tracking_uri=mlflow.get_tracking_uri(), metric_name='evaluation_metric')
        
    study.optimize(**{
        'func': lambda trial: objective(trial, X_train, y_train, X_val, y_val, model), #objective, 
        'n_trials': 40,
        'n_jobs': -1, #TODO: change back to -1 when not using mlflow
        'show_progress_bar': True,
        #'callbacks' : [mlflow_callback]
    })
    
    #end of mlflow.start_run() indent

    #study.best_params
    #study.best_value

    #below replaced with gen_full_pipeline param check
    #pipeline_params = gen_full_pipeline(model).get_params() #grab what the actual params are in our pipeline
    #pipeline_best_params = {key:val for key,val in study.best_params.items() if key in pipeline_params} #filter all best_params - double check
    #create new copy of the pipeline for outside optuna
    pipeline = gen_full_pipeline(model, study.best_params)
    train_indices_to_remove = get_remove_indices_training_data(y_train_d, 
                                                               study['data_filter__outlier_y'], 
                                                               label_issues_b, 
                                                               study['data_filter__label_issues_b']) 
    X_train = X_train.drop(train_indices_to_remove).reset_index(drop=True)
    y_train = y_train.drop(train_indices_to_remove).reset_index(drop=True)

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_val)
    with open("reports/results_report.txt", "a+") as results_report:
        results_report.write(str(model.__class__)+ '\n')
        results_report.write("Best Score:" + str(study.best_value)+ '\n')
        results_report.write("Model Setup:" + str(study.best_params) + '\n\n')
        results_report.write(str(classification_report(y_val, preds)))

    return study, pipeline



# %% [markdown]
# ### Classification Modeling - Stage 1: Will they donate in next 6 months or not?

# %% [markdown]
# below are my different objective functions I will optimize for each of the models in the below cell

# %%

model_options = {'xgb': xgb.XGBClassifier(eval_metric='logloss', use_label_encoder=False),
    'rf': RandomForestClassifier(class_weight='balanced'),
    'svc': SVC(class_weight='balanced', random_state=42)
}


# %%
mlflow.set_tracking_uri("http://localhost:5000")

# %%
#help to estimate roughly the model__scale_pos_weight 
sum_wpos = y_train_b.value_counts()[1]
sum_wneg = y_train_b.value_counts()[0]
print(f"best guess scale_pos_weight: {sum_wneg/sum_wpos}")


#different objectives
def xgb_class_objective(trial, X_train, y_train, X_val, y_val, model):
    #model = model_options['xgb']
    
    #non_pipeline_params = {
    #    #data filtering
    #    'data_filter__label_issues_b': trial.suggest_float('data_filter__label_issues_b', 0.0, 0.1),
    #    'data_filter__outlier_y': trial.suggest_int('data_filter__outlier_y', 1500, 2500),
    #}
    params = {
        #data filtering
        'data_filter__label_issues_b': trial.suggest_float('data_filter__label_issues_b', 0.0, 0.1),
        'data_filter__outlier_y': trial.suggest_int('data_filter__outlier_y', 1500, 2500),

        #global pipeline params
        'interaction_gen__interaction_only': trial.suggest_categorical('interaction_gen__interaction_only', [True, False]),

        #model specific params
        'model__subsample': trial.suggest_float('model__subsample', 0.5, 1.0),
        'model__eval_metric': trial.suggest_categorical('model__eval_metric', ['merror', 'mlogloss']), 
        #'model__lambda': trial.suggest_float('model__lambda', 0.8, 1.0), - since we aren't focusing on feature selection
        'model__n_estimators': trial.suggest_int('model__n_estimators', 100, 300),
        'model__max_depth': trial.suggest_int('model__max_depth', 4, 8),
        'model__learning_rate': trial.suggest_float('model__learning_rate', 0.01, 0.2),
        'model__colsample_bytree': trial.suggest_float('model__colsample_bytree', 0.5, 1.0),
        'model__scale_pos_weight': trial.suggest_float('model__scale_pos_weight', 1, 20), #higher since pos class is underweighted
        'model__gamma': trial.suggest_float('model__gamma', 0, 5),
        'model__reg_lambda': trial.suggest_float('model__reg_lambda', 1e-4, 10)
    }
    
    
    train_indices_to_remove = get_remove_indices_training_data(y_train_d, 
                                                               params['data_filter__outlier_y'], #remove outliers
                                                               label_issues_b, 
                                                               params['data_filter__label_issues_b']) #remove noisy labels
    X_train = X_train.drop(train_indices_to_remove).reset_index(drop=True)
    y_train = y_train.drop(train_indices_to_remove).reset_index(drop=True)

    
    pipeline = gen_full_pipeline(model = model, params = params) #create the full model pipeline
    scores = evaluate_model(pipeline, X_train, y_train)
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_val)

    #ps = precision_score(y_v, y_pred)
    #rs = recall_score(y_v, y_pred)
    f1_val = f1_score(y_val, y_pred)

    mlflow.log_params(params)
    mlflow.log_metrics({"precision_cv": scores["test_precision"]})
    mlflow.log_metrics({"recall_cv": scores["test_recall"]})
    mlflow.log_metrics({"f1_cv": scores["test_f1"]})
    mlflow.log_metrics({"f1_val": f1_val})

    trial.set_user_attr('precision_cv', scores["test_precision"])
    trial.set_user_attr('recall_cv', scores["test_recall"])
    trial.set_user_attr('f1_cv', scores["test_f1"])
    trial.set_user_attr('f1_val', f1_val)

    mlflow.set_tag("mlflow.runName", trial.number)
    mlflow.end_run()
    return f1_val

def rf_class_objective(trial, X_train, y_train, X_val, y_val, model):
    #model = model_options['rf']

    #non_pipeline_params = {
    #    #data filtering
    #    'data_filter__label_issues_b': trial.suggest_float('data_filter__label_issues_b', 0.0, 0.1),
    #    'data_filter__outlier_y': trial.suggest_int('data_filter__outlier_y', 1500, 2500),
    #}
    params = {
        #data filtering
        'data_filter__label_issues_b': trial.suggest_float('data_filter__label_issues_b', 0.0, 0.1),
        'data_filter__outlier_y': trial.suggest_int('data_filter__outlier_y', 1500, 2500),

        #global pipeline params
        'interaction_gen__interaction_only': trial.suggest_categorical('interaction_gen__interaction_only', [True, False]),

        #model specific params
        'model__n_estimators': trial.suggest_int('model__n_estimators', 100, 400),
        'model__max_depth': trial.suggest_int('model__max_depth', 5, 20),
        'model__min_samples_split': trial.suggest_int('model__min_samples_split', 2, 10),
        'model__min_samples_leaf': trial.suggest_int('model__min_samples_leaf', 1, 5),
        'model__class_weight': trial.suggest_categorical('model__class_weight', ['balanced', None]),
        'model__max_features': trial.suggest_categorical('model__max_features', ['sqrt', 'log2'])
    }

    train_indices_to_remove = get_remove_indices_training_data(y_train_d, 
                                                               params['data_filter__outlier_y'], #remove outliers
                                                               label_issues_b, 
                                                               params['data_filter__label_issues_b']) #remove noisy labels
    X_train = X_train.drop(train_indices_to_remove).reset_index(drop=True)
    y_train = y_train.drop(train_indices_to_remove).reset_index(drop=True)

    
    pipeline = gen_full_pipeline(model = model, params = params) #create the full model pipeline
    scores = evaluate_model(pipeline, X_train, y_train)
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_val)

    #ps = precision_score(y_v, y_pred)
    #rs = recall_score(y_v, y_pred)
    f1_val = f1_score(y_val, y_pred)

    mlflow.log_params(params)
    mlflow.log_metrics({"precision_cv": scores["test_precision"]})
    mlflow.log_metrics({"recall_cv": scores["test_recall"]})
    mlflow.log_metrics({"f1_cv": scores["test_f1"]})
    mlflow.log_metrics({"f1_val": f1_val})

    trial.set_user_attr('precision_cv', scores["test_precision"])
    trial.set_user_attr('recall_cv', scores["test_recall"])
    trial.set_user_attr('f1_cv', scores["test_f1"])
    trial.set_user_attr('f1_val', f1_val)

    mlflow.set_tag("mlflow.runName", trial.number)
    mlflow.end_run()
    return f1_val

def svc_class_objective(trial, X_train, y_train, X_val, y_val, model):
    #model = model_options['svc']

    #non_pipeline_params = {
    #    #data filtering
    #    
    #}

    params = {
        #data filtering
        'data_filter__label_issues_b': trial.suggest_float('data_filter__label_issues_b', 0.0, 0.1),
        'data_filter__outlier_y': trial.suggest_int('data_filter__outlier_y', 1500, 2500),

        #global pipeline params
        'interaction_gen__interaction_only': trial.suggest_categorical('interaction_gen__interaction_only', [True, False]),

        #model specific params
        'model__C': trial.suggest_float('model__C', 0.1, 10.0),
        'model__kernel': trial.suggest_categorical('model__kernel', ['linear', 'rbf', 'poly']),
        'model__gamma': trial.suggest_categorical('model__gamma', ['scale', 'auto'] + [10**-i for i in range(-4, 4)]),
        'model__class_weight': trial.suggest_categorical('model__class_weight', ['balanced', None]),
        'model__degree': trial.suggest_int('model__degree', 2, 5)  # If using 'poly' kernel
    }

    train_indices_to_remove = get_remove_indices_training_data(y_train_d, 
                                                               params['data_filter__outlier_y'], #remove outliers
                                                               label_issues_b, 
                                                               params['data_filter__label_issues_b']) #remove noisy labels
    X_train = X_train.drop(train_indices_to_remove).reset_index(drop=True)
    y_train = y_train.drop(train_indices_to_remove).reset_index(drop=True)

    
    pipeline = gen_full_pipeline(model = model, params = params) #create the full model pipeline
    scores = evaluate_model(pipeline, X_train, y_train)
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_val)

    #ps = precision_score(y_v, y_pred)
    #rs = recall_score(y_v, y_pred)
    f1_val = f1_score(y_val, y_pred)

    mlflow.log_params(params)
    
    mlflow.log_metrics({"precision_cv": scores["test_precision"]})
    mlflow.log_metrics({"recall_cv": scores["test_recall"]})
    mlflow.log_metrics({"f1_cv": scores["test_f1"]})
    mlflow.log_metrics({"f1_val": f1_val})

    trial.set_user_attr('precision_cv', scores["test_precision"])
    trial.set_user_attr('recall_cv', scores["test_recall"])
    trial.set_user_attr('f1_cv', scores["test_f1"])
    trial.set_user_attr('f1_val', f1_val)

    mlflow.set_tag("mlflow.runName", trial.number)
    mlflow.end_run()
    return f1_val

# %%
#setup dashboard using the below in bash terminal
#$mlflow server --backend-store-uri sqlite:///mlflow.db

# %%
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
#optuna.logging.disable_default_handler() #

# %%
#mlflow.set_experiment(experiment_name="xgbc0")
#xgb_class_study, xgb_class_pipeline = run_study(X_train, y_train_b, X_val, y_val_b, xgb_class_objective, model_options['xgb'])

# %%
#print(xgb_class_study.best_value)
#optuna.visualization.plot_optimization_history(xgb_class_study)

# %%
X_train.shape
X_val.shape
y_train_b.unique()

# %%
#mlflow.set_experiment(experiment_name="rfc15")
rf_class_study, rf_class_pipeline = run_study(X_train, y_train_b.astype(int), X_val, y_val_b.astype(int), rf_class_objective, model_options['rf'])
print(rf_class_study.best_value)

# %%
print(rf_class_study.best_value)
optuna.visualization.plot_optimization_history(rf_class_study)

# %%
mlflow.set_experiment(experiment_name="svcc1")
svc_class_study, svc_class_pipeline = run_study(X_train, y_train_b, X_val, y_val_b, svc_class_objective, model_options['svc'])

# %%
#print(svc_class_study.best_value)
#optuna.visualization.plot_optimization_history(svc_class_study)

# %%
#optuna.visualization.plot_param_importances(xgb_class_study)

# %%
#save the best model
#save the best model
best_class_pipeline = xgb_class_pipeline
joblib.dump(best_class_pipeline, "models/pipeline_classifier.pkl", compress=1)

# %% [markdown]
# # Continous modeling - Stage 2: Having predicted this group will donate, how much will they donate now?

# %%
import numpy as np

# %% [markdown]
# we need to adjust our training dataset and evaluation dataset rows now to only be those predicted from step 1 to have donated. Then we train to get close to actual donation amount, add it up, and see how far we are from the true $ amount received from that group. Also in context of how many revenue we didn't capture due to step 1 missing out.
# 
# We are not going to use the training data to train this one as it doesn't give us real world simulation here. We could eventually do this if we can add some additional checks to monitor performance and that end performance of the whole system isn't being effected by compounding any overfitting of the classification model. 

# %%
def gen_cleaning_pipeline():
    "the basic data cleaning pipeline from earlier we used"
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ])
    numerical_missing_onehot_transformer = Pipeline(steps=[
        ('missing_indicator', MissingIndicator()),
        
    ])

    numerical_transformer = Pipeline(steps=[
        #('missing_indicator', MissingIndicator()),
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', MinMaxScaler()),
        
    ])

    preprocessing_pipeline = ColumnTransformer(transformers=
            [('num_onehot_missing', numerical_missing_onehot_transformer, num_cols),
            ('num', numerical_transformer, num_cols), #selector(dtype_exclude="numeric")
            ('cat', categorical_transformer, cat_cols)], #selector(dtype_include="object")
        remainder='drop')
    #preprocessing_pipeline = #Pipeline(steps=[('preprocessor', column_trans)])
    return preprocessing_pipeline

def gen_feature_selection_pipeline_continuous():
    """uses Boruta to run feature selection in the pipeline for us"""
    fs_model = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42) #feature selection model
    boruta_features = BorutaPy(
        verbose=0,
        estimator=fs_model,
        n_estimators='auto',
        max_iter=10,  
        random_state=42,
    )
    return boruta_features

def gen_iteraction_features_pipeline():
    """will iteract all features together to create additional information for the model"""
    interaction_generator = PolynomialFeatures(degree = 1, \
                interaction_only=False, include_bias=False)
    return interaction_generator

def gen_full_pipeline(model, params = {}):
    """pulls together all the pipeline pieces to reduce data leakage"""
    data_cleaner = gen_cleaning_pipeline()
    interaction_generator = gen_iteraction_features_pipeline()
    boruta_features = gen_feature_selection_pipeline_continuous()
    
    pipeline = Pipeline([
        ('data_cleaner', data_cleaner),
        ('interaction_gen', interaction_generator),
        #('feature_selection', boruta_features), 
        ('model', model)
    ])

    pipeline_best_params = {key:val for key,val in params.items() if key in pipeline.get_params()} #filter to make sure best params are in pipeline

    pipeline.set_params(**pipeline_best_params)

    return pipeline




# %%
#having the pipeline fit on the train data...
valid_selector_index = xgb_class_pipeline.predict(X_val) #give me from the validation df the index that would be predicted to make a donation
X_val_s1_pred = X_val[valid_selector_index.astype(np.bool)].reset_index(drop=True) #new validation data filtered from stage 1 predictions
y_val_s1_pred_d = y_val_d[valid_selector_index.astype(np.bool)].reset_index(drop=True)

test_selector_index = xgb_class_pipeline.predict(X_test) #give me from the test df the index that would be predicted to make a donation
X_test_s1_pred = X_test[test_selector_index.astype(np.bool)].reset_index(drop=True) #new test data filtered from stage 1 predictions
y_test_s1_pred_d = y_test_d[test_selector_index.astype(np.bool)].reset_index(drop=True)

# %%
model_options_continuous = {'xgb_c': xgb.XGBRegressor(tree_method='approx'),
                            'rf_c': RandomForestRegressor()}
#Additional testing with more models would be a great option to add here

def my_custom_scorer_continuous(estimator, X, y):
    y_pred = estimator.predict(X)
    mae = mean_absolute_error(y, y_pred)
    mape = mean_absolute_percentage_error(y, y_pred)
    return {'mape': mape, 'mae': mae}

def evaluate_model_continuous(model, X, y):
    """does the cross validation evaluation approach with KFold"""
    cv = RepeatedStratifiedKFold(n_splits=3, 
                                 n_repeats=2, 
                                 random_state=1)
    scores = cross_validate(model, X, y, 
                             scoring=my_custom_scorer_continuous, 
                             cv=cv, n_jobs=-1)
    averaged_scores = {"test_mape" : scores["test_mape"].mean(),
                       "test_mae" : scores["test_mae"].mean()}
    return averaged_scores

def run_study_cont(X_train, y_train, X_val, y_val, objective, model):
    pipeline = None
    
    study = optuna.create_study(**{
        'direction': 'minimize',
        'sampler': optuna.samplers.TPESampler(seed=37),
        'pruner': optuna.pruners.MedianPruner(n_warmup_steps=10)
    })


    #with mlflow.start_run():
        
    study.optimize(**{
        'func': lambda trial: objective(trial, X_train, y_train, X_val, y_val, model), #objective, 
        'n_trials': 5,
        'n_jobs': -1, #TODO: change back to -1 when not using mlflow
        'show_progress_bar': True,
        #'callbacks' : [mlflow_callback]
    })

    #below replaced with gen_full_pipeline param check
    #pipeline_params = gen_full_pipeline(model).get_params() #grab what the actual params are in our pipeline
    #pipeline_best_params = {key:val for key,val in study.best_params.items() if key in pipeline_params} #filter all best_params
    #create new copy of the pipeline for outside optuna
    pipeline = gen_full_pipeline(model, study.best_params)
    
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_val)
    with open("reports/results_report.txt", "a+") as results_report:
        results_report.write(str(model.__class__)+ '\n')
        results_report.write("Best Score:" + str(study.best_value)+ '\n')
        results_report.write("Model Setup:" + str(study.best_params) + '\n\n')


    return study, pipeline

# %%
def xgb_contin_objective(trial, X_train, y_train, X_val, y_val, model):
    #model = model_options_continuous['xgb_c']
    
    #turning this off since we didn't have time to do all the same data outlier checks for the continuous case
    """non_pipeline_params = {
        #data filtering
        'data_filter__label_issues_b': trial.suggest_float('data_filter__label_issues_b', 0.0, 0.1),
        'data_filter__outlier_y': trial.suggest_int('data_filter__outlier_y', 1500, 2500),
    }"""
    params = {

        #global pipeline params
        'interaction_gen__interaction_only': trial.suggest_categorical('interaction_gen__interaction_only', [True, False]),

        #model specific params
        "model__max_depth": trial.suggest_int("max_depth", 4, 8),
        "model__min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "model__gamma": trial.suggest_float("gamma", 0.0, 5),
        "model__subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "model__colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "model__learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "model__lambda": trial.suggest_float("lambda", 0.8, 1.0),
        "model__alpha": trial.suggest_float("alpha", 1e-8, 10.0),
        "model__n_estimators": trial.suggest_int("n_estimators", 50, 120),
    }
    
    pipeline = gen_full_pipeline(model = model, params = params)
    #scores = evaluate_model_continuous(pipeline, X_train, y_train)
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_val)

    #ps = precision_score(y_v, y_pred)
    #rs = recall_score(y_v, y_pred)
    mape_val = mean_absolute_percentage_error(y_val, y_pred) #does the held out approach
    mae_val = mean_absolute_error(y_val, y_pred)

    mlflow.log_params(params)
    #mlflow.log_metrics({"mape_cv": scores["test_mape"]})
    #mlflow.log_metrics({"mae_cv": scores["test_mae"]})

    mlflow.log_metrics({"mape_val": mape_val})

    #trial.set_user_attr('mape_cv', scores["test_precision"])
    #trial.set_user_attr('mae_cv', scores["test_recall"])

    trial.set_user_attr('mape_val', mape_val)
    trial.set_user_attr('mae_val', mae_val)

    mlflow.set_tag("mlflow.runName", trial.number)
    mlflow.end_run()

    return mae_val

def rf_contin_objective(trial, X_train, y_train, X_val, y_val, model):
    #model = model_options_continuous['xgb_c']
    
    #turning this off since we didn't have time to do all the same data outlier checks for the continuous case
    """non_pipeline_params = {
        #data filtering
        'data_filter__label_issues_b': trial.suggest_float('data_filter__label_issues_b', 0.0, 0.1),
        'data_filter__outlier_y': trial.suggest_int('data_filter__outlier_y', 1500, 2500),
    }"""
    params = {

        #global pipeline params
        'interaction_gen__interaction_only': trial.suggest_categorical('interaction_gen__interaction_only', [True, False]),

        #model specific params
        "model__n_estimators": trial.suggest_int("n_estimators", 50, 200),
        "model__max_depth": trial.suggest_int("max_depth", 4, 12),
        "model__min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "model__max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
    }
    
    pipeline = gen_full_pipeline(model = model, params = params)
    #scores = evaluate_model_continuous(pipeline, X_train, y_train) #does CV approach
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_val)

    mape_val = mean_absolute_percentage_error(y_val, y_pred) #held out evaluation approach
    mae_val = mean_absolute_error(y_val, y_pred)

    mlflow.log_params(params)
    #mlflow.log_metrics({"mape_cv": scores["test_mape"]})
    #mlflow.log_metrics({"mae_cv": scores["test_mae"]})

    mlflow.log_metrics({"mape_val": mape_val})

    #trial.set_user_attr('mape_cv', scores["test_precision"])
    #trial.set_user_attr('mae_cv', scores["test_recall"])

    trial.set_user_attr('mape_val', mape_val)
    trial.set_user_attr('mae_val', mae_val)

    mlflow.set_tag("mlflow.runName", trial.number)
    mlflow.end_run()

    return mae_val

# %%
xgb_cont_study, xgb_cont_pipeline = run_study_cont(X_val_s1_pred, y_val_s1_pred_d, X_test_s1_pred, y_test_s1_pred_d, xgb_contin_objective, model_options_continuous['xgb_c'])


# %%
print(xgb_cont_study.best_value)
optuna.visualization.plot_optimization_history(xgb_cont_study)

# %%
X_val_s1_pred.shape

# %%
mlflow.end_run()

# %%
mlflow.set_experiment(experiment_name="fr_cont_1")
rf_cont_study, rf_cont_pipeline = run_study_cont(X_val_s1_pred, y_val_s1_pred_d, X_test_s1_pred, y_test_s1_pred_d, rf_contin_objective, model_options_continuous['rf_c'])

# %%
print(rf_cont_study.best_value)


# %%
optuna.visualization.plot_optimization_history(rf_cont_study)

# %%
#Evaluate the Models

# %%
#save the best model
best_cont_pipeline = rf_cont_pipeline
joblib.dump(best_cont_pipeline, "models/pipeline_continuous.pkl", compress=1)

# %% [markdown]
# # Calibration

# %%
def measure_coverage(cp_pipeline, X_test, y_test, alpha=0.1):  # Assuming alpha = 0.1 for 90% confidence
    intervals = cp_pipeline.predict(X_test, alpha= alpha)

    # Check if true values are within the prediction intervals
    coverage = np.mean([y_test[i] >= intervals[i][0] and y_test[i] <= intervals[i][1] for i in range(len(y_test))])

    return coverage

def run_study_cp(X_train, y_train, X_val, y_val, objective, model):
    pipeline = None
    
    study = optuna.create_study(**{
        'direction': 'maximize',
        'sampler': optuna.samplers.TPESampler(seed=37),
        'pruner': optuna.pruners.MedianPruner(n_warmup_steps=10)
    })


    #with mlflow.start_run():
        
    study.optimize(**{
        'func': lambda trial: objective(trial, X_train, y_train, X_val, y_val), #objective, 
        'n_trials': 5,
        'n_jobs': -1, #TODO: change back to -1 when not using mlflow
        'show_progress_bar': True,
        #'callbacks' : [mlflow_callback]
    })

    #below replaced with gen_full_pipeline param check
    #pipeline_params = gen_full_pipeline(model).get_params() #grab what the actual params are in our pipeline
    #pipeline_best_params = {key:val for key,val in study.best_params.items() if key in pipeline_params} #filter all best_params
    #create new copy of the pipeline for outside optuna
    pipeline = gen_full_pipeline(model, study.best_params)
    
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_val)
    with open("reports/results_report.txt", "a+") as results_report:
        results_report.write(str(model.__class__)+ '\n')
        results_report.write("Best Score:" + str(study.best_value)+ '\n')
        results_report.write("Model Setup:" + str(study.best_params) + '\n\n')


    return study, pipeline

# %%
#grab best params for the model to set it up now, it becomes harder once nested
just_best_cont_model = best_cont_pipeline.get_params()['model']

model_options_conformal = {'cp': IcpRegressor(RegressorNc(RegressorAdapter(just_best_cont_model), AbsErrorErrFunc()))}

# %%
def objective_cp(trial, X_train, y_train, X_val, y_val):
    # Use best_params_xgb from XGBoost optimization
    model = model_options_conformal['cp']
    
    model.set_sigma(trial.suggest_float("sigma", 0.1, 1.0))
    
    #train test split to make calibrate model
    X_train, X_cal, y_train, y_cal = train_test_split(
    X_train, y_train, test_size=0.05, random_state=42, stratify=target_b
    )
    pipeline = gen_full_pipeline(model = model, params = best_cont_pipeline.get_params())
    pipeline.fit(X_train, y_train)
    pipeline.named_steps['model'].calibrate(X_cal, y_cal)

    predictions = pipeline.predict(X_val, significance=0.10)


    # Evaluate coverage probability
    coverage = measure_coverage(pipeline, X_val, y_val, alpha=0.1)
    return coverage

# %%
cp_study, cp_pipeline = run_study_cont(X_val_s1_pred, y_val_s1_pred_d, X_test_s1_pred, y_test_s1_pred_d, xgb_contin_objective, model_options_conformal['cp'])

# %%
cp_study.best_value

# %%
optuna.visualization.plot_optimization_history(cp_study)

# %%


# %% [markdown]
# # Wrapup
# 
# I would by no means consider this a final solution yet. There is a lot more validation of results to be performed from equity and robustness standpoints. Some of the first set of next tests I would run would be seeing how robust the classification and continuous models are to vary degrees of noise added to the inputs, if there are particular clusters of individuals we are doing a poor job of identifying and catering to, and assessing overall calibration levels of the model across different probabilities. Additionally, I would want to break down the data into very low percentile and high high percentile outcomes for dollars donated and see how both of our models do for those groups compared to say the middle 90%. More research can be done to look into high leverage points as well. 
# 
# A couple of business pieces of information that may be important involve marginal costs for each reach out to effectively move break even costs to a higher donation amount. Additionally, looking into lagged effects to expand our focus beyond the "6 month horizon" we chose as targets can lead to great accuracy for the models overall. I believe that some features can be built to represent momentum for particular individuals can help identify better the groups tagged as noisy labels by cleanlab simply because we had a hard time identifying them at that point. 
# 
# Overall, I think model performance is a great start. There is some tweaking to be done to the optimization to find some better global optimum points. I believe the test group gave us a fairly accurate assesment of this model performance conditional on no major data drift occurences. 

# %% [markdown]
# Calibration: It assesses how well a model's predicted probabilities align with the actual frequencies of outcomes.
# Visualizing Calibration:
# Reliability Diagram: Plot predicted probabilities against actual observed frequencies for various probability bins. A perfectly calibrated model would create a diagonal line.
# Calibration Plot: Plot predicted probabilities against observed outcomes, grouping predictions into bins.
# Quantitative Measures:
# Brier Score: Average squared difference between predicted probabilities and actual outcomes (0-1, lower is better).
# Expected Calibration Error (ECE): Average absolute difference between predicted probabilities and actual frequencies within bins.

# %% [markdown]
# 


