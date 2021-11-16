from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score
import pandas as pd
from imblearn.over_sampling import SMOTE

import sys, os,  pickle
from datetime import datetime

import warnings
warnings.filterwarnings("ignore")

data_path = './dataset/'
global_model_path = './global_model/'

if not os.path.exists(global_model_path):
    os.makedirs(global_model_path)

def load_change_metrics_df(cur_proj):
    if cur_proj == 'qt':
        start = 1308350292
        end = 1395090476
    elif cur_proj == 'openstack':
        start = 1322599384
        end = 1393590700
    change_metrics = pd.read_csv(data_path+cur_proj+'_metrics.csv')
    
    change_metrics = change_metrics[(change_metrics['author_date'] >= start) & 
                                    (change_metrics['author_date'] <= end)]
    
    change_metrics['self'] = [1 if s is True else 0 for s in change_metrics['self']]
    change_metrics['defect'] = change_metrics['bugcount'] > 0
    change_metrics['new_date'] = change_metrics['author_date'].apply(lambda x: datetime.fromtimestamp(x))
    
    change_metrics = change_metrics.sort_values(by='new_date')
    change_metrics['new_date'] = change_metrics['new_date'].apply(lambda x: x.strftime('%m-%d-%Y'))
    
    change_metrics['rtime'] = (change_metrics['rtime']/3600)/24
    change_metrics['age'] = (change_metrics['age']/3600)/24
    change_metrics = change_metrics.reset_index()
    change_metrics = change_metrics.set_index('commit_id')
    
    bug_label = change_metrics['defect']
    
    change_metrics = change_metrics.drop(['author_date', 'new_date', 'bugcount','fixcount','revd','tcmt','oexp','orexp','osexp','osawr','defect']
                                         ,axis=1)
    change_metrics = change_metrics.fillna(value=0)
    
    
    return change_metrics, bug_label

def split_train_test_data(feature_df, label, percent_split = 70):
    _p_percent_len = int(len(feature_df)*(percent_split/100))
    x_train = feature_df.iloc[:_p_percent_len]
    y_train = label.iloc[:_p_percent_len]
    
    x_test = feature_df.iloc[_p_percent_len:]
    y_test = label.iloc[_p_percent_len:]
    
    return x_train, x_test, y_train, y_test

def prepare_data(proj_name, mode = 'all'):
    if mode not in ['train','test','all']:
        print('this function accepts "train","test","all" mode only')
        return
    
    change_metrics, bug_label = load_change_metrics_df(proj_name) 
    
    with open(data_path+proj_name+'_non_correlated_metrics.txt','r') as f:
        metrics = f.read()
    
    metrics_list = metrics.split('\n')
    
    non_correlated_change_metrics = change_metrics[metrics_list]

    x_train, x_test, y_train, y_test = split_train_test_data(non_correlated_change_metrics, bug_label, percent_split = 70)
    
    if mode == 'train':
        return x_train,y_train
    elif mode == 'test':
        return x_test, y_test
    elif mode == 'all':
        return x_train, x_test, y_train, y_test

def train_global_model(proj_name, x_train,y_train, global_model_name = 'RF'):
    global_model_name = global_model_name.upper()
    if global_model_name not in ['RF','LR']:
        print('wrong global model name. the global model name must be RF or LR')
        return
    
    smt = SMOTE(k_neighbors=5, random_state=42, n_jobs=24)
    new_x_train, new_y_train = smt.fit_resample(x_train, y_train)
    
    if global_model_name == 'RF':
        global_model = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)
    elif global_model_name == 'LR':
        global_model = LogisticRegression(random_state=0, n_jobs=-1)
        
    global_model.fit(new_x_train, new_y_train)
    pickle.dump(global_model, open(os.path.join(global_model_path,proj_name+'_'+global_model_name+'_global_model.pkl'),'wb'))

def eval_global_model(proj_name, x_test,y_test, global_model_name = 'RF'):
    global_model_name = global_model_name.upper()
    if global_model_name not in ['RF','LR']:
        print('wrong global model name. the global model name must be RF or LR')
        return
    global_model = pickle.load(open(os.path.join(global_model_path,proj_name+'_'+global_model_name+'_global_model.pkl'),'rb'))

    pred = global_model.predict(x_test)
    prob = global_model.predict_proba(x_test)[:,1]

    auc = roc_auc_score(y_test, prob)
    f1 = f1_score(y_test, pred)

    print('AUC: {}, F1: {}'.format(auc,f1))


proj_name = sys.argv[1]
global_model_name = sys.argv[2]

x_train, x_test, y_train, y_test = prepare_data(proj_name, mode = 'all')


train_global_model(proj_name, x_train, y_train,global_model_name)
eval_global_model(proj_name, x_test,y_test, global_model_name)
