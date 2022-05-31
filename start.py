
import src
input_path = 'src/data/alice.csv'
output_path = 'src/data/data_new/train.csv'

clean_train_input_path = 'src/data/data_new/train.csv'
clean_train_output_path = 'src/data/data_new/train_text.txt'

X = 'src/data/data_new/X'

params_logit = src.load_yaml('src/params/logit.yaml')
params_xgb = src.load_yaml('src/params/best_xgb.yaml')
params_randomized = src.load_yaml('src/params/random_xgb.yaml')

output_score_log = 'src/data/score/roc_auc_log.csv'
output_score_xgb = 'src/data/score/roc_auc_xgb.csv'
output_param = 'src/params/best_xgb.yaml'
if __name__ == '__main__':
    src.get_sites(input_path,output_path)
    src.clean(clean_train_input_path, clean_train_output_path)
    X = src.vector(clean_train_output_path)
    train_data, val_data, train_labels, val_labels = src.split(X,input_path,0.3)
    train_data,train_labels = src.smote_train(train_data,train_labels)
    # src.logit(train_data,train_labels,val_data,val_labels,output_score_log,**params_logit)
    #src.search_params(train_data, train_labels, output_param, **params_randomized)
    src.xgb(train_data,train_labels,val_data,val_labels,output_score_xgb,**params_xgb)
