import argparse
import pandas as pd
from sklearn import metrics
from tqdm import tqdm


def evaluate_results(log_file):
    # read the logits file
    pred = pd.read_csv(log_file)
    
    
    pred['logits'] = pred['logits'].str.replace(r'\[\s*', '', regex=True)
    pred['logits'] = pred['logits'].str.replace(r'\s*\]', '', regex=True)
    s = pred['logits'].str.split('\s+', expand=True)
    pred = pd.concat([pred, s.rename(columns = {0:'pr_control', 1:'pr_stemi', 2:'pr_nstemi'})], axis=1)
    pred['prob_mi'] = pred['pr_stemi'].astype('float') + pred['pr_nstemi'].astype('float')

    # C-statistic / AUROC
    metrics.roc_auc_score(pred['labels'], pred['prob_mi'])
    # pr
    metrics.average_precision_score(pred['labels'], pred['prob_mi'])
    
    # print the results
    tqdm.write(f"Results: control vs MI (STEMI+NSTEMI)")
    tqdm.write(f"ROC AUC: {metrics.roc_auc_score(pred['labels'], pred['prob_mi']):.4f}")
    tqdm.write(f"PR AUC: {metrics.average_precision_score(pred['labels'], pred['prob_mi']):.4f}")

# main
if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--log_file", type=str, default="logs/logits.csv", help="path to logs")
    config, _ = parser.parse_known_args()
    
    evaluate_results(config.log_file)