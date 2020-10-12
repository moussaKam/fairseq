import os
import numpy as np
import argparse

parse = argparse.ArgumentParser()
parse.add_argument('--path_models', type=str)
args = parse.parse_args()

rouge1 = []
rouge2 = []
rougeL = []

directories = os.listdir(args.path_models)

for directory in directories:
    path_scores = os.path.join(args.path_models, directory, 'score.csv')
    with open(path_scores) as f:
        for line in f:
            if 'rouge1-F' in line:
                rouge1.append(float(line.split(',')[2]))
            elif 'rouge2-F' in line:
                rouge2.append(float(line.split(',')[2]))
            elif 'rougeL-F' in line:
                rougeL.append(float(line.split(',')[2]))

print('Rouge1:\n', 'mean: {:0.4f}\t'.format(np.mean(rouge1)), 'median: {:0.4f}\t'.format(np.median(rouge1)), 'std: {:0.4f}\t'.format(np.std(rouge1)))
print('\nRouge2:\n', 'mean: {:0.4f}\t'.format(np.mean(rouge2)), 'median: {:0.4f}\t'.format(np.median(rouge2)), 'std: {:0.4f}\t'.format(np.std(rouge2)))
print('\nRougeL:\n', 'mean: {:0.4f}\t'.format(np.mean(rougeL)), 'median: {:0.4f}\t'.format(np.median(rougeL)), 'std: {:0.4f}\t'.format(np.std(rougeL)))
