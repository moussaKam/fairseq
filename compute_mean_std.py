import os
import numpy as np
from tensorboard.backend.event_processing import event_accumulator
import argparse

parse = argparse.ArgumentParser()
parse.add_argument('--path_events', type=str, help='path to the tensorboard events')
args = parse.parse_args()

path_events = args.path_events
assert os.path.isdir(path_events)

directories = [os.path.join(path_events, el) for el in os.listdir(path_events)]

best_test = []

for directory in directories:
    valid_scores = []
    test_scores = []
    valid_events = os.path.join(directory,'valid')
    test_events = os.path.join(directory,'test')
    ea = event_accumulator.EventAccumulator(valid_events, size_guidance={event_accumulator.SCALARS: 0})
    ea.Reload()
    for el in ea.Scalars('accuracy'):
        valid_scores.append(el.value)
    ea = event_accumulator.EventAccumulator(test_events, size_guidance={event_accumulator.SCALARS: 0})
    ea.Reload()
    for el in ea.Scalars('accuracy'):
        test_scores.append(el.value)
    maxi = 0 
    max_test = []
    for i, el in enumerate(valid_scores):
        if el >= maxi:
            maxi = el
            max_test = test_scores[i]
    best_test.append(max_test)
    
print('mean: {}'.format(np.mean(best_test)), 'std: {}'.format(np.std(best_test)))
