"""
stratified_grouped_split.py

I implemented the function `stratified_group_shuffle_split` 
for one of my projects. It seems potentially useful for 
other projects, so I've put it in this repo for safekeeping.
"""

from collections import Counter, defaultdict
import numpy as np


def stratified_group_shuffle_split(class_labels, group_labels, split_fracs=[0.8,0.2]):
    """
    A randomized algorithm for generating shuffled splits that 
    (1) are stratified by `class_labels` and
    (2) are grouped by `group_labels`.

    The size/number of splits are defined by `split_fracs` (iterable)
    
    *Assume discrete class labels.*
    """
    split_fracs = np.array(split_fracs)
    
    # Get the unique classes and their occurrences
    cls_counts = Counter(class_labels)
    unq_classes = np.array(sorted(list(cls_counts.keys())))
    cls_encoder = {cls: idx for idx, cls in enumerate(unq_classes)}
    cls_counts = np.vectorize(lambda x: cls_counts[x])(unq_classes) 
    
    # Compute a grid of "capacities": ideal quantities 
    # of samples for each (group, class) pair.
    capacity = np.outer(split_fracs, cls_counts)
    
    # Get the unique groups, in random order.
    unq_groups = np.random.permutation(np.unique(group_labels))
    
    # Collect information about the groups' samples and labels
    gp_encoder = {gp: idx for idx, gp in enumerate(unq_groups)}
    gp_vecs = np.zeros((len(unq_groups), len(unq_classes)))
    gp_to_samples = defaultdict(lambda : [])
    for idx, (gp, cls) in enumerate(zip(group_labels, class_labels)):
        gp_vecs[gp_encoder[gp], cls_encoder[cls]] += 1
        gp_to_samples[gp].append(idx)
    
    # We will assign groups to these splits
    split_sets = [set() for _ in split_fracs]
    
    # Randomly assign groups to splits
    for i in range(gp_vecs.shape[0]):
        # Randomization is weighted by (a) the group's class distribution and
        # (b) available capacities in the splits.
        gp_counts = gp_vecs[i,:]
        weight_vec = np.dot(capacity, gp_counts)
        split_idx = np.random.choice(len(split_fracs), p=weight_vec/np.sum(weight_vec))
        
        # Add group to split; decrement capacity
        split_sets[split_idx].add(unq_groups[i])
        capacity[split_idx,:] -= gp_counts
        capacity[capacity < 0] = 0  # no negative capacities!
    
    # Convert output to a list of index arrays
    split_idxs = [np.array(sorted(sum((gp_to_samples[gp] for gp in s), [])), dtype=int) for s in split_sets]
 
    # Return splits
    return split_idxs

