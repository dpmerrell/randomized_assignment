# randomized_assignment

A small collection of potentially useful randomized-assignment algorithms.

These are algorithms I've come up with in the course of doing research.

I think I might use them in the future, so I've put them here for safekeeping.
For now it's just a collection of python files&mdash;no packaging or anything.

# Contents

* `stratified_grouped_split.py`.
  Generate stratified, grouped, shuffled splits of a dataset.
  This is useful for machine learning model validation.
  Scikit-learn has functions for generating *stratified* splits,
  *grouped* splits, and *shuffled* splits, but I'm not sure
  it has a function for doing _all_ of those things jointly.
* `degree_constrained_random_graph.py`. 
  Generate random graphs whose nodes have specified degrees. 
  This is useful for some network analyses where we need to compare
  against a "null distribution" of random graphs.
