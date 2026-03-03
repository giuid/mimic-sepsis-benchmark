#!/bin/bash
PROJECT_ROOT="/home/guido/Code/charite/baselines"
cd $PROJECT_ROOT
cp artifacts_pruned/pruned_graph.pkl graph/artifacts_pruned/pruned_graph.pkl
cp artifacts_pruned/subgraph_nodes.csv graph/artifacts_pruned/subgraph_nodes.csv
cd graph
python3 relational_prior.py
