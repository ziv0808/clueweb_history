#!/bin/bash

nohup python wieghted_ranking_model.py $1 $2 $3 $4 $5 1 20&
nohup python wieghted_ranking_model.py $1 $2 $3 $4 $5 21 40&
nohup python wieghted_ranking_model.py $1 $2 $3 $4 $5 41 60&
nohup python wieghted_ranking_model.py $1 $2 $3 $4 $5 61 80&
nohup python wieghted_ranking_model.py $1 $2 $3 $4 $5 81 100&
nohup python wieghted_ranking_model.py $1 $2 $3 $4 $5 101 120&
nohup python wieghted_ranking_model.py $1 $2 $3 $4 $5 121 140&
nohup python wieghted_ranking_model.py $1 $2 $3 $4 $5 141 160&
nohup python wieghted_ranking_model.py $1 $2 $3 $4 $5 161 180&
nohup python wieghted_ranking_model.py $1 $2 $3 $4 $5 181 200&