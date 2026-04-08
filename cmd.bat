

python -m examples.graphwiz.graphwiz_eval_generic --source test --subset connectivity --data_root ./data --prefer_local 1 --budget 1000 --lm_name chatgpt --max_samples 1000 --use_cache 0
python -m examples.graphwiz.graphwiz_eval_generic --source test --subset cycle --data_root ./data --prefer_local 1 --budget 1000 --lm_name chatgpt --max_samples 1000 --use_cache 0
python -m examples.graphwiz.graphwiz_eval_generic --source test --subset shortest --data_root ./data --prefer_local 1 --budget 1000 --lm_name chatgpt --max_samples 1000 --use_cache 0
python -m examples.graphwiz.graphwiz_eval_generic --source test --subset triangle --data_root ./data --prefer_local 1 --budget 1000 --lm_name chatgpt --max_samples 1000 --use_cache 0
python -m examples.graphwiz.graphwiz_eval_generic --source test --subset substructure --data_root ./data --prefer_local 1 --budget 1000 --lm_name chatgpt --max_samples 1000 --use_cache 0
python -m examples.graphwiz.graphwiz_eval_generic --source test --subset topology --data_root ./data --prefer_local 1 --budget 1000 --lm_name chatgpt --max_samples 1000 --use_cache 0
python -m examples.graphwiz.graphwiz_eval_generic --source test --subset flow --data_root ./data --prefer_local 1 --budget 1000 --lm_name chatgpt --max_samples 1000 --use_cache 0
python -m examples.graphwiz.graphwiz_eval_generic --source test --subset bipartite --data_root ./data --prefer_local 1 --budget 1000 --lm_name chatgpt --max_samples 1000 --use_cache 0
python -m examples.graphwiz.graphwiz_eval_generic --source test --subset hamilton --data_root ./data --prefer_local 1 --budget 1000 --lm_name chatgpt --max_samples 1000 --use_cache 0
