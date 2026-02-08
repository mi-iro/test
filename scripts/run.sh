cd test

python run_retrieval.py --config configs/config_mmlong_no_judge.yaml

python run_generation.py --config configs/config_finrag_en.yaml --num_threads 16

python run_evaluation.py --config configs/config_mmlong_no_judge.yaml
