#nohup python main.py  --gpu_id 2  --data_name=Beauty --latent_clr_weight=0.6 --reparam_dropout_rate=0.1 --lr=0.001 --hidden_size=128 --max_seq_length=50 --hidden_dropout_prob=0.3 --num_hidden_layers=1 --weight_decay=0.0 --num_attention_heads=4 --model_name=MultiVAERec --attention_probs_dropout_prob=0.0 --anneal_cap=0.2 --total_annealing_step=10000 > log_beauty_0726.log 2>&1 &
#
#nohup python main.py --gpu_id 3 --data_name=Tools_and_Home_Improvement --latent_clr_weight=0.4 --lr=0.001 --hidden_size=128 --max_seq_length=100 --hidden_dropout_prob=0.3 --num_hidden_layers=1 --weight_decay=0.0 --num_attention_heads=4 --model_name=VAGRecVD --attention_probs_dropout_prob=0.3 --anneal_cap=0.4 --total_annealing_step=5000 > log_tool_0726.log 2>&1 &
#
#nohup python main.py --gpu_id 2 --data_name=Toys_and_Games --latent_clr_weight=0.3 --lr=0.001 --hidden_size=128 --max_seq_length=100 --hidden_dropout_prob=0.3 --num_hidden_layers=1 --weight_decay=0.0 --num_attention_heads=4 --model_name=VAGRecVD --attention_probs_dropout_prob=0.3 --anneal_cap=0.2 --total_annealing_step=10000 > log_toy_0726.log 2>&1 &

#nohup python main.py --gpu_id 4 --data_name=Office_Products --latent_clr_weight=0.3 --lr=0.001 --hidden_size=128 --max_seq_length=100 --hidden_dropout_prob=0.3 --num_hidden_layers=1 --weight_decay=0.0 --num_attention_heads=4 --model_name=VAGRec --attention_probs_dropout_prob=0.3 --anneal_cap=0.2 --total_annealing_step=20000 > log_office_0726 2>&1 &
#
#nohup python main.py --gpu_id 1 --interest_celoss_coeff 0.01 --gm_kl_coeff 0.00001 --data_name=Office_Products --latent_clr_weight=0.3 --lr=0.001 --hidden_size=128 --max_seq_length=100 --hidden_dropout_prob=0.3 --num_hidden_layers=1 --weight_decay=0.0 --num_attention_heads=4 --model_name=VAGRec --attention_probs_dropout_prob=0.3 --anneal_cap=0.2 --total_annealing_step=20000 > log_office_0726_0.01_0.00001 2>&1 &
#nohup python main.py --gpu_id 4 --interest_celoss_coeff 0.001 --gm_kl_coeff 0.0001 --data_name=Office_Products --latent_clr_weight=0.3 --lr=0.001 --hidden_size=128 --max_seq_length=100 --hidden_dropout_prob=0.3 --num_hidden_layers=1 --weight_decay=0.0 --num_attention_heads=4 --model_name=VAGRec --attention_probs_dropout_prob=0.3 --anneal_cap=0.2 --total_annealing_step=20000 > log_office_0726_0.001_0.0001 2>&1 &
#nohup python main.py --gpu_id 4 --interest_celoss_coeff 0.01 --gm_kl_coeff 0.0001 --data_name=Office_Products --latent_clr_weight=0.3 --lr=0.001 --hidden_size=128 --max_seq_length=100 --hidden_dropout_prob=0.3 --num_hidden_layers=1 --weight_decay=0.0 --num_attention_heads=4 --model_name=VAGRec --attention_probs_dropout_prob=0.3 --anneal_cap=0.2 --total_annealing_step=20000 > log_office_0726_0.01_0.0001 2>&1 &
#nohup python main.py --gpu_id 4 --interest_celoss_coeff 0.001 --gm_kl_coeff 0.001 --data_name=Toys_and_Games --latent_clr_weight=0.3 --lr=0.001 --hidden_size=128 --max_seq_length=100 --hidden_dropout_prob=0.3 --num_hidden_layers=1 --weight_decay=0.0 --num_attention_heads=4 --model_name=VAGRecVD --attention_probs_dropout_prob=0.3 --anneal_cap=0.2 --total_annealing_step=10000 > log_toy_0726_0.001_0.0001.log 2>&1 &
#nohup python main.py --gpu_id 4 --interest_celoss_coeff 0.001 --gm_kl_coeff 0.0001 --data_name=Toys_and_Games --latent_clr_weight=0.3 --lr=0.001 --hidden_size=128 --max_seq_length=100 --hidden_dropout_prob=0.3 --num_hidden_layers=1 --weight_decay=0.0 --num_attention_heads=4 --model_name=VAGRecVD --attention_probs_dropout_prob=0.3 --anneal_cap=0.2 --total_annealing_step=10000 > log_toy_0726_0.001_0.001.log 2>&1 &
#nohup python main.py --gpu_id 1 --interest_celoss_coeff 0.001 --gm_kl_coeff 0.0001  --data_name=Tools_and_Home_Improvement --latent_clr_weight=0.4 --lr=0.001 --hidden_size=128 --max_seq_length=100 --hidden_dropout_prob=0.3 --num_hidden_layers=1 --weight_decay=0.0 --num_attention_heads=4 --model_name=VAGRecVD --attention_probs_dropout_prob=0.3 --anneal_cap=0.4 --total_annealing_step=5000 > log_tool_0726_0.001_0.0001.log 2>&1 &
#nohup python main.py --gpu_id 1 --interest_celoss_coeff 0.001 --gm_kl_coeff 0.001  --data_name=Tools_and_Home_Improvement --latent_clr_weight=0.4 --lr=0.001 --hidden_size=128 --max_seq_length=100 --hidden_dropout_prob=0.3 --num_hidden_layers=1 --weight_decay=0.0 --num_attention_heads=4 --model_name=VAGRecVD --attention_probs_dropout_prob=0.3 --anneal_cap=0.4 --total_annealing_step=5000 > log_tool_0726_0.001_0.001.log 2>&1 &

#nohup python main.py --gpu_id 1 --n_interest 2  --gm_kl_coeff 0.00001 --data_name=Office_Products --latent_clr_weight=0.3 --lr=0.001 --hidden_size=128 --max_seq_length=100 --hidden_dropout_prob=0.3 --num_hidden_layers=1 --weight_decay=0.0 --num_attention_heads=4 --attention_probs_dropout_prob=0.3 --anneal_cap=0.2 --total_annealing_step=20000 > log.log 2>&1 &
#nohup python main.py --gpu_id 2 --n_interest 4  --gm_kl_coeff 0.00001 --data_name=Office_Products --latent_clr_weight=0.3 --lr=0.001 --hidden_size=128 --max_seq_length=100 --hidden_dropout_prob=0.3 --num_hidden_layers=1 --weight_decay=0.0 --num_attention_heads=4 --attention_probs_dropout_prob=0.3 --anneal_cap=0.2 --total_annealing_step=20000 > log.log 2>&1 &
#nohup python main.py --gpu_id 3 --n_interest 8  --gm_kl_coeff 0.00001 --data_name=Office_Products --latent_clr_weight=0.3 --lr=0.001 --hidden_size=128 --max_seq_length=100 --hidden_dropout_prob=0.3 --num_hidden_layers=1 --weight_decay=0.0 --num_attention_heads=4 --attention_probs_dropout_prob=0.3 --anneal_cap=0.2 --total_annealing_step=20000 > log.log 2>&1

#nohup python main.py --gpu_id 1 --n_interest 2  --gm_kl_coeff 0.00001 --data_name=Office_Products --latent_clr_weight=0.3 --lr=0.001 --hidden_size=128 --max_seq_length=100 --hidden_dropout_prob=0.3 --num_hidden_layers=1 --weight_decay=0.0 --num_attention_heads=4 --attention_probs_dropout_prob=0.3 --anneal_cap=0.3 --total_annealing_step=10000 > log.log 2>&1 &
#nohup python main.py --gpu_id 2 --n_interest 4  --gm_kl_coeff 0.00001 --data_name=Office_Products --latent_clr_weight=0.3 --lr=0.001 --hidden_size=128 --max_seq_length=100 --hidden_dropout_prob=0.3 --num_hidden_layers=1 --weight_decay=0.0 --num_attention_heads=4 --attention_probs_dropout_prob=0.3 --anneal_cap=0.3 --total_annealing_step=10000 > log.log 2>&1 &
#nohup python main.py --gpu_id 3 --n_interest 8  --gm_kl_coeff 0.00001 --data_name=Office_Products --latent_clr_weight=0.3 --lr=0.001 --hidden_size=128 --max_seq_length=100 --hidden_dropout_prob=0.3 --num_hidden_layers=1 --weight_decay=0.0 --num_attention_heads=4 --attention_probs_dropout_prob=0.3 --anneal_cap=0.3 --total_annealing_step=10000 > log.log 2>&1 &

#nohup python main.py --gpu_id 1 --n_interest 2  --gm_kl_coeff 0.0001 --data_name=Office_Products --latent_clr_weight=0.3 --lr=0.001 --hidden_size=128 --max_seq_length=100 --hidden_dropout_prob=0.3 --num_hidden_layers=1 --weight_decay=0.0 --num_attention_heads=4 --attention_probs_dropout_prob=0.3 --anneal_cap=0.3 --total_annealing_step=10000 > log.log 2>&1 &
#nohup python main.py --gpu_id 2 --n_interest 4  --gm_kl_coeff 0.0001 --data_name=Office_Products --latent_clr_weight=0.3 --lr=0.001 --hidden_size=128 --max_seq_length=100 --hidden_dropout_prob=0.3 --num_hidden_layers=1 --weight_decay=0.0 --num_attention_heads=4 --attention_probs_dropout_prob=0.3 --anneal_cap=0.3 --total_annealing_step=10000 > log.log 2>&1 &
#nohup python main.py --gpu_id 3 --n_interest 8  --gm_kl_coeff 0.0001 --data_name=Office_Products --latent_clr_weight=0.3 --lr=0.001 --hidden_size=128 --max_seq_length=100 --hidden_dropout_prob=0.3 --num_hidden_layers=1 --weight_decay=0.0 --num_attention_heads=4 --attention_probs_dropout_prob=0.3 --anneal_cap=0.3 --total_annealing_step=10000 > log.log 2>&1 &

#nohup python main.py --gpu_id 1 --n_interest 2 --interest_celoss_coeff 0.001 --gm_kl_coeff 0.001 --data_name=Toys_and_Games --latent_clr_weight=0.3 --lr=0.001 --hidden_size=128 --max_seq_length=100 --hidden_dropout_prob=0.3 --num_hidden_layers=1 --weight_decay=0.0 --num_attention_heads=4 --model_name=VAGRecVD --attention_probs_dropout_prob=0.3 --anneal_cap=0.2 --total_annealing_step=10000 > log_toy_0726_0.001_0.0001.log 2>&1 &
#nohup python main.py --gpu_id 4 --interest_celoss_coeff 0.001 --gm_kl_coeff 0.0001 --data_name=Toys_and_Games --latent_clr_weight=0.3 --lr=0.001 --hidden_size=128 --max_seq_length=100 --hidden_dropout_prob=0.3 --num_hidden_layers=1 --weight_decay=0.0 --num_attention_heads=4 --model_name=VAGRecVD --attention_probs_dropout_prob=0.3 --anneal_cap=0.2 --total_annealing_step=10000 > log_toy_0726_0.001_0.001.log 2>&1 &

#nohup python main.py --gpu_id 1 --n_interest 2  --gm_kl_coeff 0.001 --data_name=Office_Products --latent_clr_weight=0.3 --lr=0.001 --hidden_size=128 --max_seq_length=100 --hidden_dropout_prob=0.3 --num_hidden_layers=1 --weight_decay=0.0 --num_attention_heads=4 --attention_probs_dropout_prob=0.3 --anneal_cap=0.3 --total_annealing_step=10000 > log.log 2>&1 &
#nohup python main.py --gpu_id 2 --n_interest 4  --gm_kl_coeff 0.001 --data_name=Office_Products --latent_clr_weight=0.3 --lr=0.001 --hidden_size=128 --max_seq_length=100 --hidden_dropout_prob=0.3 --num_hidden_layers=1 --weight_decay=0.0 --num_attention_heads=4 --attention_probs_dropout_prob=0.3 --anneal_cap=0.3 --total_annealing_step=10000 > log.log 2>&1 &
#nohup python main.py --gpu_id 3 --n_interest 8  --gm_kl_coeff 0.001 --data_name=Office_Products --latent_clr_weight=0.3 --lr=0.001 --hidden_size=128 --max_seq_length=100 --hidden_dropout_prob=0.3 --num_hidden_layers=1 --weight_decay=0.0 --num_attention_heads=4 --attention_probs_dropout_prob=0.3 --anneal_cap=0.3 --total_annealing_step=10000 > log.log 2>&1 &


#nohup python main.py  --gpu_id 1  --n_interest 2  --gm_kl_coeff 0.001 --data_name=Beauty --latent_clr_weight=0.6 --reparam_dropout_rate=0.1 --lr=0.001 --hidden_size=128 --max_seq_length=50 --hidden_dropout_prob=0.3 --num_hidden_layers=1 --weight_decay=0.0 --num_attention_heads=4 --model_name=MultiVAERec --attention_probs_dropout_prob=0.0 --anneal_cap=0.2 --total_annealing_step=10000 > log_beauty_0726.log 2>&1 &
#nohup python main.py  --gpu_id 2  --n_interest 4  --gm_kl_coeff 0.001 --data_name=Beauty --latent_clr_weight=0.6 --reparam_dropout_rate=0.1 --lr=0.001 --hidden_size=128 --max_seq_length=50 --hidden_dropout_prob=0.3 --num_hidden_layers=1 --weight_decay=0.0 --num_attention_heads=4 --model_name=MultiVAERec --attention_probs_dropout_prob=0.0 --anneal_cap=0.2 --total_annealing_step=10000 > log_beauty_0726.log 2>&1 &
#nohup python main.py  --gpu_id 3  --n_interest 8  --gm_kl_coeff 0.001 --data_name=Beauty --latent_clr_weight=0.6 --reparam_dropout_rate=0.1 --lr=0.001 --hidden_size=128 --max_seq_length=50 --hidden_dropout_prob=0.3 --num_hidden_layers=1 --weight_decay=0.0 --num_attention_heads=4 --model_name=MultiVAERec --attention_probs_dropout_prob=0.0 --anneal_cap=0.2 --total_annealing_step=10000 > log_beauty_0726.log 2>&1 &
#
#nohup python main.py  --gpu_id 1  --n_interest 2  --gm_kl_coeff 0.0001 --data_name=Beauty --latent_clr_weight=0.6 --reparam_dropout_rate=0.1 --lr=0.001 --hidden_size=128 --max_seq_length=50 --hidden_dropout_prob=0.3 --num_hidden_layers=1 --weight_decay=0.0 --num_attention_heads=4 --model_name=MultiVAERec --attention_probs_dropout_prob=0.0 --anneal_cap=0.2 --total_annealing_step=10000 > log_beauty_0726.log 2>&1 &
#nohup python main.py  --gpu_id 2  --n_interest 4  --gm_kl_coeff 0.0001 --data_name=Beauty --latent_clr_weight=0.6 --reparam_dropout_rate=0.1 --lr=0.001 --hidden_size=128 --max_seq_length=50 --hidden_dropout_prob=0.3 --num_hidden_layers=1 --weight_decay=0.0 --num_attention_heads=4 --model_name=MultiVAERec --attention_probs_dropout_prob=0.0 --anneal_cap=0.2 --total_annealing_step=10000 > log_beauty_0726.log 2>&1 &
#nohup python main.py  --gpu_id 3  --n_interest 8  --gm_kl_coeff 0.0001 --data_name=Beauty --latent_clr_weight=0.6 --reparam_dropout_rate=0.1 --lr=0.001 --hidden_size=128 --max_seq_length=50 --hidden_dropout_prob=0.3 --num_hidden_layers=1 --weight_decay=0.0 --num_attention_heads=4 --model_name=MultiVAERec --attention_probs_dropout_prob=0.0 --anneal_cap=0.2 --total_annealing_step=10000 > log_beauty_0726.log 2>&1 &
#
#
#nohup python main.py  --gpu_id 1  --n_interest 2  --gm_kl_coeff 0.00001 --data_name=Beauty --latent_clr_weight=0.6 --reparam_dropout_rate=0.1 --lr=0.001 --hidden_size=128 --max_seq_length=50 --hidden_dropout_prob=0.3 --num_hidden_layers=1 --weight_decay=0.0 --num_attention_heads=4 --model_name=MultiVAERec --attention_probs_dropout_prob=0.0 --anneal_cap=0.2 --total_annealing_step=10000 > log_beauty_0726.log 2>&1 &
#nohup python main.py  --gpu_id 2  --n_interest 4  --gm_kl_coeff 0.00001 --data_name=Beauty --latent_clr_weight=0.6 --reparam_dropout_rate=0.1 --lr=0.001 --hidden_size=128 --max_seq_length=50 --hidden_dropout_prob=0.3 --num_hidden_layers=1 --weight_decay=0.0 --num_attention_heads=4 --model_name=MultiVAERec --attention_probs_dropout_prob=0.0 --anneal_cap=0.2 --total_annealing_step=10000 > log_beauty_0726.log 2>&1 &
#nohup python main.py  --gpu_id 3  --n_interest 8  --gm_kl_coeff 0.00001 --data_name=Beauty --latent_clr_weight=0.6 --reparam_dropout_rate=0.1 --lr=0.001 --hidden_size=128 --max_seq_length=50 --hidden_dropout_prob=0.3 --num_hidden_layers=1 --weight_decay=0.0 --num_attention_heads=4 --model_name=MultiVAERec --attention_probs_dropout_prob=0.0 --anneal_cap=0.2 --total_annealing_step=10000 > log_beauty_0726.log 2>&1
#
#nohup python main.py --gpu_id 1  --n_interest 2  --gm_kl_coeff 0.001 --data_name=Tools_and_Home_Improvement --latent_clr_weight=0.4 --lr=0.001 --hidden_size=128 --max_seq_length=100 --hidden_dropout_prob=0.3 --num_hidden_layers=1 --weight_decay=0.0 --num_attention_heads=4 --model_name=VAGRecVD --attention_probs_dropout_prob=0.3 --anneal_cap=0.4 --total_annealing_step=5000 > log_tool_0726.log 2>&1 &
#nohup python main.py --gpu_id 2  --n_interest 4  --gm_kl_coeff 0.001 --data_name=Tools_and_Home_Improvement --latent_clr_weight=0.4 --lr=0.001 --hidden_size=128 --max_seq_length=100 --hidden_dropout_prob=0.3 --num_hidden_layers=1 --weight_decay=0.0 --num_attention_heads=4 --model_name=VAGRecVD --attention_probs_dropout_prob=0.3 --anneal_cap=0.4 --total_annealing_step=5000 > log_tool_0726.log 2>&1 &
#nohup python main.py --gpu_id 3  --n_interest 8  --gm_kl_coeff 0.001 --data_name=Tools_and_Home_Improvement --latent_clr_weight=0.4 --lr=0.001 --hidden_size=128 --max_seq_length=100 --hidden_dropout_prob=0.3 --num_hidden_layers=1 --weight_decay=0.0 --num_attention_heads=4 --model_name=VAGRecVD --attention_probs_dropout_prob=0.3 --anneal_cap=0.4 --total_annealing_step=5000 > log_tool_0726.log 2>&1 &
#
#nohup python main.py --gpu_id 1  --n_interest 2  --gm_kl_coeff 0.0001 --data_name=Tools_and_Home_Improvement --latent_clr_weight=0.4 --lr=0.001 --hidden_size=128 --max_seq_length=100 --hidden_dropout_prob=0.3 --num_hidden_layers=1 --weight_decay=0.0 --num_attention_heads=4 --model_name=VAGRecVD --attention_probs_dropout_prob=0.3 --anneal_cap=0.4 --total_annealing_step=5000 > log_tool_0726.log 2>&1 &
#nohup python main.py --gpu_id 2  --n_interest 4  --gm_kl_coeff 0.0001 --data_name=Tools_and_Home_Improvement --latent_clr_weight=0.4 --lr=0.001 --hidden_size=128 --max_seq_length=100 --hidden_dropout_prob=0.3 --num_hidden_layers=1 --weight_decay=0.0 --num_attention_heads=4 --model_name=VAGRecVD --attention_probs_dropout_prob=0.3 --anneal_cap=0.4 --total_annealing_step=5000 > log_tool_0726.log 2>&1 &
#nohup python main.py --gpu_id 3  --n_interest 8  --gm_kl_coeff 0.0001 --data_name=Tools_and_Home_Improvement --latent_clr_weight=0.4 --lr=0.001 --hidden_size=128 --max_seq_length=100 --hidden_dropout_prob=0.3 --num_hidden_layers=1 --weight_decay=0.0 --num_attention_heads=4 --model_name=VAGRecVD --attention_probs_dropout_prob=0.3 --anneal_cap=0.4 --total_annealing_step=5000 > log_tool_0726.log 2>&1 &
#
#nohup python main.py --gpu_id 1  --n_interest 2  --gm_kl_coeff 0.00001 --data_name=Tools_and_Home_Improvement --latent_clr_weight=0.4 --lr=0.001 --hidden_size=128 --max_seq_length=100 --hidden_dropout_prob=0.3 --num_hidden_layers=1 --weight_decay=0.0 --num_attention_heads=4 --model_name=VAGRecVD --attention_probs_dropout_prob=0.3 --anneal_cap=0.4 --total_annealing_step=5000 > log_tool_0726.log 2>&1 &
#nohup python main.py --gpu_id 2  --n_interest 4  --gm_kl_coeff 0.00001 --data_name=Tools_and_Home_Improvement --latent_clr_weight=0.4 --lr=0.001 --hidden_size=128 --max_seq_length=100 --hidden_dropout_prob=0.3 --num_hidden_layers=1 --weight_decay=0.0 --num_attention_heads=4 --model_name=VAGRecVD --attention_probs_dropout_prob=0.3 --anneal_cap=0.4 --total_annealing_step=5000 > log_tool_0726.log 2>&1 &
#nohup python main.py --gpu_id 3  --n_interest 8  --gm_kl_coeff 0.00001 --data_name=Tools_and_Home_Improvement --latent_clr_weight=0.4 --lr=0.001 --hidden_size=128 --max_seq_length=100 --hidden_dropout_prob=0.3 --num_hidden_layers=1 --weight_decay=0.0 --num_attention_heads=4 --model_name=VAGRecVD --attention_probs_dropout_prob=0.3 --anneal_cap=0.4 --total_annealing_step=5000 > log_tool_0726.log 2>&1
#
#
#nohup python main.py --gpu_id 4 --n_interest 2  --gm_kl_coeff 0.001 --data_name=Toys_and_Games --latent_clr_weight=0.3 --lr=0.001 --hidden_size=128 --max_seq_length=100 --hidden_dropout_prob=0.3 --num_hidden_layers=1 --weight_decay=0.0 --num_attention_heads=4 --model_name=VAGRecVD --attention_probs_dropout_prob=0.3 --anneal_cap=0.2 --total_annealing_step=10000 > log_toy_0726.log 2>&1 &
#nohup python main.py --gpu_id 2 --n_interest 4  --gm_kl_coeff 0.001 --data_name=Toys_and_Games --latent_clr_weight=0.3 --lr=0.001 --hidden_size=128 --max_seq_length=100 --hidden_dropout_prob=0.3 --num_hidden_layers=1 --weight_decay=0.0 --num_attention_heads=4 --model_name=VAGRecVD --attention_probs_dropout_prob=0.3 --anneal_cap=0.2 --total_annealing_step=10000 > log_toy_0726.log 2>&1 &
#nohup python main.py --gpu_id 3 --n_interest 8  --gm_kl_coeff 0.001 --data_name=Toys_and_Games --latent_clr_weight=0.3 --lr=0.001 --hidden_size=128 --max_seq_length=100 --hidden_dropout_prob=0.3 --num_hidden_layers=1 --weight_decay=0.0 --num_attention_heads=4 --model_name=VAGRecVD --attention_probs_dropout_prob=0.3 --anneal_cap=0.2 --total_annealing_step=10000 > log_toy_0726.log 2>&1 &
#
#
#nohup python main.py --gpu_id 4 --n_interest 2  --gm_kl_coeff 0.0001 --data_name=Toys_and_Games --latent_clr_weight=0.3 --lr=0.001 --hidden_size=128 --max_seq_length=100 --hidden_dropout_prob=0.3 --num_hidden_layers=1 --weight_decay=0.0 --num_attention_heads=4 --model_name=VAGRecVD --attention_probs_dropout_prob=0.3 --anneal_cap=0.2 --total_annealing_step=10000 > log_toy_0726.log 2>&1 &
#nohup python main.py --gpu_id 2 --n_interest 4  --gm_kl_coeff 0.0001 --data_name=Toys_and_Games --latent_clr_weight=0.3 --lr=0.001 --hidden_size=128 --max_seq_length=100 --hidden_dropout_prob=0.3 --num_hidden_layers=1 --weight_decay=0.0 --num_attention_heads=4 --model_name=VAGRecVD --attention_probs_dropout_prob=0.3 --anneal_cap=0.2 --total_annealing_step=10000 > log_toy_0726.log 2>&1 &
#nohup python main.py --gpu_id 3 --n_interest 8  --gm_kl_coeff 0.0001 --data_name=Toys_and_Games --latent_clr_weight=0.3 --lr=0.001 --hidden_size=128 --max_seq_length=100 --hidden_dropout_prob=0.3 --num_hidden_layers=1 --weight_decay=0.0 --num_attention_heads=4 --model_name=VAGRecVD --attention_probs_dropout_prob=0.3 --anneal_cap=0.2 --total_annealing_step=10000 > log_toy_0726.log 2>&1 &
#
#nohup python main.py --gpu_id 4 --n_interest 2  --gm_kl_coeff 0.00001 --data_name=Toys_and_Games --latent_clr_weight=0.3 --lr=0.001 --hidden_size=128 --max_seq_length=100 --hidden_dropout_prob=0.3 --num_hidden_layers=1 --weight_decay=0.0 --num_attention_heads=4 --model_name=VAGRecVD --attention_probs_dropout_prob=0.3 --anneal_cap=0.2 --total_annealing_step=10000 > log_toy_0726.log 2>&1 &
#nohup python main.py --gpu_id 2 --n_interest 4  --gm_kl_coeff 0.00001 --data_name=Toys_and_Games --latent_clr_weight=0.3 --lr=0.001 --hidden_size=128 --max_seq_length=100 --hidden_dropout_prob=0.3 --num_hidden_layers=1 --weight_decay=0.0 --num_attention_heads=4 --model_name=VAGRecVD --attention_probs_dropout_prob=0.3 --anneal_cap=0.2 --total_annealing_step=10000 > log_toy_0726.log 2>&1 &
#nohup python main.py --gpu_id 3 --n_interest 8  --gm_kl_coeff 0.00001 --data_name=Toys_and_Games --latent_clr_weight=0.3 --lr=0.001 --hidden_size=128 --max_seq_length=100 --hidden_dropout_prob=0.3 --num_hidden_layers=1 --weight_decay=0.0 --num_attention_heads=4 --model_name=VAGRecVD --attention_probs_dropout_prob=0.3 --anneal_cap=0.2 --total_annealing_step=10000 > log_toy_0726.log 2>&1

nohup python main.py --gpu_id 1 --n_interest 4  --data_name=Office_Products --prior uni --output_dir output_abl  --latent_clr_weight=0.3 --lr=0.001 --hidden_size=128 --max_seq_length=100 --hidden_dropout_prob=0.3 --num_hidden_layers=1 --weight_decay=0.0 --num_attention_heads=4 --attention_probs_dropout_prob=0.3 --anneal_cap=0.3 --total_annealing_step=10000 > log.log 2>&1 &
nohup python main.py --gpu_id 1 --n_interest 4  --data_name=Office_Products --prior uni --flag_train_interest 0 --output_dir output_abl --latent_clr_weight=0.3 --lr=0.001 --hidden_size=128 --max_seq_length=100 --hidden_dropout_prob=0.3 --num_hidden_layers=1 --weight_decay=0.0 --num_attention_heads=4 --attention_probs_dropout_prob=0.3 --anneal_cap=0.3 --total_annealing_step=10000 > log.log 2>&1 &

nohup python main.py  --gpu_id 2  --n_interest 4  --data_name=Beauty --prior uni --output_dir output_abl --latent_clr_weight=0.6 --reparam_dropout_rate=0.1 --lr=0.001 --hidden_size=128 --max_seq_length=50 --hidden_dropout_prob=0.3 --num_hidden_layers=1 --weight_decay=0.0 --num_attention_heads=4 --model_name=MultiVAERec --attention_probs_dropout_prob=0.0 --anneal_cap=0.2 --total_annealing_step=10000 > log_beauty_0726.log 2>&1 &
nohup python main.py  --gpu_id 2  --n_interest 4  --data_name=Beauty --prior uni --flag_train_interest 0 --output_dir output_abl --latent_clr_weight=0.6 --reparam_dropout_rate=0.1 --lr=0.001 --hidden_size=128 --max_seq_length=50 --hidden_dropout_prob=0.3 --num_hidden_layers=1 --weight_decay=0.0 --num_attention_heads=4 --model_name=MultiVAERec --attention_probs_dropout_prob=0.0 --anneal_cap=0.2 --total_annealing_step=10000 > log_beauty_0726.log 2>&1 &

nohup python main.py --gpu_id 3  --n_interest 2  --prior uni --output_dir output_abl --data_name=Tools_and_Home_Improvement --latent_clr_weight=0.4 --lr=0.001 --hidden_size=128 --max_seq_length=100 --hidden_dropout_prob=0.3 --num_hidden_layers=1 --weight_decay=0.0 --num_attention_heads=4 --model_name=VAGRecVD --attention_probs_dropout_prob=0.3 --anneal_cap=0.4 --total_annealing_step=5000 > log_tool_0726.log 2>&1
nohup python main.py --gpu_id 3  --n_interest 2  --prior uni  --flag_train_interest 0 --output_dir output_abl --data_name=Tools_and_Home_Improvement --latent_clr_weight=0.4 --lr=0.001 --hidden_size=128 --max_seq_length=100 --hidden_dropout_prob=0.3 --num_hidden_layers=1 --weight_decay=0.0 --num_attention_heads=4 --model_name=VAGRecVD --attention_probs_dropout_prob=0.3 --anneal_cap=0.4 --total_annealing_step=5000 > log_tool_0726.log 2>&1

nohup python main.py --gpu_id 3 --n_interest 4   --prior uni --output_dir output_abl --data_name=Toys_and_Games --latent_clr_weight=0.3 --lr=0.001 --hidden_size=128 --max_seq_length=100 --hidden_dropout_prob=0.3 --num_hidden_layers=1 --weight_decay=0.0 --num_attention_heads=4 --model_name=VAGRecVD --attention_probs_dropout_prob=0.3 --anneal_cap=0.2 --total_annealing_step=10000 > log_toy_0726.log 2>&1 &
nohup python main.py --gpu_id 3 --n_interest 4    --prior uni  --flag_train_interest 0 --output_dir output_abl --data_name=Toys_and_Games --latent_clr_weight=0.3 --lr=0.001 --hidden_size=128 --max_seq_length=100 --hidden_dropout_prob=0.3 --num_hidden_layers=1 --weight_decay=0.0 --num_attention_heads=4 --model_name=VAGRecVD --attention_probs_dropout_prob=0.3 --anneal_cap=0.2 --total_annealing_step=10000 > log_toy_0726.log 2>&1 &
