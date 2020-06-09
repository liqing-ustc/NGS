# conda create -y -n ngs anaconda python=3.7
# source activate ngs
# pip install -r requirements.txt

python main.py --mode BS --num_epochs 100 --lr 0.0005
python main.py --mode RL --num_epochs 200 --pretrain 'pretrain-sym_net/sym_net_acc50.ckpt'
python main.py --mode MAPO --num_epochs 200 --pretrain 'pretrain-sym_net/sym_net_acc50.ckpt'
python main.py --mode RL --num_epochs 200
python main.py --mode MAPO --num_epochs 200

python main.py --mode BS --data_used 0.75 --num_epochs 100
python main.py --mode BS --data_used 0.50 --num_epochs 100
python main.py --mode BS --data_used 0.25 --num_epochs 100

# python main.py --mode MAPO --data_used 0.75 --num_epochs 100
# python main.py --mode MAPO --data_used 0.50 --num_epochs 100
# python main.py --mode MAPO --data_used 0.25 --num_epochs 100

# python main.py --mode MAPO --data_used 0.75 --num_epochs 100 --pretrain 'pretrain-sym_net/sym_net_acc50.ckpt'
# python main.py --mode MAPO --data_used 0.50 --num_epochs 100 --pretrain 'pretrain-sym_net/sym_net_acc50.ckpt'
# python main.py --mode MAPO --data_used 0.25 --num_epochs 100 --pretrain 'pretrain-sym_net/sym_net_acc50.ckpt'

