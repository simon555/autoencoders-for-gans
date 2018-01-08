sinter --gres=gpu --qos=low
source activate mypy35
python main.py --idxModel=Resnet_Modified --NbatchTrain=64

