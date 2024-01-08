# LogCL
The code of LogCL

### Process data
First, unpack the data files 

For the three ICEWS datasets 'ICEWS18', 'ICEWS14', 'ICEWS05-15', 'GDELT', go into the dataset folder in the `./data` directory and run the following command to construct the static graph and the query historical subgraph.
```
cd ./data/
python get_his_subg.py
cd ./<dataset>
python ent2word.py
cd .. 
python get_his_subg.py
```

### Train models
Then the following commands can be used to train the proposed models. By default, dev set evaluation results will be printed when training terminates.

1. Train models
```
python src/main.py -d ICEWS14 --train-history-len 7 --test-history-len 7 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --gpu=0 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm --weight 0.5  --entity-prediction --angle 10 --discount 1 --pre-weight 0.9  --pre-type all --add-static-graph  --temperature 0.03
```
### Cite
Please cite our paper if you find this code useful for your research.
~~~
@article{chen2023local,
  title={Local-Global History-aware Contrastive Learning for Temporal Knowledge Graph Reasoning},
  author={Chen, Wei and Wan, Huaiyu and Wu, Yuting and Zhao, Shuyuan and Cheng, Jiayaqi and Li, Yuxin and Lin, Youfang},
  journal={arXiv preprint arXiv:2312.01601},
  year={2023}
}
~~~


