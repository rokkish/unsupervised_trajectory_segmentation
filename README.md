## requirements

## 実行手順

仮想環境
```
conda activate py35
```

学習と予測
```
python segnet_with_kmeans.py  --myloss --secmax --time --net segnet --alpha 0.1 --lambda_p 0.01 --tau 10000 -e 2 --epoch_all 1 --start 1 --end 2 --animal bird -d test
```

## Environment
torch>=1.1.0
python==3.5
