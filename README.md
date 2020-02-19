## requirements

## 実行手順

仮想環境
```
conda activate py35
```

学習と予測
```
python segnet_with_kmeans.py --myloss --secmax --time --start 20 --end 21 --alpha 0.1 --lambda_p 0.01 --tau 10000
```

## Environment
torch>=1.1.0
python==3.5
