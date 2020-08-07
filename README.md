# adain

style transfer

## train

- 訓練時は以下で
- データセットの前処理をし直す際は--perporcessを指定
- configのtrain_c_dirにコンテンツ訓練データのdirを記述
- configのtrain_s_dirにスタイル訓練データのdirを記述
- train_resultに結果などを出力

```
python train.py [--preprocess]
```

## eval

- configのtrain_c_dirにコンテンツ訓練データのdirを記述
- configのtrain_s_dirにスタイル訓練データのdirを記述
- eval_resultに結果を出力

```
python eval.py
```

