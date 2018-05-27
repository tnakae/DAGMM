# DAGMM Tensorflow 版
DAGMM (Deep Autoencoding Gaussian Mixture Model) の Tensorflow 実装です。

この実装は、次の論文：
**Deep Autoencoding Gaussian Mixture Model for Unsupervised Anomaly Detection**
[[Bo Zong et al (2018)]](https://openreview.net/pdf?id=BJJLHbb0-)
に記載された内容に準じて実装しました。

※この実装は論文著者とは無関係です。

# 動作要件
- python 3
- Tensorflow
- Numpy

# 利用方法
DAGMMを利用するには、まずDAGMMオブジェクトを生成します。
コンストラクタにおいて次の4つの引数の指定が必須です。

- ``comp_hiddens`` : intのリスト
  - 圧縮モデル(Compression Network)における層構造を指定します。
  - 例えば、``[n1, n2]``のように指定した場合、圧縮モデルは次のようになります:
  ``input_size -> n1 -> n2 -> n1 -> input_sizes``
- ``comp_activation`` : 関数
  - 圧縮モデルにおける活性化関数
- ``est_hiddens`` : intのリスト
  - GMMの所属確率を案出する推測モデル(Estimation Network)における
    層構造を指定します。
  - リストの最後の要素は、GMMにおける隠れクラスの数(n_comp)となります。
  - 例えば、``[n1, n2]``のように指定した場合、推測モデルは次のようになります。
    ``input_size -> n1 -> n2``, 最後の要素 ``n2`` は隠れクラス数となります。
- ``est_activation`` : function
  - 推測モデルにおける活性化関数

オブジェクト生成後、学習データに対してあてはめ(fit)を行い、
その後、スコアを算出したいデータに対して予測(predict)を行います。
(scikit-learnにおける予測モデルの利用方法と似ています)

オプションの詳細については dagmm/dagmm.py の docstring を参照してください。

# 利用例
## シンプルな例
``` python
import tensorflow as tf
from dagmm import DAGMM

# 初期化
model = DAGMM(
  comp_hiddens=[32,16,2], comp_activation=tf.nn.tanh,
  est_hiddens=[16.8], est_activation=tf.nn.tanh,
  est_dropout_ratio=0.25
)
# 学習データを当てはめる
model.fit(x_train)

# エネルギーの算出
# (エネルギーが高いほど異常)
energy = model.predict(x_test)
```

## Jupyter Notebook サンプル
Jupyter notebook での実行サンプルを用意しました。
このサンプルでは、混合ガウス分布に対して適用した結果となっています。
(sklearn が必要です)

# 補足

# 混合正規分布(GMM)の実装について
論文では、エネルギーの定式化で混合正規分布の直接的な表記がされています。
この算出では、多次元正規分布の逆行列が必要となりますが、場合によっては
逆行列の計算ができません。

これを避けるために、この実装では共分散行列のコレスキー分解(Cholesky Decomposition)
を用いています(Tensorflow における GMM の実装でも同様のロジックがあり、参考にしました)

``DAGMM.fit()``において、共分散行列のコレスキー分解をしておき、算出された
三角行列を ``DAGMM.predict()`` で利用しています。

さらに、共分散行列の対角行列にあらかじめ小さな値(1e-3)を加えることで、
安定的にコレスキー分解ができるようにしています。
(Tensorflow の GMM でも同様のロジックがあり、DAGMMの別実装の実装者も
同じ事情について言及しています)
