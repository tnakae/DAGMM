# DAGMM Tensorflow 版
DAGMM (Deep Autoencoding Gaussian Mixture Model) の Tensorflow 実装です。

この実装は、次の論文：
**Deep Autoencoding Gaussian Mixture Model for Unsupervised Anomaly Detection**
[[Bo Zong et al (2018)]](https://openreview.net/pdf?id=BJJLHbb0-)
に記載された内容に準じて実装しました。

※この実装は論文著者とは無関係です。

# 動作要件
- python (3.5-3.6)
- Tensorflow <= 1.15 (2系には未対応です)
- Numpy
- sklearn

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

オプションの詳細については [dagmm/dagmm.py](dagmm/dagmm.py) の docstring を参照してください。

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

# 学習済みモデルをディレクトリに保存する
model.save("./fitted_model")

# 学習済みモデルをディレクトリから読み込む
model.restore("./fitted_model")
```

## Jupyter Notebook サンプル
次のJupyter notebook の実行サンプルを用意しました。
- [DAGMM の利用例](Example_DAGMM_ja.ipynb) :
このサンプルでは、混合正規分布に対して適用した結果となっています。
利用方法を手っ取り早く知りたい場合、まずこのサンプルを見てください。
- [KDDCup99 10% データによる異常検知評価](KDDCup99_ja.ipynb) :
論文と同条件により、KDDCup99 10% データに対する異常検知を実施し、
精度評価を行うサンプルです(pandasが必要です)

# 補足
## 混合正規分布(GMM)の実装について
論文では、エネルギーの定式化で混合正規分布の直接的な表記がされています。
この算出では、多次元正規分布の逆行列が必要となりますが、場合によっては
逆行列の計算ができません。

これを避けるために、この実装では共分散行列のコレスキー分解(Cholesky Decomposition)
を用いています([Tensorflow における GMM の実装]((https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/factorization/python/ops/gmm_ops.py))でも同様のロジックがあり、参考にしました)

``DAGMM.fit()``において、共分散行列のコレスキー分解をしておき、算出された
三角行列を ``DAGMM.predict()`` で利用しています。

さらに、共分散行列の対角行列にあらかじめ小さな値(1e-6)を加えることで、
安定的にコレスキー分解ができるようにしています。
(Tensorflow の GMM でも同様のロジックがあり、
[DAGMMの別実装の実装者](https://github.com/danieltan07/dagmm)も
同じ事情について言及しています)

## 共分散パラメータ λ2 について
共分散の対角成分を制御するパラメータλ2のデフォルト値は
0.0001 としてあります（論文では 0.005 がおすすめとなっている）
これは、0.005 とした場合に共分散が大きくなりすぎて、大きなクラスタ
が選ばれる傾向にあったためです。ただしこれはデータの傾向、および
前処理の手順（例えば、データの正規化の方法）にも依存すると考えられます。
意図した精度が得られない場合は、λ2 の値をコントロールすることを
お勧めします。
