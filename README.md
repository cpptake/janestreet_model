# Jane Street Market Prediction
<a>https://www.kaggle.com/c/jane-street-market-prediction

## 最終提出モデルの説明
### モデル
* EmbeddingNN(Pytorch)
* ResNet(Pytorch)
* NN with earlystop(tensorflow)
上記3モデルのアンサンブル

### 特徴
* 特徴量にfeature_neutralizationを適用し、Overfitを防ぐ
<a>https://www.kaggle.com/code1110/janestreet-avoid-overfit-feature-neutralization
* Optunaを用いてHyperparameterチューニングを実施


