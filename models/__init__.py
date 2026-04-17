# models/__init__.py

from .xgboost import XGBoostModel
from .randomforest import RandomForestModel
from .cnn import CNNClassifier
from .transformer_onehot import TransformerOneHotModel
from .transformer_onehot_vh import TransformerOneHotModel_VH
from .transformer_lm_old import TransformerLMModel


available_models = {
    "xgboost": XGBoostModel,
    "rf": RandomForestModel,
    "cnn": CNNClassifier,
    "transformer_onehot": TransformerOneHotModel,   
    "transformer_onehot_vh": TransformerOneHotModel_VH,
    "transformer_lm": TransformerLMModel
}