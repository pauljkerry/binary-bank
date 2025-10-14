from typing import Type
import importlib

_REGISTRY = {
    "xgb":  "src.models.xgb_cv_trainer:XGBCVTrainer",
    "lgbm": "src.models.lgbm_cv_trainer:LGBMCVTrainer",
    "cb": "src.models.cb_cv_trainer:CBCVTrainer",
    "rfr": "src.models.rfr_cv_trainer:RFRCVTrainer",
    "rfc": "src.models.rfc_cv_trainer:RFCCVTrainer",
    "mlp": "src.models.mlp_cv_trainer:MLPCVTrainer",
    "realmlp": "src.models.realmlp_cv_trainer:RealMLPCVTrainer",
    "tabnet": "src.models.tabnet_cv_trainer:TabNetCVTrainer",
    "logreg": "src.models.logreg_cv_trainer:LogRegCVTrainer",
    "lasso": "src.models.lasso_cv_trainer:LassoCVTrainer",
    "ridge": "src.models.ridge_cv_trainer:RidgeCVTrainer"
}


def _resolve(spec: str):
    module, name = spec.split(":")
    return getattr(importlib.import_module(module), name)


def get_trainer(model_type: str) -> Type:
    try:
        spec = _REGISTRY[model_type]
    except KeyError:
        raise ValueError(
            f"Unknown model type: {model_type}. Available: "
            f"{', '.join()}"
        )
    return _resolve(spec)
