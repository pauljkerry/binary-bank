def get_trainer(model_type: str):
    """
    model名を受け取りCVTrainerを返す関数

    Parameters
    ----------
    model_type : str
        モデル名

    Returns
    -------
    CVTrainer : Class
        CVTrainer
    """
    if model_type == "xgb":
        from src.models.xgb.xgb_cv_trainer import XGBCVTrainer
        return XGBCVTrainer
    elif model_type == "lgbm":
        from src.models.lgbm.lgbm_cv_trainer import LGBMCVTrainer
        return LGBMCVTrainer
    elif model_type == "cb":
        from src.models.cb.cb_cv_trainer import CBCVTrainer
        return CBCVTrainer
    elif model_type == "rfr":
        from src.models.rfr.rfr_cv_trainer import RFRCVTrainer
        return RFRCVTrainer
    elif model_type == "rfc":
        from src.models.rfc.rfc_cv_trainer import RFCCVTrainer
        return RFCCVTrainer
    elif model_type == "mlp":
        from src.models.mlp.mlp_cv_trainer import MLPCVTrainer
        return MLPCVTrainer
    elif model_type == "tabnet":
        from src.models.tabnet.tabnet_cv_trainer import TabNetCVTrainer
        return TabNetCVTrainer
    elif model_type == "logreg":
        from src.models.logreg.logreg_cv_trainer import LogRegCVTrainer
        return LogRegCVTrainer
    elif model_type == "ridge":
        from src.models.ridge.ridge_cv_trainer import RidgeCVTrainer
        return RidgeCVTrainer
    elif model_type == "lasso":
        from src.models.lasso.lasso_cv_trainer import LassoCVTrainer
        return LassoCVTrainer
    else:
        raise ValueError(f"Unknown model type: {model_type}")