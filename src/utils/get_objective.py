def get_objective(model_type: str):
    """
    model名を受け取りCVTrainerを返す関数

    Parameters
    ----------
    model_type : str
        モデル名

    Returns
    -------
    create_objective : function
        Optunaのobjective作成関数
    """
    if model_type == "xgb":
        from src.models.xgb.xgb_objective import create_objective
        return create_objective
    elif model_type == "lgbm":
        from src.models.lgbm.lgbm_objective import create_objective
        return create_objective
    elif model_type == "cb":
        from src.models.cb.cb_objective import create_objective
        return create_objective
    elif model_type == "rfr":
        from src.models.rfr.rfr_objective import create_objective
        return create_objective
    elif model_type == "rfc":
        from src.models.rfc.rfc_objective import create_objective
        return create_objective
    elif model_type == "mlp":
        from src.models.mlp.mlp_objective import create_objective
        return create_objective
    elif model_type == "logreg":
        from src.models.logreg.logreg_objective import create_objective
        return create_objective
    elif model_type == "ridge":
        from src.models.ridge.ridge_objective import create_objective
        return create_objective
    else:
        raise ValueError(f"Unknown model type: {model_type}")