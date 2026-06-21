"""
Unified factory function for Domain Predictors in DGS.

Supported types:
- 'dist' / 'kl'    : Standard KL-divergence based (AdaptiveDomainPredictor_Dist)
- 'svd'  : SVD-regularized KL covariance (AdaptiveDomainPredictor_SVD)
"""

from .adaptive_domain_predictor import AdaptiveDomainPredictor_Dist
from .svd_domain_predictor import AdaptiveDomainPredictor_SVD

_TYPE_MAP = {
    'dist': AdaptiveDomainPredictor_Dist,
    'kl': AdaptiveDomainPredictor_Dist,
    'svd': AdaptiveDomainPredictor_SVD
}


def AdaptiveDomainPredictor(domain_predictor_cfg, **kwargs):
    """
    Unified factory function for domain predictors.

    Args:
        domain_predictor_cfg: Config object (dict or ConfigDict) with a 'type' attribute.
        **kwargs: Forwarded to the chosen predictor class (num_tasks, seen_tasks, etc.).

    Returns:
        Instantiated domain predictor.
    """
    pred_type = getattr(domain_predictor_cfg, 'type', 'svd')

    assert pred_type in _TYPE_MAP, f"Unknown domain predictor type: {pred_type}"
    cls = _TYPE_MAP[pred_type]

    return cls(domain_predictor_cfg=domain_predictor_cfg, **kwargs)
