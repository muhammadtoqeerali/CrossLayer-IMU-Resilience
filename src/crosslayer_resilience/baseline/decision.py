from __future__ import annotations

import torch


HISTORICAL_PREDICTION_BIAS = 0.9

ACTIVITY_CLASS = 0
FALLING_CLASS = 1


def decision_from_probabilities(
    probabilities: torch.Tensor,
    prediction_bias: float = HISTORICAL_PREDICTION_BIAS,
) -> torch.Tensor:
    """
    Exact recovered historical streaming decision semantics.

    Original behavior:

        output defaults to Activity
        if max(probabilities) > prediction_bias:
            output = argmax(probabilities)

    With the binary historical task and prediction_bias=0.9:

        Falling iff P(Falling) > 0.9
        Activity otherwise

    The comparison is intentionally strict (>).

    This historical decision rule remains explicit and separate from ordinary
    argmax classification.
    """
    if probabilities.ndim != 2:
        raise ValueError(
            f"Expected [B,C], got {tuple(probabilities.shape)}"
        )

    max_prob, argmax_class = probabilities.max(dim=1)

    output = torch.zeros(
        probabilities.shape[0],
        dtype=torch.long,
        device=probabilities.device,
    )

    accepted = max_prob > prediction_bias

    output[accepted] = argmax_class[accepted]

    return output


def decision_from_logits(
    logits: torch.Tensor,
    prediction_bias: float = HISTORICAL_PREDICTION_BIAS,
) -> torch.Tensor:
    """
    Historical deployment-style decision from already-computed logits.
    """
    probabilities = torch.softmax(
        logits,
        dim=1,
    )

    return decision_from_probabilities(
        probabilities,
        prediction_bias=prediction_bias,
    )


def argmax_from_logits(
    logits: torch.Tensor,
) -> torch.Tensor:
    """
    Standard classifier decision.

    This is kept separate from the historical 0.9 streaming rule so the two
    evaluation semantics can never be silently mixed.
    """
    if logits.ndim != 2:
        raise ValueError(
            f"Expected [B,C], got {tuple(logits.shape)}"
        )

    return logits.argmax(dim=1)
