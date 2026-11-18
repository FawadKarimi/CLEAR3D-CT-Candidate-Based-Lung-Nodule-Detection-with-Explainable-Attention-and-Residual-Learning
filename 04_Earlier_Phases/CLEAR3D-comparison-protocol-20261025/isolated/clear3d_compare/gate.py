"""Execution gate. A draft decision record cannot start a GPU run."""

REQUIRED = {
    "reference_release": "HOLD",
    "cap4000": "NOT_ADOPTED",
    "evaluation_population": "ADOPTED",
    "gpu_budget": "ADOPTED",
}


def launch_allowed(decisions):
    return all(decisions.get(key) == value for key, value in REQUIRED.items())
