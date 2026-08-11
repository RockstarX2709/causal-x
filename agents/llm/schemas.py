from pydantic import BaseModel, Field
from typing import List


class DecisionRecommendation(BaseModel):

    recommendation: str = Field(
        description=(
            "The recommended business action."
        )
    )

    rationale: List[str] = Field(
        description=(
            "Evidence-based reasons supporting "
            "the recommendation."
        )
    )

    causal_evidence: str = Field(
        description=(
            "Explanation of the causal evidence."
        )
    )

    forecast_evidence: str = Field(
        description=(
            "Explanation of forecast evidence."
        )
    )

    risk_assessment: str = Field(
        description=(
            "Explanation of decision risk."
        )
    )

    audit_assessment: str = Field(
        description=(
            "Explanation of audit confidence."
        )
    )

    confidence: str = Field(
        description=(
            "HIGH, MEDIUM, or LOW."
        )
    )

    human_approval_required: bool = Field(
        description=(
            "Whether human approval is required "
            "before executing the recommendation."
        )
    )

    caveats: List[str] = Field(
        description=(
            "Important limitations or caveats."
        )
    )
    