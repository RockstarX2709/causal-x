import json


class EvidenceBuilder:

    @staticmethod
    def build(
        business_analysis,
        causal_analysis,
        forecast,
        optimization,
        risk,
        audit
    ):

        package = {
            "business_analysis":
                business_analysis,

            "causal_analysis":
                causal_analysis,

            "forecast":
                forecast,

            "optimization":
                optimization,

            "risk":
                risk,

            "audit":
                audit,
        }

        return json.dumps(
            package,
            indent=2,
            default=str
        )