from models.risk_model import (
    RiskModel
)


class RiskAgent:

    def __init__(self):

        self.model = RiskModel(
            simulations=10_000,
            seed=42
        )

    def analyze(
        self,
        expected_profit,
        volatility=0.10
    ):

        report = self.model.analyze(
            expected_profit=
                expected_profit,

            volatility=
                volatility
        )

        probability_loss = (
            report[
                "probability_of_loss"
            ]
        )

        if probability_loss < 0.05:

            risk_level = "LOW"

        elif probability_loss < 0.15:

            risk_level = "MEDIUM"

        else:

            risk_level = "HIGH"

        report[
            "risk_level"
        ] = risk_level

        return report


if __name__ == "__main__":

    agent = RiskAgent()

    result = agent.analyze(
        expected_profit=100_000_000,
        volatility=0.12
    )

    print("=" * 65)
    print("CAUSAL-X RISK AGENT")
    print("=" * 65)

    for key, value in result.items():

        if key == "probability_of_loss":

            print(
                f"{key}: "
                f"{value * 100:.2f}%"
            )

        elif isinstance(value, float):

            print(
                f"{key}: "
                f"₹{value:,.2f}"
            )

        else:

            print(
                f"{key}: {value}"
            )