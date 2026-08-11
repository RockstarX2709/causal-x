class ScenarioAgent:

    def create_price_scenarios(
        self,
        base_price,
        changes=None
    ):

        if changes is None:

            changes = [
                -0.10,
                -0.075,
                -0.05,
                -0.025,
                0.0,
                0.025,
                0.05,
                0.075,
                0.10,
            ]

        scenarios = []

        for change in changes:

            scenarios.append(
                {
                    "type":
                        "price",

                    "price_change":
                        change,

                    "new_price":
                        base_price
                        * (1 + change),
                }
            )

        return scenarios

    def create_marketing_scenarios(
        self,
        current_marketing
    ):

        multipliers = [
            0.75,
            0.90,
            1.00,
            1.10,
            1.25,
        ]

        scenarios = []

        for multiplier in multipliers:

            scenarios.append(
                {
                    "type":
                        "marketing",

                    "marketing_change":
                        multiplier - 1,

                    "marketing_spend":
                        current_marketing
                        * multiplier,
                }
            )

        return scenarios

    def create_combined_scenarios(
        self,
        base_price,
        current_marketing
    ):

        price_changes = [
            -0.10,
            -0.05,
            0.0,
            0.05,
            0.10,
        ]

        marketing_multipliers = [
            0.75,
            1.0,
            1.25,
        ]

        scenarios = []

        for price_change in price_changes:

            for multiplier in (
                marketing_multipliers
            ):

                scenarios.append(
                    {
                        "price_change":
                            price_change,

                        "new_price":
                            base_price
                            * (1 + price_change),

                        "marketing_spend":
                            current_marketing
                            * multiplier,
                    }
                )

        return scenarios


if __name__ == "__main__":

    agent = ScenarioAgent()

    scenarios = (
        agent.create_combined_scenarios(
            base_price=10_000,
            current_marketing=300_000
        )
    )

    print("=" * 65)
    print("CAUSAL-X SCENARIO AGENT")
    print("=" * 65)

    print(
        f"Generated {len(scenarios)} scenarios."
    )

    for scenario in scenarios[:10]:

        print(scenario)