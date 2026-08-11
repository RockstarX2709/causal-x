import numpy as np


class BusinessConstraints:

    def __init__(
        self,
        min_price_ratio=0.90,
        max_price_ratio=1.10,
        min_marketing=100_000,
        max_marketing=500_000,
        minimum_inventory=0,
        maximum_inventory=100_000,
    ):

        self.min_price_ratio = (
            min_price_ratio
        )

        self.max_price_ratio = (
            max_price_ratio
        )

        self.min_marketing = (
            min_marketing
        )

        self.max_marketing = (
            max_marketing
        )

        self.minimum_inventory = (
            minimum_inventory
        )

        self.maximum_inventory = (
            maximum_inventory
        )

    # =========================================================
    # PRICE
    # =========================================================

    def price_bounds(
        self,
        baseline_price
    ):

        return (
            baseline_price
            * self.min_price_ratio,

            baseline_price
            * self.max_price_ratio,
        )

    # =========================================================
    # MARKETING
    # =========================================================

    def marketing_bounds(self):

        return (
            self.min_marketing,
            self.max_marketing,
        )

    # =========================================================
    # CHECK PRICE
    # =========================================================

    def valid_price(
        self,
        price,
        baseline_price
    ):

        lower, upper = (
            self.price_bounds(
                baseline_price
            )
        )

        return (
            lower
            <= price
            <= upper
        )

    # =========================================================
    # CHECK MARKETING
    # =========================================================

    def valid_marketing(
        self,
        marketing
    ):

        return (
            self.min_marketing
            <= marketing
            <= self.max_marketing
        )

    # =========================================================
    # CHECK INVENTORY
    # =========================================================

    def valid_inventory(
        self,
        inventory
    ):

        return (
            self.minimum_inventory
            <= inventory
            <= self.maximum_inventory
        )

    # =========================================================
    # CHECK COMPLETE DECISION
    # =========================================================

    def validate(
        self,
        price,
        baseline_price,
        marketing,
        ending_inventory
    ):

        violations = []

        if not self.valid_price(
            price,
            baseline_price
        ):

            violations.append(
                "Price constraint violated"
            )

        if not self.valid_marketing(
            marketing
        ):

            violations.append(
                "Marketing budget constraint violated"
            )

        if not self.valid_inventory(
            ending_inventory
        ):

            violations.append(
                "Inventory constraint violated"
            )

        return {
            "valid":
                len(violations) == 0,

            "violations":
                violations,
        }


if __name__ == "__main__":

    constraints = BusinessConstraints()

    result = constraints.validate(
        price=9_500,
        baseline_price=10_000,
        marketing=300_000,
        ending_inventory=5_000,
    )

    print("=" * 65)
    print("CAUSAL-X BUSINESS CONSTRAINTS")
    print("=" * 65)

    print(
        f"Valid: {result['valid']}"
    )

    if result["violations"]:

        print("\nViolations:")

        for violation in (
            result["violations"]
        ):

            print(
                f"  - {violation}"
            )