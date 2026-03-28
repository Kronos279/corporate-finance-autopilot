from app.config import settings
from app.models.company import Assumptions, SensitivityCell, SensitivityResult
from app.services.financial_engine.model_builder import dcf_valuation, forecast


def sensitivity_table(
	base_assumptions: Assumptions,
	param_x: str,
	param_y: str,
	range_x: list[float],
	range_y: list[float],
	starting_revenue: float | None = None,
	net_debt: float = 0.0,
	shares_outstanding: float | None = None,
	current_price: float | None = None,
) -> SensitivityResult:
	starting_revenue = starting_revenue if starting_revenue is not None else settings.default_starting_revenue
	shares_outstanding = shares_outstanding if shares_outstanding is not None else settings.default_shares_outstanding
	current_price = current_price if current_price is not None else settings.default_current_price
	cells: list[SensitivityCell] = []

	for x_val in range_x:
		for y_val in range_y:
			scenario = base_assumptions.model_copy(deep=True)
			if hasattr(scenario, param_x):
				setattr(scenario, param_x, x_val)
			if hasattr(scenario, param_y):
				setattr(scenario, param_y, y_val)

			fcst = forecast(scenario, years=5, starting_revenue=starting_revenue)
			dcf = dcf_valuation(
				fcst,
				scenario.wacc,
				scenario.terminal_growth_rate,
				current_price=current_price,
				net_debt=net_debt,
				shares_outstanding=shares_outstanding,
			)

			cells.append(
				SensitivityCell(
					param_x_value=float(x_val),
					param_y_value=float(y_val),
					output_value=float(dcf.implied_share_price),
				)
			)

	return SensitivityResult(
		param_x_name=param_x,
		param_y_name=param_y,
		output_name="implied_share_price",
		cells=cells,
	)
