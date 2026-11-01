import pandas as pd

from app.services.profiler import profile_dataset
from app.services.tools import run_tool
from app.agent.workflow import run_analysis


def frame():
    return pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=6, freq="MS"),
        "region": ["North", "South", "North", "South", "North", "South"],
        "revenue": [100, 90, 120, 70, 150, 50],
        "note": ["a", "b", None, "d", "e", "f"],
    })


def test_profile_reports_quality_and_types():
    result = profile_dataset(frame()).as_dict()
    assert result["rows"] == 6
    assert result["columns"] == 4
    assert result["missing"] == {"note": 1}
    assert result["date_columns"] == ["date"]
    assert result["duplicate_rows"] == 0


def test_profile_detects_duplicates_and_numeric_summary():
    df = pd.concat([frame(), frame().iloc[[0]]], ignore_index=True)
    result = profile_dataset(df).as_dict()
    assert result["duplicate_rows"] == 1
    assert result["numeric_stats"]["revenue"]["mean"] is not None


def test_groupby_and_trend_tools():
    df = frame()
    grouped = run_tool("groupby_aggregation", df, {"group_by": "region", "metric": "revenue"})
    assert grouped["groups"][0]["region"] == "North"
    trend = run_tool("trend_analysis", df, {"date_column": "date", "metric": "revenue"})
    assert len(trend["points"]) == 6
    assert trend["points"][0]["total"] == 100


def test_missing_correlation_and_outlier_tools():
    df = frame()
    assert run_tool("missing_analysis", df)["total_missing"] == 1
    correlations = run_tool("correlation_analysis", df.assign(units=[1, 2, 3, 4, 5, 6]))
    assert correlations["strongest_pairs"][0]["left"] in {"revenue", "units"}
    outliers = run_tool("outlier_detection", df.assign(revenue=[1, 1, 1, 1, 1, 1000]), {"column": "revenue"})
    assert outliers["outlier_count"] == 1


def test_segment_change_comparison_uses_equal_date_windows():
    df = pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=12, freq="MS").repeat(2),
        "region": ["North", "South"] * 12,
        "revenue": [100, 100, 100, 100, 100, 100, 50, 110, 50, 110, 50, 110,
                    50, 110, 50, 110, 50, 110, 50, 110, 50, 110, 50, 110],
    })
    result = run_tool("segment_change_analysis", df, {
        "date_column": "date", "group_by": "region", "metric": "revenue"
    })
    assert result["groups"][0]["region"] == "North"
    assert result["groups"][0]["change"] < 0


def test_tool_rejects_unknown_columns_and_tools():
    df = frame()
    try:
        run_tool("groupby_aggregation", df, {"group_by": "made_up", "metric": "revenue"})
    except ValueError as exc:
        assert "group_by" in str(exc)
    else:
        raise AssertionError("unknown column was accepted")
    try:
        run_tool("execute_python", df)
    except ValueError as exc:
        assert "not registered" in str(exc)
    else:
        raise AssertionError("unregistered tool was accepted")


def test_langgraph_workflow_plans_validates_and_reports():
    result = run_analysis(frame(), "What is the revenue trend over time and by region?")
    assert result["report"]["dataset_overview"]["rows"] == 6
    assert result["report"]["analysis_performed"]
    nodes = [event["node"] for event in result["trace"]]
    assert "planning" in nodes and "analysis" in nodes and "validation" in nodes and "reporting" in nodes
    assert result["report"]["validation"]["passed"]


def test_planner_never_injects_unregistered_tool():
    from app.agent.workflow import _validate_plan
    profile = profile_dataset(frame()).as_dict()
    plan = _validate_plan({"steps": [{"tool": "shell", "params": {}}, {"tool": "groupby_aggregation",
                          "params": {"group_by": "region", "metric": "revenue"}}]}, profile)
    assert [step["tool"] for step in plan["steps"]] == ["groupby_aggregation"]


def test_validation_routes_to_a_complementary_analysis_when_needed():
    from app.agent.workflow import validation_node, route_after_validation
    state = {
        "results": [{"tool": "trend_analysis", "data": {"metric": "revenue"}}],
        "profile": profile_dataset(frame()).as_dict(),
        "iteration": 1,
        "max_iterations": 3,
        "trace": [],
    }
    update = validation_node(state)
    assert update["validation"]["needs_more_analysis"] is True
    assert route_after_validation({**state, **update}) == "analysis"
