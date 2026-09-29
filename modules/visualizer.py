from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# histogram plot
def build_histogram(df: pd.DataFrame, column: str) -> go.Figure:
    fig = px.histogram(
        df,
        x=column,
        nbins=30,
        title=f"Distribution of {column}",
        color_discrete_sequence=["#4C78A8"],
    )
    fig.update_layout(
        xaxis_title=column,
        yaxis_title="Count",
        bargap=0.05,
    )
    return fig


# scatter plot
def build_scatter(
    df: pd.DataFrame, x_column: str, y_column: str, color_column: str | None = None
) -> go.Figure:
    fig = px.scatter(
        df,
        x=x_column,
        y=y_column,
        color=color_column,
        title=f"{y_column} vs {x_column}",
        opacity=0.6,
        trendline="ols"
        if df[x_column].dtype.kind in "if" and df[y_column].dtype.kind in "if"
        else None,
    )
    fig.update_layout(xaxis_title=x_column, yaxis_title=y_column)
    return fig


# box plot
def build_box_plot(
    df: pd.DataFrame, column: str, group_by: str | None = None
) -> go.Figure:
    fig = px.box(
        df,
        y=column,
        x=group_by,
        title=f"Box Plot of {column}" + (f" by {group_by}" if group_by else ""),
        points="outliers",
    )
    fig.update_layout(yaxis_title=column)
    return fig


# correlation heatmap
def build_correlation_heatmap(correlation_matrix: pd.DataFrame) -> go.Figure:
    fig = px.imshow(
        correlation_matrix,
        text_auto=".2f",
        color_continuous_scale="RdBu",
        zmin=-1,
        zmax=1,
        title="Correlation Heatmap",
    )
    fig.update_layout(xaxis_title="", yaxis_title="")
    return fig
