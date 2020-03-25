import plotly.graph_objects as go
import plotly


def data_ww(ggg):
    """The data_ww function will form graph objects for visualization

    Args:
        ggg (ggg): the DataFrame for visualization


    Returns:
        figures: plotly figure objects for visualization

    """
    # calcutae the counts of genre
    genre_counts = ggg.groupby("genre").count()["message"]

    # create plotly object
    bar1 = []
    bar1.append(go.Bar(x=list(genre_counts.index), y=genre_counts))

    # create plotly layout
    barlayout1 = dict(
        title="Distribution of Message Genres",
        yaxis=dict(title="Count"),
        xaxis=dict(title="Genre"),
    )

    # calculate percentages
    categories_ = ggg.iloc[:, 2:]
    catseries = (
        categories_.sum() / categories_.shape[0] * 100
    ).sort_values(ascending=False)

    # create plotly graph object
    bar2 = []
    bar2.append(
        go.Bar(
            x=catseries.index.tolist(),
            y=[round(b, 2) for b in catseries.values.tolist()],
        )
    )

    barlayout2 = dict(
        title="Distribution of Message Categories",
        yaxis=dict(title="Percent"),
        xaxis=dict(tickangle=45, title_standoff = 205, automargin=True,title="Categories") # ,automargin=True ,title="Categories"
    )

    # calculate the percentages for  message categories
    categories_ = ggg.iloc[:, 2:]
    allcat = (
        categories_.sum() /
        categories_.shape[0] *
        100).sort_values(
        ascending=False) > 10
    topten = allcat[allcat].index.tolist()

    # calculate the percentages of genre for seven top message categories
    gg = ggg.groupby("genre")[topten].sum()
    gg2 = gg.apply(lambda x: x / x.sum(), axis=1)

    # get the list of values
    valuelist = [gg2.iloc[i, :].tolist() for i in range(len(gg2.index))]
    colindex = [gg.columns.tolist()] * len(gg2.index)

    groupbar = []

    # create plotly object for group bar
    for indx, name in enumerate(gg.index.tolist()):
        groupbar.append(
            go.Bar(
                x=colindex[indx],
                y=[round(y * 100, 2) for y in valuelist[indx]],
                name=name,
                type="bar",
                # text=[str(round(y * 100, 2)) for y in valuelist[indx]], 
                text = [f'{round(y * 100, 1)}%' for y in valuelist[indx]],
                textposition="auto",
                hoverinfo="none",
                # opacity=0.5,
                marker_line_color="rgb(8,48,107)",
                marker_line_width=1.5
                # marker_color=cmm[indx]
            )
        )

    # create the group bar layout
    groupbarlayout = dict(
        barmode="group",
        title="Percentage Distribution of High Message Category by Genre",
        yaxis=dict(showticklabels=False, ticks=" ", ticktext = " " ,showgrid=False, visible=False),# title="Percent", visible=False, ,showticklabels=False
        xaxis=dict(title="Categories"),
    )

    figures = []

    figures.append(dict(data=bar1, layout=barlayout1))
    figures.append(dict(data=bar2, layout=barlayout2))
    figures.append(dict(data=groupbar, layout=groupbarlayout))

    return figures
