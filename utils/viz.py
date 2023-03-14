import pandas as pd
import plotly.express as px


def load_world_map_fatalities(df, color, projection, scope, color_scale, dict_countries, max_fat):
    df = df.iloc[:, 1:]
    grouped = df.groupby(["iso3", "yearmon"]).sum().reset_index().sort_values(['yearmon'], ascending=False)
    grouped = grouped.rename(columns={"iso3": "ISO_A3"})
    grouped["CountryName"] = grouped["ISO_A3"].map(lambda x: dict_countries[x.lower()])
    fig = px.choropleth(
        grouped[::-1],
        color=color,
        featureidkey="properties.ISO_A3",
        locations="ISO_A3",
        hover_name="CountryName",
        color_continuous_scale= color_scale,
        width=800,
        height=600,
        projection=projection,
        scope=scope,
        range_color=[0, max_fat],
        hover_data={'yearmon':False,
                    'predicted_fatalities': ':.2f',
                    'ISO_A3': False},
        animation_frame='yearmon',
        labels={
            'predicted_fatalities': 'Predicted Fatalities',
            'yearmon': 'Year & Month'
        }
    )
    return fig


def load_world_map_TopFlop(df_temp, projection, scope, color):
    fig = px.choropleth(df_temp,
                        color=color,
                        featureidkey="properties.ISO_A3",
                        locations="ISO_A3",
                        hover_name="CountryName",
                        color_discrete_sequence=["green", "red"],
                        width=800,
                        height=550,
                        projection=projection,
                        scope=scope,
                        hover_data={'DATE': False,
                                    'OCoDi': False,
                                    'ISO_A3': False,
                                    'Highest_Lowest': False},
                        labels={
                            'Highest_Lowest': 'Highest and lowest OCoDi Scores',
                            'highest': 'Highest OCoDi Scores',
                            'lowest': 'Lowest OCoDi Scores'
                        }
                        )
    return fig

