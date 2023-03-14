from datetime import datetime

import pandas as pd
import streamlit as st
import plotly.express as px
import numpy as np
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_extras.metric_cards import style_metric_cards

from data.lists import country_list, dict_scores, dict_scope
from utils.general import get_img_with_href
from utils.load_preprocess_data import load_countries, preprocess_score_table, groupby_scores_time_agg, preprocess_country_table, create_high_low_list
from utils.viz import load_world_map_TopFlop, load_world_map_fatalities

st.set_page_config(
    page_title="OCoDi",
    page_icon="chart_with_upwards_trend",
    layout="wide",
)

# Paths to data sources
path_to_full_scaled = "data/full_scaled.csv"
path_to_countries_data = "data/countries.csv"
path_to_predictions = "data/predictions.csv"
path_to_logo = "data/images/kompzkfe_logo.png"

# Loading the data from sources

df_countries = load_countries(path_to_countries_data)
dict_countries = dict(zip(df_countries.alpha3, df_countries.name))
df_high_low = create_high_low_list(path_to_full_scaled, dict_countries, year=2022)
df_highest = df_high_low[df_high_low["Highest_Lowest"]=='highest'].sort_values(by="OCoDi", ascending=False)
df_lowest = df_high_low[df_high_low["Highest_Lowest"]=='lowest']
df_predictions = pd.read_csv(path_to_predictions)
max_fat = df_predictions["predicted_fatalities"].max()


# Front-end part of the app
sub = st.container()

with sub:
    st.title("An Interpretable Deep Learning Approach to Domain-Specific Dictionary Creation: A Use Case for Conflict Prediction")
    st.markdown("**Contributors: <a href='https://www.unibw.de/ciss-en/kompz-kfe/team/sonja-haffner-m-sc' style='text-decoration: none; color: black; font-weight: bold;'>Sonja Häffner</a>, <a href='https://www.unibw.de/ciss-en/kompz-kfe/team/dr-rer-nat-martin-hofer' style='text-decoration: none; color: black; font-weight: bold;'>Martin Hofer</a>, <a href='https://www.uni-regensburg.de/wirtschaftswissenschaften/bwl-roesch/team/maximilian-nagl/index.html' style='text-decoration: none; color: black; font-weight: bold;'>Maximilian Nagl</a>, and <a href='https://www.unibw.de/ciss-en/kompz-kfe/team/julian-walterskirchen-m-sc' style='text-decoration: none; color: black; font-weight: bold;'>Julian Walterskirchen</a>**", unsafe_allow_html=True)
    st.write(
        "This website serves as a companion to our paper published in [Political Analysis](https://www.cambridge.org/core/journals/political-analysis). It allows the user to explore different aspects of our **Objective Conflict Dictionary (OCoDi)**. For more details read the paper here: [https://doi.org/10.1017/pan.2023.7](https://doi.org/10.1017/pan.2023.7). The code and data for the paper can be found here: https://doi.org/10.7910/DVN/Y5INRM. We would like to thank Marje Kaack for supporting the development of this web application.")
    st.subheader("Abstract")
    st.write("Recent advancements in natural language processing (NLP) methods have significantly improved their performance. However, more complex NLP models are more difficult to interpret and computationally expensive. Therefore, we propose an approach to dictionary creation that carefully balances the trade-off between complexity and interpretability. This approach combines a deep neural network architecture with techniques to improve model explainability to automatically build a domain-specific dictionary. As an illustrative use case of our approach, we create an objective dictionary that can infer conflict intensity from text data. We train the neural networks on a corpus of conflict reports and match them with conflict event data. This corpus consists of over 14,000 expert-written International Crisis Group (ICG) CrisisWatch reports between 2003 and 2021. Sensitivity analysis is used to extract the weighted words from the neural network to build the dictionary. In order to evaluate our approach, we compare our results to state-of-the-art deep learning language models, text-scaling methods, as well as standard, non-specialized, and conflict event dictionary approaches. We are able to show that our approach outperforms other approaches while retaining interpretability.")


# Navigation in sidebar to jump to the different topics
with st.sidebar:
    img_html = get_img_with_href(path_to_logo, 'https://www.unibw.de/ciss-en/kompz-kfe/')
    st.markdown(img_html, unsafe_allow_html=True)
    add_vertical_space(3)
    st.title("Sections")
    st.markdown(f'''
    <a href={'#trends-in-conflict-intensity'}><button style="background-color:White; border: 2px solid black; width: 100%; border-radius:8px; font-site:20px ">Trends in Conflict Intensity</button></a>
    ''', unsafe_allow_html=True)
    st.markdown(f'''
        <a href={'#predicting-fatalities'}><button style="background-color:White; border: 2px solid black; width: 100%; border-radius:8px; font-site:20px ">Predicting Fatalities</button></a>
        ''', unsafe_allow_html=True)
    st.markdown(f'''
        <a href={'#time-series-analysis'}><button style="background-color:White; border: 2px solid black; width: 100%; border-radius:8px; font-site:20px ">Time Series Analysis</button></a>
        ''', unsafe_allow_html=True)
    st.markdown(f'''
            <a href={'#nlp-methods'}><button style="background-color:White; border: 2px solid black; width: 100%; border-radius:8px; font-site:20px ">NLP Methods</button></a>
            ''', unsafe_allow_html=True)

# WorldMap of TopFlop countries colored

# Conflict trends
st.subheader("Trends in Conflict Intensity")
st.write("The map below shows the countries with the highest and lowest OCoDi Scores for 2022. These Scores give an indication of how conflictual the reporting by [CrisisWatch](https://www.crisisgroup.org/crisiswatch) was in 2022. In general, higher values are associated with higher levels of fatalities. The highest and lowest OCoDi Scores for 2022 can be explored in the below world map or by clicking on the expandable *List of Trends*.")
fig_world_map_top_flop = load_world_map_TopFlop(df_high_low,
                                                "natural earth",
                                                "world",
                                                "Highest_Lowest")

st.plotly_chart(fig_world_map_top_flop, use_container_width=True)

# Highest/Lowest OCoDi score country comparison
with st.expander(label="**List of Trends**"):
    st.write("Countries with the highest and lowest OCoDi scores averaged over 2022.")
    add_vertical_space(1)
    st.write(
        """
        <style>
        [data-testid="stMetricDelta"] svg {
            display: none;
        }
        div[data-testid="stMetric"]
        </style>
        """,
        unsafe_allow_html=True,
    )
    # highest and lowest 5 OCoDi Scores in 2022
    highest, lowest = st.columns(2)
    with highest:
        st.markdown('<p style="color:red; font-size:30px"> Most conflictual reporting', unsafe_allow_html=True)
        for i, row in df_highest.iterrows():
            st.metric(label = "", value=row["CountryName"], delta='{:.4f}'.format(row['OCoDi']), delta_color="inverse")
    with lowest:
        st.markdown('<p style="color:green; font-size:30px"> Least conflictual reporting', unsafe_allow_html=True)
        for i, row in df_lowest.iterrows():
            st.metric(label="", value=row["CountryName"], delta='{:.4f}'.format(row['OCoDi']), delta_color="inverse")
    style_metric_cards()
add_vertical_space(2)
# Predicting Fatalities Component
st.subheader("Predicting Fatalities")
st.write("In our paper we also assess how well our dictionary can infer conflict related fatalities from documents not used in the training of our dictionary. The dictionary was trained on reports between 2003 and 2020 and was then applied to reports published in 2021 and 2022. The scores obtained for these reports were then used as features in XGBoost models to predict fatalities in 2021 and 2022. These predictions are not true forecasts of the future, but can be viewed as a text regression task. In the world map below one can investigate how much conflict related fatalities our model predicted for each month in 2021 and 2022.")
sel_scope = st.selectbox(
    "Select a region",
    options=list(dict_scope.keys()),
    format_func=lambda x: dict_scope[x]
)


# Creation of map of fatality predictions
fig_world_map_fat = load_world_map_fatalities(df_predictions,
                                              "predicted_fatalities",
                                              "natural earth",
                                              sel_scope,
                                              "reds",
                                              dict_countries,
                                              round(max_fat))
st.plotly_chart(fig_world_map_fat, use_container_width=True)

add_vertical_space(5)
#
st.subheader("Time Series Analysis")
st.write("To get a more detailed understanding of how well OCoDi captures conflict intensity (log fatalities) in reports, one can compare different countries over time (Country Comparison) or compare scores from different natural language processing (NLP) methods for one country (Score Comparison). More details on the different NLP methods can be found below.")
timeseries = st.radio("Time Series Comparison",
                      ("Country Comparison", "Score Comparison"),
                      index=0,
                      horizontal=True,
                      label_visibility="collapsed")
st.subheader(timeseries)
if timeseries == "Country Comparison":
    st.write("Select **countries** to compare, a **score** to analyse and a **time period**.")
    c1, c2 = st.columns([1, 1])
    with c1:
        countryOption = st.multiselect("Select a country",
                                       country_list,
                                       format_func=lambda x: dict_countries[x.lower()],
                                       default=["NGA"])
    with c2:
        sel_scores = st.selectbox("Select a Score",
                                  options=list(dict_scores.keys()),
                                  format_func=lambda x: dict_scores[x],
                                  index=0
                                  )
    emp_c,c3, c4 = st.columns([0.13, 5, 1])
    with c3:
        timePeriod = st.slider("Select a time period",
                               value=(datetime(2003, 8, 31), datetime(2021, 12, 31)),
                               format="MM-YYYY"
                               )
    with c4:
        sel_agg_period = st.radio("Select a aggregation period",
                                  options=("monthly", "yearly"),
                                  horizontal=True)

    df_scores_agg = groupby_scores_time_agg(preprocess_score_table(path_to_full_scaled), sel_agg_period)
    df_scores_sel = df_scores_agg[
        (df_scores_agg["iso3"].isin(countryOption)) &
        (df_scores_agg["DATE"] <= timePeriod[1]) &
        (df_scores_agg["DATE"] >= timePeriod[0])
        ]
    fig = px.line(df_scores_sel,
                  x="DATE",
                  y=sel_scores,
                  color="iso3",
                  hover_data={
                      "iso3":False,
                      "lnFatalities":":.4f"
                  },
                  labels={
                      "iso3": "ISO-Country Code",
                      "DATE": "Date"
                  })
    st.plotly_chart(fig, use_container_width=True)



if timeseries == "Score Comparison":
    st.write("Select **scores** to compare, a **country** to analyse and a **time period**.")
    col_sel_cnty, col_scores_cnty = st.columns([1, 1])
    with col_sel_cnty:
        countryOption_country = st.selectbox("Select a country",
                                             country_list,
                                             format_func=lambda x: dict_countries[x.lower()],
                                             index=0)
    with col_scores_cnty:
        sel_scores_country = st.multiselect("Select Scores to compare",
                                            options=list(dict_scores.keys()),
                                            format_func=lambda x: dict_scores[x],
                                            default=["OCoDi"]
                                            )
    emp_col,col_timeperiod_cnty, col_aggperiod_cnty = st.columns([0.13, 5, 1])
    with col_timeperiod_cnty:
        timePeriod_country = st.slider("Select a time period",
                                       value=(datetime(2003, 8, 31), datetime(2021, 12, 31)),
                                       format="MM-YYYY"
                                       )
    with col_aggperiod_cnty:
        sel_agg_period_country = st.radio("Select a aggregation period",
                                          options=("monthly", "yearly"),
                                          horizontal=True)

    df_country = preprocess_country_table(path_to_full_scaled,
                                          sel_scores_country,
                                          countryOption_country,
                                          timePeriod_country[0],
                                          timePeriod_country[1])

    if sel_agg_period_country == "yearly":
        df_country = df_country.groupby([
            pd.Grouper(key="SCORE_NAME"),
            pd.Grouper(key="DATE", freq="Y")]).agg(
            SCORE_VALUE=pd.NamedAgg(column="SCORE_VALUE", aggfunc=np.mean), ).reset_index()

    fig = px.line(df_country,
                  x="DATE",
                  y="SCORE_VALUE",
                  color="SCORE_NAME",
                  hover_data={"SCORE_NAME": False,
                              "SCORE_VALUE": ":.4f"},
                  labels={
                      "SCORE_NAME": "Score Name",
                      "SCORE_VALUE": "Score Value",
                      "DATE": "Date"
                  })
    st.plotly_chart(fig, use_container_width=True)


st.subheader("NLP Methods")
st.write("In order to compare the performance of our OCoDi score to other common NLP methods, we also calculate alternative (sentiment) scores employing the following methods: First, we calculate sentiment scores for each document based on two popular sentiment dictionaries: The [Harvard IV-4 dictionary](https://pypi.org/project/pysentiment2/) and [Valence Aware Dictionary and sEntiment Reasoner - VADER ](https://github.com/cjhutto/vaderSentiment)"
         ". We also analyze our text data with the [PETRARCH2](https://github.com/openeventdata/petrarch2) system that employs the conflict-specific CAMEO and TABARI event extraction dictionaries and use the CAMEO conflict-cooperation scale to assign scores to each text. Next, we rely on two different document scaling techniques ([Wordscores](https://www.tcd.ie/Political_Science/wordscores/index.html) and [Wordfish](http://www.wordfish.org/)) "
         "to infer relative document positions from our evaluation corpus. We also fine-tune a [ConfliBERT](https://github.com/eventdata/ConfliBERT) model on CrisisWatch reports and then directly predict fatalities for the test data. All scores as described above are calculated at the country-month level and matched with monthly aggregated fatalities from the [UCDP GED database](https://ucdp.uu.se/).  ")