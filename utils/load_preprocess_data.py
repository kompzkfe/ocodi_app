import streamlit as st
import pandas as pd
import numpy as np
import json

# Auxiliary functions for streamlit frontend
@st.cache_data
def load_polarity_country_list(temp_path):
    return pd.read_csv(temp_path)


@st.cache_data
def load_full_dataset(temp_path):
    return pd.read_csv(temp_path)


@st.cache_data
def load_countries(temp_path):
    return pd.read_csv(temp_path)



@st.cache_data
def create_high_low_list(path_full_scaled, dict_countries, year=2022, n=5):
    df = pd.read_csv(path_full_scaled, header=0, usecols=['iso3', "yearmon", "OCoDi"])
    df = df.rename(columns={"iso3": "ISO_A3"})
    df['DATE'] = pd.to_datetime(df['yearmon'])
    df = df.set_index('DATE')
    mean_df = df.groupby([df.index.year, 'ISO_A3'])['OCoDi'].mean()
    mean_df = mean_df.reset_index()
    year_df = mean_df[mean_df['DATE'] == year].sort_values('OCoDi')
    top_bottom_df = pd.concat([year_df.head(n), year_df.tail(n)])
    top_bottom_df["CountryName"] = top_bottom_df["ISO_A3"].map(lambda x: dict_countries[x.lower()])
    rank = top_bottom_df['OCoDi'].rank(method='dense', ascending=False)
    top_bottom_df['Highest_Lowest'] = rank.apply(lambda x: 'highest' if x <= n else 'lowest')
    return top_bottom_df


@st.cache_data
def preprocess_score_table(temp_path):
    df_score = pd.read_csv(temp_path)
    df_score["DATE"] = pd.to_datetime(df_score["yearmon"])
    df = df_score[["iso3", "year", "DATE", "lnFatalities", "OCoDi", "HGI4", "Vader", "Wordscores",
                   "Wordfish", "ConfliBERT", "CAMEO"]]
    return df


@st.cache_data
def preprocess_country_table(temp_path, list_scores, country, start_date, end_date):
    df_country = pd.read_csv(temp_path)
    df_country["DATE"] = pd.to_datetime(df_country["yearmon"])
    df_country_sel = df_country[
        (df_country["iso3"] == country) &
        (df_country["DATE"] <= end_date) &
        (df_country["DATE"] >= start_date)
        ]
    df_country_sel = df_country_sel.drop(columns=["yearmon", "year", "month"])
    df_country_sel = df_country_sel.melt(id_vars=["DATE", "iso3"],
                                         var_name="SCORE_NAME",
                                         value_name="SCORE_VALUE")
    return df_country_sel[df_country_sel["SCORE_NAME"].isin(list_scores)]


@st.cache_data
def groupby_scores_time_agg(df, agg_period="yearly"):
    if agg_period == "monthly":
        return df
    elif agg_period == "yearly":
        grouped = df.groupby(
            [
                pd.Grouper(key="iso3"),
                pd.Grouper(key="DATE", freq="Y")
            ]
        ).agg(
            lnFatalities=pd.NamedAgg(column="lnFatalities", aggfunc=np.mean),
            OCoDi=pd.NamedAgg(column="OCoDi", aggfunc=np.mean),
            HGI4=pd.NamedAgg(column="HGI4", aggfunc=np.mean),
            Vader=pd.NamedAgg(column="Vader", aggfunc=np.mean),
            Wordscores=pd.NamedAgg("Wordscores", aggfunc=np.mean),
            Wordfish=pd.NamedAgg(column="Wordfish", aggfunc=np.mean),
            ConfliBERT=pd.NamedAgg(column="ConfliBERT", aggfunc=np.mean),
            CAMEO=pd.NamedAgg("CAMEO", aggfunc="sum")
        )
        return grouped.reset_index()




