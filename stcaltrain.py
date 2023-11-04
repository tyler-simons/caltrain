import requests
import pandas as pd
import streamlit as st
import pytz
import datetime
from streamlit_extras.badges import badge
from bs4 import BeautifulSoup
from functions.ct_functions import (
    ping_caltrain,
    get_schedule,
    assign_train_type,
    is_northbound,
    build_caltrain_df,
    create_train_df,
)

st.set_page_config(page_title="Caltrain Platform", page_icon="🚆", layout="centered")


st.title("🚊 Caltrain Platform 🚂")
caltrain_stations = pd.read_csv("stop_ids.csv")
col1, col2 = st.columns([2, 1])

col1.markdown(
    """
    Track when the next trains leave from your station and where they are right now. Choose a destination to filter for trains that stop there.
    """
)

col1, col2 = st.columns([2, 1])
chosen_station = col1.selectbox("Choose Origin Station", caltrain_stations["stopname"], index=10)
chosen_destination = col1.selectbox(
    "Choose Destination Station", ["--"] + caltrain_stations["stopname"].tolist(), index=0
)
caltrain_data = ping_caltrain(chosen_station, destination=chosen_destination)
api_working = True if type(caltrain_data) == pd.DataFrame else False

# Allow switch between live data and scheduled data
if api_working:
    display = col1.radio(
        "Show trains",
        ["Live", "Scheduled"],
        horizontal=True,
        help="Live shows only trains that have already left the station",
    )
    schedule_chosen = True
else:
    display = col1.radio(
        "Show trains",
        ["Live", "Scheduled"],
        horizontal=True,
        help="Live shows only trains that have already left the station",
        index=1,
        disabled=True,
    )
    schedule_chosen = False

if display == "Scheduled":
    col1, col2 = st.columns([2, 1])
    if schedule_chosen:
        col1.info("📆 Pulling the current schedule from the Caltrain website...")
    else:
        col1.error(
            "❌ Caltrain Live Map API is currently down. Pulling the current schedule from the Caltrain website instead..."
        )
    # If the chosen destination is before the chosen station, then the direction is southbound
    if chosen_destination != "--" and chosen_destination != chosen_station:
        if is_northbound(chosen_station, chosen_destination):
            caltrain_data = get_schedule("northbound", chosen_station, chosen_destination)
        else:
            caltrain_data = get_schedule("southbound", chosen_station, chosen_destination)

    else:
        caltrain_data = pd.concat(
            [
                get_schedule("northbound", chosen_station, chosen_destination),
                get_schedule("southbound", chosen_station, chosen_destination),
            ]
        )

else:
    col1.success("✅ Caltrain Live Map API is up and running.")
st.write(caltrain_data)
caltrain_data["Train Type"] = caltrain_data["Train #"].apply(lambda x: assign_train_type(x))

# Localize to Pacific Time
caltrain_data["Departure Time"] = (
    pd.to_datetime(caltrain_data["Departure Time"])
    .dt.tz_localize("UTC")
    .dt.tz_convert("US/Pacific")
    .dt.strftime("%I:%M %p")
)

# Split the caltrain data based on direction and drop the direction column
caltrain_data_nb = caltrain_data.query("Direction == 'NB'").drop("Direction", axis=1)
caltrain_data_sb = (
    caltrain_data.query("Direction == 'SB'").drop("Direction", axis=1).reset_index(drop=True)
)

# Reset the index to 1, 2, 3.
caltrain_data_nb.index = caltrain_data_nb.index + 1
caltrain_data_sb.index = caltrain_data_sb.index + 1

col1, col2 = st.columns([2, 1])

# Display the dataframes split by Train #, Scheduled Departure, Current Stop and the other columns
col1.subheader("Northbound Trains")
nb_data = caltrain_data_nb.T
nb_data.columns = nb_data.iloc[0]
nb_data = nb_data.drop(nb_data.index[0])
col1.dataframe(nb_data, use_container_width=True)

col1.subheader("Southbound Trains")
sb_data = caltrain_data_sb.T
sb_data.columns = sb_data.iloc[0]
sb_data = sb_data.drop(sb_data.index[0])
col1.dataframe(sb_data, use_container_width=True)

if col1.button("Refresh Data"):
    st.experimental_rerun()

# Definitions
col1.markdown("---")
col1.subheader("Definitions")
col1.markdown(
    """
1. **Train Number** - The train ID. The first digit indicates the train type.
2. **Departure Time** - The scheduled departure time from the **Origin** station.
3. **ETA** - The estimated number of hours and minutes to the **Origin** station.
4. **Train Type** - Local trains make all stops. Limited and Bullet make fewer.
"""
)

col1.subheader("About")
col1.markdown(
    """
- This app pulls _real-time_ data from the [Caltrain Live Map](https://www.caltrain.com/schedules/faqs/real-time-station-list). It was created to solve the issue of arriving at the Caltrain station while the train is behind schedule. This app will tell you when the next train is leaving, and about how long it will take to arrive at the station. 

- **Note:** If the Caltrain Live Map API is down, then the app will pull the current schedule from the Caltrain website instead.
"""
)

col1, col2 = st.columns([2, 1])
with col1:
    st.markdown("---")

col1, col2 = st.columns([1, 1])
with col1:
    badge("twitter", "TYLERSlMONS", "https://twitter.com/TYLERSlMONS")
with col2:
    badge("github", "tyler-simons/caltrain", "https://github.com/tyler-simons/caltrain")
