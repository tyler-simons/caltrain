import requests
import pandas as pd
import streamlit as st
import pytz
import datetime
from streamlit_extras.badges import badge

st.set_page_config(page_title="Caltrain Timetable", page_icon="🛤", layout="centered")


def create_train_df(train):
    # Create a dataframe for the train where each stop has arrival and departure times
    stops_df = pd.json_normalize(train["TripUpdate"]["StopTimeUpdate"])
    # If Arrival.Time is not in the columns, return None
    if "Arrival.Time" not in stops_df.columns:
        return None
    stops_df["train_num"] = train["TripUpdate"]["Trip"]["TripId"]
    stops_df["direction"] = train["TripUpdate"]["Trip"]["DirectionId"]
    # Fill in missing Arrival.Time values with Departure.Time
    stops_df["Arrival.Time"] = stops_df["Arrival.Time"].fillna(stops_df["Departure.Time"])

    # Convert the arrival and departure times to datetime objects with pacfic timezone in the format strftime ("%-I:%M:%S %p")
    tz = pytz.timezone("US/Pacific")
    stops_df["arrival_time"] = stops_df["Arrival.Time"].apply(
        lambda x: datetime.datetime.fromtimestamp(x, tz).strftime("%I:%M %p")
    )
    stops_df["departure_time"] = stops_df["Departure.Time"].apply(
        lambda x: datetime.datetime.fromtimestamp(x, tz).strftime("%I:%M %p")
    )
    # drop where Arrival.Time is null
    # Drop the arrival and departure times in seconds
    stops_df.drop(["Arrival.Time", "Departure.Time"], axis=1, inplace=True)
    return stops_df


def build_caltrain_df():
    tz = pytz.timezone("US/Pacific")
    curr_timestamp = datetime.datetime.utcnow().replace(tzinfo=tz).strftime("%s")
    curr_timestamp = int(curr_timestamp) * 1000
    ping_url = f"https://www.caltrain.com/files/rt/tripupdates/CT.json?time={curr_timestamp}"
    real_time_trains = requests.get(ping_url).json()

    all_trains = []
    for train in real_time_trains["Entities"]:
        all_trains.append(create_train_df(train))
    all_trains_df = pd.concat(all_trains)
    return all_trains_df


def ping_caltrain(station):
    ct_df = build_caltrain_df()

    # Read in stops_ids from CSV file
    stops_df = pd.read_csv("stop_ids.csv")
    # Create two dictionaries, one for stop1 to stopname and one for stop2 to stopname
    stop1_to_stopname = dict(zip(stops_df["stop1"], stops_df["stopname"]))
    stop2_to_stopname = dict(zip(stops_df["stop2"], stops_df["stopname"]))

    # Combine the two dictionaries into one
    stopmap = {**stop1_to_stopname, **stop2_to_stopname}

    # Map the stop ids to the stop names
    ct_df["StopId"] = ct_df["StopId"].astype("int").map(stopmap)

    # Map direction 0 to Northbound and 1 to southbound
    ct_df["direction"] = ct_df["direction"].map({0: "NB", 1: "SB"})

    # Get the number of stops from the first train in the dataframe until the station
    ct_df["num_stops"] = ct_df.groupby("train_num")["arrival_time"].rank(method="first", ascending=True) - 1
    ct_df_first_train = ct_df.groupby("train_num").head(1)
    ct_df_first_train = ct_df_first_train[["train_num", "StopId", "departure_time", "num_stops"]]
    ct_df = ct_df.query(f'StopId == "{station}"').sort_values(["direction", "departure_time"])
    ct_df = ct_df.merge(ct_df_first_train, on="train_num", how="left", suffixes=("", "_now"))

    # Drop StopId and arrival_time columns
    ct_df.drop(["StopId", "arrival_time", "num_stops_now"], axis=1, inplace=True)

    # Move num_stops to the end
    ct_df = ct_df[["train_num", "direction", "departure_time", "StopId_now", "departure_time_now", "num_stops"]]

    # Change column names to TN, Dir, Dep
    ct_df.columns = [
        "Train #",
        "Direction",
        "Scheduled Departure",
        "Current Location",
        "Current Location Time",
        "Stops Away",
    ]

    # Calculate the time difference between the scheduled departure and the current stop arrival
    arrs = [datetime.datetime.strptime(i, "%I:%M %p") for i in ct_df["Scheduled Departure"].tolist()]
    now = datetime.datetime.now()

    # Calculate the time difference between the scheduled departure and the current time
    diffs = [datetime.datetime.combine(datetime.date.today(), i.time()) - now for i in arrs]

    # 0 if a diff is negative
    time_diffs = [i if i.total_seconds() > 0 else datetime.timedelta(0) for i in diffs]
    time_diffs = [str(i)[0:4] for i in time_diffs]

    # Add the time difference to the dataframe
    ct_df["Time Difference"] = time_diffs

    # Convert Stops Away to int
    ct_df["Stops Away"] = ct_df["Stops Away"].astype("int")

    # If scheduled departure is equal to current stop arrival, set stops away to 0
    ct_df.loc[ct_df["Scheduled Departure"] == ct_df["Current Location Time"], "Stops Away"] = 0

    # Return the dataframe neatly formatted as a string
    ct_df = ct_df[["Train #", "Scheduled Departure", "Time Difference", "Stops Away", "Direction", "Current Location"]]
    ct_df.columns = [
        "Train #",
        "Scheduled Arrival",
        "ETA",
        "Stops Left",
        "Direction",
        "Current Location",
    ]

    return ct_df


st.title("🚊 Caltrain Real Times 🛤")
caltrain_stations = pd.read_csv("stop_ids.csv")
col1, _ = st.columns([2, 1])

col1.markdown(
    """
    This app pulls data from the [Caltrain Live Map](https://www.caltrain.com/schedules/faqs/real-time-station-list) and displays the next scheduled trains for a given station. The data is refreshed on load.

    """
)


chosen_station = col1.selectbox("Choose Station", caltrain_stations["stopname"], index=13)

caltrain_data = ping_caltrain(chosen_station)

# Reorder the columns


# Split the caltrain data based on direction and drop the direction column
caltrain_data_nb = caltrain_data.query("Direction == 'NB'").drop("Direction", axis=1)
caltrain_data_sb = caltrain_data.query("Direction == 'SB'").drop("Direction", axis=1)

# Reset the index to 1, 2, 3.
caltrain_data_nb.index = caltrain_data_nb.index + 1
caltrain_data_sb.index = caltrain_data_sb.index + 1


# Display the dataframes split by Train #, Scheduled Departure, Current Stop and the other columns
st.subheader("Northbound")
st.dataframe(caltrain_data_nb)

st.subheader("Southbound")
st.dataframe(caltrain_data_sb)

col1, col2 = st.columns([8, 4])
with col1:
    st.markdown("---")

col1, col2 = st.columns([3, 3])
with col1:
    badge("twitter", "TYLERSlMONS", "https://twitter.com/TYLERSlMONS")
with col2:
    badge("github", "tyler-simons/caltrain", "https://github.com/tyler-simons/caltrain")
