import requests
import pandas as pd
import streamlit as st
import pytz
import datetime
from streamlit_extras.badges import badge
from bs4 import BeautifulSoup

st.set_page_config(page_title="Caltrain Platform", page_icon="🚆", layout="centered")


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


# Add train type where locals are 100s and 200s, limited is 300s through 600s and bullets are 700s
def assign_train_type(x):
    if x.startswith("1") or x.startswith("2"):
        return "Local"
    elif x.startswith("3") or x.startswith("4") or x.startswith("5") or x.startswith("6"):
        return "Limited"
    else:
        return "Bullet"


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

def is_northbound(chosen_station, chosen_destination):
    """
    Returns True if the chosen destination is before
    the chosen station in the list of stations
    """
    stops = pd.read_csv("stop_ids.csv")
    station_index = stops[stops["stopname"] == chosen_station].index[0]
    destination_index = stops[stops["stopname"] == chosen_destination].index[0]
    return station_index > destination_index

def ping_caltrain(station, destination):
    try:
        ct_df = build_caltrain_df()
    except:
        return pd.DataFrame()

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

    # If the destination is not None, filter the dataframe to only include trains going to that destination
    keep_trains = None
    if destination not in ["--", station]:

        df_copy = ct_df.copy()

        # Select trains where both the origin and destination are in the stops_df
        if destination != "San Francisco":
            df_copy = df_copy.query(f'StopId == "{station}" or StopId == "{destination}"')

        # Group by train_num and count the number of stops
        df_copy = df_copy.groupby("train_num").filter(lambda x: len(x) > 1)
        keep_trains = df_copy["train_num"].unique()

    # Get the number of stops from the first train in the dataframe until the station
    ct_df["num_stops"] = (
        ct_df.groupby("train_num")["arrival_time"].rank(method="first", ascending=True) - 1
    )
    ct_df_first_train = ct_df.groupby("train_num").head(1)
    ct_df_first_train = ct_df_first_train[["train_num", "StopId", "departure_time", "num_stops"]]
    ct_df = ct_df.query(f'StopId == "{station}"').sort_values(["direction", "departure_time"])
    ct_df = ct_df.merge(ct_df_first_train, on="train_num", how="left", suffixes=("", "_now"))

    # Drop StopId and arrival_time columns
    ct_df.drop(["StopId", "arrival_time", "num_stops_now"], axis=1, inplace=True)

    # Move num_stops to the end
    ct_df = ct_df[
        [
            "train_num",
            "direction",
            "departure_time",
            "StopId_now",
            "departure_time_now",
            "num_stops",
        ]
    ]

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
    arrs = [
        datetime.datetime.strptime(i, "%I:%M %p") for i in ct_df["Scheduled Departure"].tolist()
    ]
    # If arrs are between 12am and 4am, add a day to them
    arrs = [i + datetime.timedelta(days=1) if i.hour < 4 else i for i in arrs]

    # Convert this row to the same as the other caltrain output
    pstz = pytz.timezone("US/Pacific")
    old_day = datetime.datetime(1900, 1, 1)
    old_day_time = datetime.datetime.now(tz=pstz).time()
    now = datetime.datetime.combine(
        old_day,
        datetime.time(old_day_time.hour, old_day_time.minute),
    )

    # Calculate the time difference between the scheduled departure and the current time
    diffs = [i - now for i in arrs]

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
    ct_df = ct_df[
        [
            "Train #",
            "Scheduled Departure",
            "Time Difference",
            "Stops Away",
            "Direction",
            "Current Location",
        ]
    ]
    ct_df.columns = [
        "Train #",
        "Departure Time",
        "ETA",
        "Stops Left",
        "Direction",
        "Current Location",
    ]
    # Sort by ETA
    ct_df = ct_df.sort_values("ETA")

    # If the index of the station > index of destination -- SB
    if chosen_destination != "--" and chosen_station != chosen_destination:
        if is_northbound(chosen_station, chosen_destination):
            ct_df = ct_df.query(f'Direction == "NB"')
        else:
            ct_df = ct_df.query(f'Direction == "SB"')

    if keep_trains is not None:
        ct_df = ct_df[ct_df["Train #"].isin(keep_trains)]

    # Sort by ETA
    ct_df = ct_df.sort_values("ETA")

    return ct_df


def get_schedule(datadirection, chosen_station, chosen_destination=None):

    if chosen_destination == "--" or chosen_station == chosen_destination:
        chosen_destination = None

    # Pull the scheduled train times from this url
    url = "https://www.caltrain.com/?active_tab=route_explorer_tab"

    # Get the html from the url
    html = requests.get(url).content

    # Parse the html
    soup = BeautifulSoup(html, "html.parser")

    # Get the table from the html
    table = soup.find(
        "table",
        attrs={
            "class": "caltrain_schedule table table-striped",
            "data-direction": datadirection,
        },
    )
    table_body = table.find("tbody")

    # Get the rows from the table
    rows = table_body.find_all("tr")

    # Get the data from the rows
    data = []
    for row in rows:
        cols = row.find_all("td")
        cols = [ele.text.strip() for ele in cols]
        data.append([ele for ele in cols if ele])

    # Convert the data to a dataframe
    df = pd.DataFrame(data)
    # Shift the first row over by 1 to the right
    first_row_vals = df.iloc[0, :][:-1]
    df.iloc[0, :] = ["Zone"] + first_row_vals.tolist()

    # Drop the first column and any nas
    df = df.drop(0, axis=1)

    # Set the first column as the index
    df.index = df[1]

    # Make the first row the column names
    new_header = df.iloc[0]
    df = df[1:]
    df.columns = new_header

    # Drop the first column
    df = df.drop(df.columns[0], axis=1)

    # Drop any columns with the value -- in the 2nd row
    if chosen_destination:
        df = df[[i in [chosen_station, chosen_destination] for i in df.index]]
        df.replace("--", pd.NA, inplace=True)
        df = df.dropna(axis=1)

        # Drop the second row
        df = df.drop(df.index[1])
    else:
        df = df[df.index == chosen_station]

    # Convert this row to the same as the other caltrain output
    pstz = pytz.timezone("US/Pacific")
    old_day = datetime.datetime(1900, 1, 1)
    old_day_time = datetime.datetime.now(tz=pstz).time()
    now = datetime.datetime.combine(
        old_day,
        datetime.time(old_day_time.hour, old_day_time.minute),
    )
    weekday = True if datetime.datetime.now(tz=pstz).weekday() < 5 else False

    # Transpose the dataframe
    df = df.T.reset_index()

    # Drop any rows with the value -- in the 2nd column
    df = df[df.iloc[:, 1] != "--"]

    df.columns = ["Train #", "Departure Time"]
    df["Direction"] = datadirection
    # Map NB to Northbound and SB to Southbound
    df["Direction"] = df["Direction"].map({"northbound": "NB", "southbound": "SB"})
    df["ETA"] = [datetime.datetime.strptime(i, "%I:%M%p") for i in df["Departure Time"].tolist()]

    # If the hour is between 12 and 4, add a day to the ETA
    df["ETA"] = [
        i + datetime.timedelta(days=1) if i.hour < 4 else i for i in df["ETA"].tolist()
    ]

    # Sort by the ETA
    df.sort_values(by="ETA", inplace=True)

    # Calculate the time difference between the scheduled departure and the current time
    diffs = [i - now for i in df["ETA"].tolist()]

    # 0 if a diff is negative
    time_diffs = [i if i.total_seconds() > 0 else datetime.timedelta(0) for i in diffs]
    time_diffs = [i.total_seconds() for i in time_diffs]

    # Convert the time difference to a string like 00:00
    time_diffs = [str(datetime.timedelta(seconds=i))[0:4] for i in time_diffs]

    # Add the time difference to the dataframe
    df["ETA"] = time_diffs

    # Drop the trains that have already left
    df = df[df["ETA"] != "0:00"]
    df.reset_index(drop=True, inplace=True)
    df.dropna(inplace=True)

    # Drop any SF stations northbound
    if chosen_station in ["San Francisco"]:
        df = df[df["Direction"] != "NB"]

    # Filter for only Train # 200s trains if weekday, otherwise exclude them
    if not weekday:
        df = df[df["Train #"].str.startswith("2")]
    else:
        df = df[~df["Train #"].str.startswith("2")]

    return df.head(5)


st.title("🚊 Caltrain Platform 🚂")
caltrain_stations = pd.read_csv("stop_ids.csv")
col1, col2 = st.columns([2, 1])

col1.markdown(
    """
    Track when the next trains leave from your station and where they are right now. Choose a destination to filter, if you like.
    """
)

col1, col2 = st.columns([2, 1])
chosen_station = col1.selectbox("Choose Origin Station", caltrain_stations["stopname"], index=13)
chosen_destination = col1.selectbox(
    "Choose Destination Station", ["--"] + caltrain_stations["stopname"].tolist(), index=0
)
caltrain_data = ping_caltrain(chosen_station, chosen_destination)
api_working = True if caltrain_data.shape[1] != 0 else False

# Allow switch between live data and scheduled data
if api_working:
    display = col1.radio("Displaying", ['Live data', 'Scheduled data'], horizontal=True, help="Live only shows trains active now, not necessarily all scheduled trains")
    schedule_chosen = True
else:
    display = col1.radio("Displaying", ['Live data', 'Scheduled data'], horizontal=True, help="Live only shows trains active now, not necessarily all scheduled trains", index=1, disabled=True)
    schedule_chosen = False

if display == 'Scheduled data':
    col1, col2 = st.columns([2, 1])
    if schedule_chosen:
        col1.info(
            "📆 Pulling the current schedule from the Caltrain website..."
        )
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

caltrain_data["Train Type"] = caltrain_data["Train #"].apply(lambda x: assign_train_type(x))


# Split the caltrain data based on direction and drop the direction column
caltrain_data_nb = caltrain_data.query("Direction == 'NB'").drop("Direction", axis=1)
caltrain_data_sb = caltrain_data.query("Direction == 'SB'").drop("Direction", axis=1).reset_index(drop=True)

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
2. **Dep. Time** - The scheduled departure time from the **Origin** station.
3. **ETA** - The estimated time of arrival to the **Origin** station.
4. **Stops Left** - The number of stops until the train arrives at the **Origin** station.
3. **Current Stop** - The closest stop to where the train is currently located.
4. **Train Type** - Local trains make all stops. Limited and Bullet make fewer.
""")

col1.subheader("About")
col1.markdown("""
- This app pulls _real-time_ data from the [Caltrain Live Map](https://www.caltrain.com/schedules/faqs/real-time-station-list). It was created to solve the issue of arriving at the Caltrain station while the train is behind schedule. This app will tell you when the next train is leaving, and about how long it will take to arrive at the station. 

- **Note:** If the Caltrain Live Map API is down, then the app will pull the current schedule from the Caltrain website instead.
"""
)


if col1.button("Refresh 🔄"):
    st.experimental_rerun()

col1, col2 = st.columns([2, 1])
with col1:
    st.markdown("---")

col1, col2 = st.columns([1, 1])
with col1:
    badge("twitter", "TYLERSlMONS", "https://twitter.com/TYLERSlMONS")
with col2:
    badge("github", "tyler-simons/caltrain", "https://github.com/tyler-simons/caltrain")
