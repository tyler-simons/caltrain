import requests
import pandas as pd
import streamlit as st
import pytz
import datetime
from streamlit_extras.badges import badge
from bs4 import BeautifulSoup


def to_time(seconds):
    delta = datetime.timedelta(seconds=seconds)
    return (datetime.datetime.utcfromtimestamp(0) + delta).strftime("%H:%M")


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


def build_caltrain_df(stopname):
    tz = pytz.timezone("US/Pacific")

    # read in the station list and get the matching urlname
    stops_df = pd.read_csv("stop_ids.csv")

    # Get the urlname for the chosen station
    chosen_station_urlname = stops_df[stops_df["stopname"] == stopname]["urlname"].tolist()[0]

    curr_timestamp = datetime.datetime.utcnow().replace(tzinfo=tz).strftime("%s")
    curr_timestamp = int(curr_timestamp) * 1000
    # ping_url = f"https://www.caltrain.com/files/rt/vehiclepositions/CT.json?time={curr_timestamp}"
    ping_url = f"https://www.caltrain.com/gtfs/stops/{chosen_station_urlname}/predictions"
    real_time_trains = requests.get(ping_url).json()

    # Assuming `json_data` is the JSON object you provided
    json_data = real_time_trains

    # Initialize a list to collect data
    data = []

    # Loop through the 'data' part of the JSON
    # Loop through the 'data' part of the JSON
    for entry in json_data["data"]:
        # Each 'entry' corresponds to a 'stop' and its 'predictions'
        stop_predictions = entry.get("predictions", [])
        for prediction in stop_predictions:
            trip_update = prediction.get("TripUpdate", {})
            trip = trip_update.get("Trip", {})
            stop_time_updates = trip_update.get("StopTimeUpdate", [])

            for stop_time_update in stop_time_updates:
                # Extract the required information
                train_number = trip.get("TripId")
                route_id = trip.get("RouteId")
                stop_id = stop_time_update.get("StopId")

                # Convert the arrival and departure timestamps to human-readable format
                arrival_timestamp = stop_time_update.get("Arrival", {}).get("Time")
                departure_timestamp = stop_time_update.get("Departure", {}).get("Time")

                eta = (
                    datetime.datetime.fromtimestamp(arrival_timestamp, tz).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                    if arrival_timestamp
                    else None
                )
                departure = (
                    datetime.datetime.fromtimestamp(departure_timestamp, tz).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                    if departure_timestamp
                    else None
                )

                # Get the route type from the 'meta' part using the route_id
                route_info = json_data["meta"]["routes"].get(route_id, {})
                train_type = next(
                    (item.get("value") for item in route_info.get("title", [])), "Unknown"
                )

                # Append the collected data to the list
                data.append(
                    {
                        "Train Number": train_number,
                        "Train Type": train_type,
                        "ETA": eta,
                        "Departure": departure,
                        "Route ID": route_id,
                        "Stop ID": stop_id,
                    }
                )
    # Create a DataFrame from the collected data
    df = pd.DataFrame(data)

    if df.empty:
        return pd.DataFrame()

    lt = (
        df["ETA"]
        .apply(
            lambda x: (
                datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S").replace(tzinfo=tz)
                - datetime.datetime.now(tz)
            )
        )
        .to_list()
    )
    lt = [i.total_seconds() for i in lt]
    # Change to hours:minutes
    lt = [str(datetime.timedelta(seconds=i))[0:4] for i in lt]
    # Remove negatives
    lt = [i if i[0] != "-" else "-" for i in lt]
    df["departs_in"] = lt
    # If the stopID is even, the train is northbound, otherwise it is southbound -- make this a string
    df["direction"] = df["Stop ID"].apply(lambda x: "SB" if int(x) % 2 == 0 else "NB")

    return df


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
    # try:
    ct_df = build_caltrain_df(station)
    # except:
    # return False
    if ct_df.empty:
        return pd.DataFrame(columns=["Train #", "Direction", "Departure Time", "ETA"])

    # Move num_stops to the end
    ct_df = ct_df[["Train Number", "direction", "Departure", "departs_in"]]
    # Change column names to TN, Dir, Dep
    ct_df.columns = [
        "Train #",
        "Direction",
        "Departure Time",
        "ETA",
    ]

    # Clean up the dataframe
    ct_df.dropna(inplace=True)
    ct_df = ct_df[ct_df["ETA"] != "-"]

    deps = [
        datetime.datetime.strptime(i, "%Y-%m-%d %H:%M:%S").strftime("%I:%M %p")
        for i in ct_df["Departure Time"].tolist()
    ]
    ct_df["Departure Time"] = deps

    nb_sched = get_schedule("northbound", station, destination, rows_return=100)
    sb_sched = get_schedule("southbound", station, destination, rows_return=100)

    # st.write(nb_sched, sb_sched, ct_df)

    # Merge the scheduled and real time dataframes
    sched = pd.concat([nb_sched, sb_sched])
    sched["ETA_sched"] = sched["ETA"]
    sched = sched[["Train #", "ETA_sched"]]
    merged = ct_df.merge(sched, how="inner", on=["Train #"], suffixes=("_test", "_sched"))

    # Calculate the difference between the scheduled and real time
    # st.write(merged)
    merged["diff"] = [
        datetime.datetime.strptime(i, "%H:%M") - datetime.datetime.strptime(j, "%H:%M")
        for i, j in zip(merged["ETA"], merged["ETA_sched"])
        if i != "-" and j != "-"
    ]
    merged["total_seconds"] = [i.total_seconds() for i in merged["diff"]]
    # merged["diff"] = merged["diff"].astype(int)

    # Change the minutes to a time
    merged["diff"] = [to_time(i) for i in merged["total_seconds"].tolist()]
    # st.write(merged)

    return ct_df


def get_schedule(datadirection, chosen_station, chosen_destination=None, rows_return=5):
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
    df["ETA"] = [i + datetime.timedelta(days=1) if i.hour < 4 else i for i in df["ETA"].tolist()]

    # Sort by the ETA
    df.sort_values(by="ETA", inplace=True)

    # Calculate the time difference between the scheduled departure and the current time
    diffs = [i - now for i in df["ETA"].tolist()]

    # 0 if a diff is negative
    time_diffs = [i if i.total_seconds() > 0 else datetime.timedelta(0) for i in diffs]
    time_diffs = [i.total_seconds() for i in time_diffs]

    # Convert the time difference to a string like 00:00
    time_diffs = [to_time(i) for i in time_diffs]

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

    return df.head(rows_return)
