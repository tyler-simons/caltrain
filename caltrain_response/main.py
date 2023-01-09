import logging
import requests
from twilio.rest import Client
import os
import pandas as pd
import yaml
import pytz
import datetime


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

    # Filter for the desired station and for the first row of each train
    ct_df_first_train = ct_df.groupby("train_num").head(1)
    ct_df = ct_df.query(f'StopId == "{station}"').sort_values(["direction", "departure_time"])
    ct_df = ct_df.merge(
        ct_df_first_train[["train_num", "StopId", "departure_time"]], on="train_num", how="left", suffixes=("", "_now")
    )

    # Drop StopId and arrival_time columns
    ct_df.drop(["StopId", "arrival_time"], axis=1, inplace=True)

    # Map direction 0 to Northbound and 1 to southbound
    ct_df["direction"] = ct_df["direction"].map({0: "NB", 1: "SB"})

    # Change column names to TN, Dir, Dep
    ct_df.columns = ["#", "Dir", "Dep", "Cur", "Dep"]

    # Return the dataframe neatly formatted as a string
    return format_df_as_text(ct_df)


def format_df_as_text(df):
    # Fill in any missing values with empty strings
    df = df.fillna("")
    # Find the maximum length of each column
    max_lengths = df.apply(lambda x: x.str.len()).max()

    # Format each row as a string
    rows = []
    for row in df.itertuples(index=False):
        # Format the first three columns
        formatted_row = [f"{v:<{max_lengths[i]}}" for i, v in enumerate(row[:3])]
        # Format the fourth and fifth columns as a single line, wrapped in parentheses
        formatted_row.append(f"\n({row[3]} {row[4]})")
        rows.append(" ".join(formatted_row))

    # Join the rows with newline characters
    return "\n\n".join(rows)


def send_twilio_message(message_body: str, account_sid: str, auth_token: str, from_number: str, to_number: str):
    """Send a text message to a phone number from a twilio account
    Args:
        message_body (str): Text to show up in the message
        account_sid (str): Twilio account SID
        auth_token (str): Twilio account auth token
        from_number (str): Twilio provided phone number
        to_number (str): Phone number to send the message to

    Returns:
        str: Message sent success
    """
    client = Client(account_sid, auth_token)
    message = client.messages.create(body=message_body, from_=from_number, to=to_number)

    return f"Message Sent: {message.sid}"


def main(request):
    """Scrape Global Entry for interviews
    Args:
        event (dict): Event payload.
        context (google.cloud.functions.Context): Metadata for the event.
    """
    print("in main function, checking for caltrain")

    # Read in config variables from config.yaml
    ACCOUNT_SID = os.environ["ACCOUNT_SID"]
    AUTH_TOKEN = os.environ["AUTH_TOKEN"]
    FROM_NUMBER = os.environ["FROM_NUMBER"]

    # Figure out where the request info lives
    print(request.values)
    station = request.values["Body"].strip()
    TO_NUMBER = request.values["From"].strip()

    # Application Default credentials are automatically created.
    station_map = {
        "rwc": "Redwood City",
        "calave": "California Ave.",
        "mp": "Menlo Park",
        "sf": "San Francisco",
        "pa": "Palo Alto",
        "hillsdale": "Hillsdale",
    }
    station = station_map.get(station, station)
    message = ping_caltrain(station)
    print(message)
    send_twilio_message(message, ACCOUNT_SID, AUTH_TOKEN, FROM_NUMBER, TO_NUMBER)
    return "OK"
