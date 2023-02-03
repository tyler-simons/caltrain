# Caltrain Real Time

A project that allows you to see where the Caltrain is based on the Real Time API that is used on the main Caltrain website. The data is formatted to show ETA, current location, and number of stops for the train to get to you.

## Usage

The app is a Streamlit app and is currently hosted [here](caltrain.streamlit.app). To run it locally, follow the instructions below.

## Installation

To get started with the project, you will need to clone this repository and install the requirements listed in the `requirements.txt` file in the base directory folder. Here's how to do it:

1. Clone the repository: `git clone https://github.com/tyler-simons/caltrain.git`
2. Install the requirements: `pip install -r requirements.txt`

Once you have installed the requirements, you can play around with the script locally.
`streamlit run stcaltrain.py`

# CaltrainText

The `caltrain_response` folder has a set up to create a Google Cloud Function that performs a similar role to the Streamlit app.

You can also deploy the function like this once you've set up your google cloud tools. See the file `deploy_caltrain_check.sh` for an example of what your deploy script could look like. You'll need to configure the options for yourself. This will deploy the function to Google Cloud, and you will be able to access it via the HTTP trigger URL provided by Google Cloud, which can trigger the Twilio function.

## Contributing

We welcome additions from the community. Please create an issue or make a PR to add something.

## License

MIT
