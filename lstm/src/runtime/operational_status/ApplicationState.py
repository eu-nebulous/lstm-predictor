import logging
import time
import traceback

import requests
import json

from runtime.operational_status.LstmPredictorState import LstmPredictorState
from runtime.utilities.InfluxDBConnector import InfluxDBConnector
from runtime.utilities.Utilities import Utilities
from dateutil import parser


class ApplicationState:

    # Forecaster variables

    def get_prediction_data_filename(self, configuration_file_location, metric_name):
        from jproperties import Properties
        p = Properties()
        with open(configuration_file_location, "rb") as f:
            p.load(f, "utf-8")
            path_to_datasets, metadata = p["path_to_datasets"]
            # application_name, metadata = p["application_name"]
            path_to_datasets = Utilities.fix_path_ending(path_to_datasets)
            return "" + str(path_to_datasets) + str(self.application_name) + "_" + metric_name + ".csv"

    def __init__(self, application_name, message_version):
        self.prediction_thread = None
        self.message_version = message_version
        self.application_name = application_name
        self.influxdb_bucket = LstmPredictorState.application_name_prefix + application_name + "_bucket"
        token = LstmPredictorState.influxdb_token

        list_bucket_url = 'http://' + LstmPredictorState.influxdb_hostname + ':8086/api/v2/buckets?name=' + self.influxdb_bucket
        create_bucket_url = 'http://' + LstmPredictorState.influxdb_hostname + ':8086/api/v2/buckets'
        headers = {
            'Authorization': 'Token {}'.format(token),
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        data = {
            'name': self.influxdb_bucket,
            'orgID': LstmPredictorState.influxdb_organization_id,
            'retentionRules': [
                {
                    'type': 'expire',
                    'everySeconds': 2592000  # 30 days (30*24*3600)
                }
            ]
        }

        response = requests.get(list_bucket_url, headers=headers)

        logging.info("The response for listing a possibly existing bucket is " + str(
            response.status_code) + " for application " + application_name)
        logging.info("Response Jan: " + str(response.json()))
        if ((response.status_code == 200) and ("buckets" in response.json()) and (len(response.json()["buckets"]) > 0)):
            logging.info("The bucket already existed for the particular application, skipping its creation...")
        else:
            logging.info("The response in the request to list a bucket is " + str(response.json()))
            logging.info("The bucket did not exist for the particular application, creation in process...")
            response = requests.post(create_bucket_url, headers=headers, data=json.dumps(data))
            logging.info("The response for creating a new bucket is " + str(response.status_code))
        self.start_forecasting = False  # Whether the component should start (or keep on) forecasting
        self.prediction_data_filename = application_name + ".csv"
        self.dataset_file_name = "lstm_dataset_" + application_name + ".csv"
        self.metrics_to_predict = []
        self.epoch_start = 0
        self.next_prediction_time = 0
        self.prediction_horizon = 120
        self.previous_prediction = None
        self.initial_metric_list_received = False
        self.lower_bound_value = {}
        self.upper_bound_value = {}


    def update_monitoring_data(self):
        Utilities.print_with_time("Starting dataset creation process...")
        logging.debug("Entered update_monitoring_data method.")

        try:
            if not self.metrics_to_predict:
                logging.warning("No metrics to predict. Exiting update_monitoring_data.")
                return

            logging.info(f"Metrics to predict: {self.metrics_to_predict}")

            for metric_index, metric_name in enumerate(self.metrics_to_predict, start=1):
                logging.debug(f"Processing metric {metric_index}/{len(self.metrics_to_predict)}: {metric_name}")

                time_interval_to_get_data_for = f"{LstmPredictorState.number_of_days_to_use_data_from}d"
                logging.debug(f"Time interval for data retrieval: {time_interval_to_get_data_for}")

                query_string = (
                    f'from(bucket: "{self.influxdb_bucket}") '
                    f'|> range(start: -{time_interval_to_get_data_for}) '
                    f'|> filter(fn: (r) => r["_measurement"] == "{metric_name}")'
                )
                logging.debug(f"Constructed InfluxDB query: {query_string}")

                influx_connector = InfluxDBConnector()
                logging.info(
                    f"Performing query for application '{self.application_name}' with bucket '{self.influxdb_bucket}'.")

                logging.debug(f"InfluxDB client configuration: {Utilities.get_fields_and_values(influx_connector)}")

                current_time = time.time()
                logging.debug("Executing InfluxDB query.")
                result = influx_connector.client.query_api().query(query_string, LstmPredictorState.influxdb_organization)
                elapsed_time = time.time() - current_time
                logging.info(f"Query executed in {elapsed_time:.2f} seconds.")

                prediction_dataset_filename = self.get_prediction_data_filename(
                    LstmPredictorState.configuration_file_location, metric_name
                )
                logging.debug(f"Prediction dataset filename: {prediction_dataset_filename}")

                if result:
                    total_records = sum(len(table.records) for table in result)
                    logging.info(f"Received {total_records} records for metric '{metric_name}'.")
                    logging.debug(f"Result details: {result}")
                else:
                    logging.warning(f"No records returned for metric '{metric_name}'.")

                # Define print_data_from_db before using it
                print_data_from_db = True  # You can adjust this flag as needed
                logging.debug(f"print_data_from_db is set to {print_data_from_db}")

                try:
                    with open(prediction_dataset_filename, 'w') as file:
                        # Write header only once per metric
                        file.write("Timestamp,ems_time," + metric_name + "\r\n")
                        logging.debug(f"Header written to {prediction_dataset_filename}.")

                        for table in result:
                            for record in table.records:
                                dt = parser.isoparse(str(record.get_time()))
                                epoch_time = int(dt.timestamp())
                                metric_value = record.get_value()

                                if print_data_from_db:
                                    file.write(f"{epoch_time},{epoch_time},{metric_value}\r\n")
                                    logging.debug(
                                        f"Wrote record to {prediction_dataset_filename}: {epoch_time}, {metric_value}")

                    logging.info(f"Successfully wrote data to {prediction_dataset_filename} for metric '{metric_name}'.")
                except IOError as io_err:
                    logging.error(f"IOError while writing to {prediction_dataset_filename}: {io_err}")
                except Exception as file_err:
                    logging.error(f"Unexpected error while writing to {prediction_dataset_filename}: {file_err}")

        except requests.exceptions.RequestException as req_err:
            logging.error(f"RequestException during InfluxDB interaction: {req_err}")
            Utilities.print_with_time("Network-related error occurred while accessing InfluxDB.")
        except json.JSONDecodeError as json_err:
            logging.error(f"JSONDecodeError: Failed to parse JSON response: {json_err}")
        except Exception as e:
            Utilities.print_with_time("Could not create new dataset as an exception was thrown")
            logging.error("An unexpected error occurred in update_monitoring_data.", exc_info=True)
            print(traceback.format_exc())

        finally:
            logging.debug("Exiting update_monitoring_data method.")
            Utilities.print_with_time("Dataset creation process completed.")
