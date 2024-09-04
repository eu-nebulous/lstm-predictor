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

    #Forecaster variables

    def get_prediction_data_filename(self,configuration_file_location,metric_name):
        from jproperties import Properties
        p = Properties()
        with open(configuration_file_location, "rb") as f:
            p.load(f, "utf-8")
            path_to_datasets, metadata = p["path_to_datasets"]
            #application_name, metadata = p["application_name"]
            path_to_datasets = Utilities.fix_path_ending(path_to_datasets)
            return "" + str(path_to_datasets) + str(self.application_name) + "_"+metric_name+ ".csv"
    def __init__(self,application_name, message_version):
        self.message_version = message_version
        self.application_name = application_name
        self.influxdb_bucket = LstmPredictorState.application_name_prefix+application_name+"_bucket"
        token = LstmPredictorState.influxdb_token

        list_bucket_url = 'http://' + LstmPredictorState.influxdb_hostname + ':8086/api/v2/buckets?name='+self.influxdb_bucket
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
                    'everySeconds': 2592000 #30 days (30*24*3600)
                }
            ]
        }

        response = requests.get(list_bucket_url, headers=headers)

        logging.info("The response for listing a possibly existing bucket is "+str(response.status_code)+" for application "+application_name)
        logging.info("Response Jan: " + str(response.json()))
        if ((response.status_code==200) and ("buckets" in response.json()) and (len(response.json()["buckets"])>0)):
                logging.info("The bucket already existed for the particular application, skipping its creation...")
        else:
            logging.info("The response in the request to list a bucket is "+str(response.json()))
            logging.info("The bucket did not exist for the particular application, creation in process...")
            response = requests.post(create_bucket_url, headers=headers, data=json.dumps(data))
            logging.info("The response for creating a new bucket is "+str(response.status_code))
        self.start_forecasting = False  # Whether the component should start (or keep on) forecasting
        self.prediction_data_filename = application_name+".csv"
        self.dataset_file_name = "lstm_dataset_"+application_name+".csv"
        self.metrics_to_predict = []
        self.epoch_start = 0
        self.next_prediction_time = 0
        self.prediction_horizon = 120
        self.previous_prediction = None
        self.initial_metric_list_received = False
        self.lower_bound_value = {}
        self.upper_bound_value = {}


    def update_monitoring_data(self):
        #query(metrics_to_predict,number_of_days_for_which_data_was_retrieved)
        #save_new_file()
        Utilities.print_with_time("Starting dataset creation process...")

        try:
            for metric_name in self.metrics_to_predict:
                time_interval_to_get_data_for = str(LstmPredictorState.number_of_days_to_use_data_from) + "d"
                print_data_from_db = True
                query_string = 'from (bucket: "'+self.influxdb_bucket+'")  |> range(start:-'+time_interval_to_get_data_for+')  |> filter(fn: (r) => r["_measurement"] == "'+metric_name+'")'
                influx_connector = InfluxDBConnector()
                logging.info("performing query for application with bucket "+str(self.influxdb_bucket))
                logging.info("The body of the query is "+query_string)
                logging.info("The configuration of the client is "+Utilities.get_fields_and_values(influx_connector))
                current_time = time.time()
                result = influx_connector.client.query_api().query(query_string, LstmPredictorState.influxdb_organization)
                elapsed_time = time.time()-current_time
                prediction_dataset_filename = self.get_prediction_data_filename(LstmPredictorState.configuration_file_location, metric_name)
                if len(result)>0:
                    logging.info(f"Performed query to the database, it took "+str(elapsed_time) + f" seconds to receive {len(result[0].records)} entries (from the first and possibly only table returned). Now logging to {prediction_dataset_filename}")
                else:
                    logging.info("No records returned from database")
                #print(result.to_values())
                with open(prediction_dataset_filename, 'w') as file:
                    for table in result:
                        #print header row
                        file.write("Timestamp,ems_time,"+metric_name+"\r\n")
                        for record in table.records:
                            dt = parser.isoparse(str(record.get_time()))
                            epoch_time = int(dt.timestamp())
                            metric_value = record.get_value()
                            if(print_data_from_db):
                                file.write(str(epoch_time)+","+str(epoch_time)+","+str(metric_value)+"\r\n")
                                # Write the string data to the file



        except Exception as e:
            Utilities.print_with_time("Could not create new dataset as an exception was thrown")
            print(traceback.format_exc())