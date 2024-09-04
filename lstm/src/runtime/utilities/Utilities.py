# Copyright (c) 2023 Institute of Communication and Computer Systems
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.        

#from morphemic.dataset import DatasetMaker
import datetime
import logging,os
import json
from influxdb_client import InfluxDBClient

from runtime.operational_status.LstmPredictorState import LstmPredictorState


class Utilities:

    @staticmethod
    def print_with_time(x):
        now = datetime.datetime.now()
        print("["+now.strftime('%Y-%m-%d %H:%M:%S')+"] "+str(x))

    @staticmethod
    def load_configuration():
        with open(LstmPredictorState.configuration_file_location, 'rb') as config_file:
            LstmPredictorState.configuration_details.load(config_file)
            LstmPredictorState.number_of_days_to_use_data_from = int(LstmPredictorState.configuration_details.get("number_of_days_to_use_data_from").data)
            LstmPredictorState.prediction_processing_time_safety_margin_seconds = int(LstmPredictorState.configuration_details.get("prediction_processing_time_safety_margin_seconds").data)
            LstmPredictorState.testing_prediction_functionality = LstmPredictorState.configuration_details.get("testing_prediction_functionality").data.lower() == "true"
            LstmPredictorState.path_to_datasets = LstmPredictorState.configuration_details.get("path_to_datasets").data
            LstmPredictorState.broker_address = LstmPredictorState.configuration_details.get("broker_address").data
            LstmPredictorState.broker_port = int(LstmPredictorState.configuration_details.get("broker_port").data)
            LstmPredictorState.broker_username = LstmPredictorState.configuration_details.get("broker_username").data
            LstmPredictorState.broker_password = LstmPredictorState.configuration_details.get("broker_password").data

            LstmPredictorState.influxdb_hostname = LstmPredictorState.configuration_details.get("INFLUXDB_HOSTNAME").data
            LstmPredictorState.influxdb_port = int(LstmPredictorState.configuration_details.get("INFLUXDB_PORT").data)
            LstmPredictorState.influxdb_username = LstmPredictorState.configuration_details.get("INFLUXDB_USERNAME").data
            LstmPredictorState.influxdb_password = LstmPredictorState.configuration_details.get("INFLUXDB_PASSWORD").data
            LstmPredictorState.influxdb_org = LstmPredictorState.configuration_details.get("INFLUXDB_ORG").data

        #This method accesses influx db to retrieve the most recent metric values.
            Utilities.print_with_time("The configuration effective currently is the following\n "+Utilities.get_fields_and_values(LstmPredictorState))

    @staticmethod
    def update_influxdb_organization_id():
        client = InfluxDBClient(url="http://" + LstmPredictorState.influxdb_hostname + ":" + str(LstmPredictorState.influxdb_port), token=LstmPredictorState.influxdb_token)
        org_api = client.organizations_api()
        # List all organizations
        organizations = org_api.find_organizations()

        # Find the organization by name and print its ID
        for org in organizations:
            if org.name == LstmPredictorState.influxdb_organization:
                logging.info(f"Organization Name: {org.name}, ID: {org.id}")
                LstmPredictorState.influxdb_organization_id = org.id
                break
    @staticmethod
    def fix_path_ending(path):
        if (path[-1] is os.sep):
            return path
        else:
            return path + os.sep

    @staticmethod
    def default_to_string(obj):
        return str(obj)
    @classmethod
    def get_fields_and_values(cls,object):
        #Returns those fields that do not start with __ (and their values)
        fields_values = {key: value for key, value in object.__dict__.items() if not key.startswith("__")}
        return json.dumps(fields_values,indent=4,default=cls.default_to_string)

