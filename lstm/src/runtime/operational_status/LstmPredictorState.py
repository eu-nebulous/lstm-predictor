# Copyright (c) 2023 Institute of Communication and Computer Systems
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.        

import threading, logging

from influxdb_client import InfluxDBClient
from jproperties import Properties

class LstmPredictorState:

    """
    The name of the lstm
    """
    forecaster_name = "lstm"
    """
    A dictionary containing statistics on the application state of individual applications
    """
    individual_application_state = {}
    """
    Fail-safe default values introduced below
    """
    application_name_prefix = "nebulous_"
    GENERAL_TOPIC_PREFIX = "eu.nebulouscloud."
    MONITORING_DATA_PREFIX = "monitoring."
    FORECASTING_CONTROL_PREFIX = "forecasting."

    #Used to create the dataset from the InfluxDB
    influxdb_organization = "my-org"
    influxdb_organization_id = "e0033247dcca0c54"
    influxdb_token = "my-super-secret-auth-token"
    influxdb_username = "my-user"
    influxdb_port = 8086
    influxdb_hostname = "localhost"
    path_to_datasets = "./datasets"
    number_of_days_to_use_data_from = 365


    configuration_file_location="/app/lstm/prediction_configuration.properties"
    configuration_details = Properties()
    prediction_processing_time_safety_margin_seconds = 20
    disconnected = True
    disconnection_handler = threading.Condition()
    testing_prediction_functionality = False
    total_time_intervals_to_predict = 8

    #Connection details
    subscribing_connector = None
    publishing_connector = None
    broker_publishers = []
    broker_consumers = []
    connector = None
    broker_address = "localhost"
    broker_port = 5672
    broker_username = "admin"
    broker_password = "nebulous"


    @staticmethod
    #TODO inspect State.connection
    def check_stale_connection():
        return (not LstmPredictorState.subscribing_connector)


