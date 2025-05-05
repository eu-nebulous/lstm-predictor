import datetime
import logging, os
import json
from influxdb_client import InfluxDBClient
from runtime.operational_status.LstmPredictorState import LstmPredictorState


class Utilities:

    @staticmethod
    def print_with_time(x):
        now = datetime.datetime.now()
        print("[" + now.strftime('%Y-%m-%d %H:%M:%S') + "] " + str(x))

    @staticmethod
    def get_config_value(env_var, config_key):
        """
        Returns the value of an environment variable if it is set;
        otherwise returns the value from the configuration file.
        """
        env_val = os.getenv(env_var)
        if env_val is not None:
            logging.info(f"Overriding {config_key} with environment variable {env_var}: {env_val}")
            return env_val
        else:
            return LstmPredictorState.configuration_details.get(config_key).data

    @staticmethod
    def load_configuration():
        # First load configuration details from the properties file.
        with open(LstmPredictorState.configuration_file_location, 'rb') as config_file:
            LstmPredictorState.configuration_details.load(config_file)

        # Now, override each value with an environment variable if available.
        LstmPredictorState.number_of_days_to_use_data_from = int(
            Utilities.get_config_value("NUMBER_OF_DAYS_TO_USE_DATA_FROM", "number_of_days_to_use_data_from")
        )

        LstmPredictorState.prediction_processing_time_safety_margin_seconds = int(
            Utilities.get_config_value("PREDICTION_PROCESSING_TIME_SAFETY_MARGIN_SECONDS", "prediction_processing_time_safety_margin_seconds")
        )

        LstmPredictorState.testing_prediction_functionality = Utilities.get_config_value(
            "TESTING_PREDICTION_FUNCTIONALITY", "testing_prediction_functionality"
        ).lower() == "true"

        LstmPredictorState.path_to_datasets = Utilities.get_config_value(
            "PATH_TO_DATASETS", "path_to_datasets"
        )

        LstmPredictorState.broker_address = Utilities.get_config_value(
            "BROKER_ADDRESS", "broker_address"
        )

        LstmPredictorState.broker_port = int(
            Utilities.get_config_value("BROKER_PORT", "broker_port")
        )

        LstmPredictorState.broker_username = Utilities.get_config_value(
            "BROKER_USERNAME", "broker_username"
        )

        LstmPredictorState.broker_password = Utilities.get_config_value(
            "BROKER_PASSWORD", "broker_password"
        )

        LstmPredictorState.influxdb_hostname = Utilities.get_config_value(
            "INFLUXDB_HOSTNAME", "INFLUXDB_HOSTNAME"
        )

        LstmPredictorState.influxdb_port = int(
            Utilities.get_config_value("INFLUXDB_PORT", "INFLUXDB_PORT")
        )

        LstmPredictorState.influxdb_username = Utilities.get_config_value(
            "INFLUXDB_USERNAME", "INFLUXDB_USERNAME"
        )

        LstmPredictorState.influxdb_token = Utilities.get_config_value(
            "INFLUXDB_TOKEN", "INFLUXDB_TOKEN"
        )

        LstmPredictorState.influxdb_org = Utilities.get_config_value(
            "INFLUXDB_ORG", "INFLUXDB_ORG"
        )

        # Log the effective configuration.
        Utilities.print_with_time("The configuration effective currently is the following\n" +
                                    Utilities.get_fields_and_values(LstmPredictorState))

    @staticmethod
    def update_influxdb_organization_id():
        client = InfluxDBClient(
            url="http://" + LstmPredictorState.influxdb_hostname + ":" + str(LstmPredictorState.influxdb_port),
            token=LstmPredictorState.influxdb_token
        )
        org_api = client.organizations_api()
        # List all organizations.
        organizations = org_api.find_organizations()

        # Find the organization by name and set its ID.
        for org in organizations:
            if org.name == LstmPredictorState.influxdb_organization:
                logging.info(f"Organization Name: {org.name}, ID: {org.id}")
                LstmPredictorState.influxdb_organization_id = org.id
                break

    @staticmethod
    def fix_path_ending(path):
        if path[-1] == os.sep:
            return path
        else:
            return path + os.sep

    @staticmethod
    def default_to_string(obj):
        return str(obj)

    @classmethod
    def get_fields_and_values(cls, obj):
        # Returns a JSON string with object fields (ignoring those starting with __) and their values.
        fields_values = {key: value for key, value in obj.__dict__.items() if not key.startswith("__")}
        return json.dumps(fields_values, indent=4, default=cls.default_to_string)
