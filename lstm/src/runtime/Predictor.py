# Copyright (c) 2023 Institute of Communication and Computer Systems
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.        

import datetime
import json
import threading
import time
import os, sys
import multiprocessing
import traceback
from subprocess import PIPE, run
from exn import core

import logging
from exn import connector
from exn.core.handler import Handler
from exn.handler.connector_handler import ConnectorHandler
from lstm.lstm import predict_with_lstm
from runtime.operational_status.ApplicationState import ApplicationState
from runtime.predictions.Prediction import Prediction
from runtime.utilities.PredictionPublisher import PredictionPublisher
from runtime.utilities.Utilities import Utilities

from runtime.operational_status.LstmPredictorState import LstmPredictorState

print_with_time = Utilities.print_with_time


def sanitize_prediction_statistics(prediction_confidence_interval, prediction_value, metric_name, application_state):
    print_with_time(
        "Inside the sanitization process with an interval of  " + prediction_confidence_interval + " and a prediction of " + str(
            prediction_value))
    lower_value_prediction_confidence_interval = float(prediction_confidence_interval.split(",")[0])
    upper_value_prediction_confidence_interval = float(prediction_confidence_interval.split(",")[1])

    """if (not application_name in LstmPredictorState.individual_application_state):
        print_with_time("There is an issue with the application name"+application_name+" not existing in individual application states")
        return prediction_confidence_interval,prediction_value_produced"""

    lower_bound_value = application_state.lower_bound_value
    upper_bound_value = application_state.upper_bound_value

    print("Lower_bound_value is " + str(lower_bound_value))
    confidence_interval_modified = False
    new_prediction_confidence_interval = prediction_confidence_interval
    if (not (metric_name in lower_bound_value)) or (not (metric_name in upper_bound_value)):
        print_with_time(
            f"Lower value is unmodified - {lower_value_prediction_confidence_interval} and upper value is unmodified - {upper_value_prediction_confidence_interval}")
        return new_prediction_confidence_interval, prediction_value

    if (lower_value_prediction_confidence_interval < lower_bound_value[metric_name]):
        lower_value_prediction_confidence_interval = lower_bound_value[metric_name]
        confidence_interval_modified = True
    elif (lower_value_prediction_confidence_interval > upper_bound_value[metric_name]):
        lower_value_prediction_confidence_interval = upper_bound_value[metric_name]
        confidence_interval_modified = True
    if (upper_value_prediction_confidence_interval > upper_bound_value[metric_name]):
        upper_value_prediction_confidence_interval = upper_bound_value[metric_name]
        confidence_interval_modified = True
    elif (upper_value_prediction_confidence_interval < lower_bound_value[metric_name]):
        upper_value_prediction_confidence_interval = lower_bound_value[metric_name]
        confidence_interval_modified = True
    if confidence_interval_modified:
        new_prediction_confidence_interval = str(lower_value_prediction_confidence_interval) + "," + str(
            upper_value_prediction_confidence_interval)
        print_with_time("The confidence interval " + prediction_confidence_interval + " was modified, becoming " + str(
            new_prediction_confidence_interval) + ", taking into account the values of the metric")
    if (prediction_value < lower_bound_value[metric_name]):
        print_with_time("The prediction value of " + str(
            prediction_value) + " for metric " + metric_name + " was sanitized to " + str(lower_bound_value))
        prediction_value = lower_bound_value
    elif (prediction_value > upper_bound_value[metric_name]):
        print_with_time("The prediction value of " + str(
            prediction_value) + " for metric " + metric_name + " was sanitized to " + str(upper_bound_value))
        prediction_value = upper_bound_value

    return new_prediction_confidence_interval, prediction_value


def predict_attribute(application_state, attribute, configuration_file_location, next_prediction_time):
    global prediction_confidence_interval
    prediction_confidence_interval_produced = False
    prediction_value_produced = False
    prediction_valid = False

    # Get the prediction data filename
    application_state.prediction_data_filename = application_state.get_prediction_data_filename(
        configuration_file_location, attribute)

    # Import the necessary function from lstm.py

    if LstmPredictorState.testing_prediction_functionality:
        print_with_time(
            "Testing, so output will be based on the horizon setting from the properties file and the last timestamp in the data")
        print_with_time("Running LSTM prediction for testing.")

        # Run LSTM prediction in testing mode (without next_prediction_time)
        prediction_results = predict_with_lstm(application_state.prediction_data_filename, attribute)
    else:
        print_with_time("Running LSTM prediction with provided next prediction time.")

        # Run LSTM prediction with the next_prediction_time
        prediction_results = predict_with_lstm(application_state.prediction_data_filename, attribute,
                                               next_prediction_time=next_prediction_time)

    # Check and parse the prediction results
    if prediction_results:
        prediction_value = prediction_results.get('prediction_value', 0)

        prediction_confidence_interval = prediction_results.get('confidence_interval',
                                                                "-10000000000000000000000000,10000000000000000000000000")
        prediction_mae = prediction_results.get('mae', 0)
        prediction_mse = prediction_results.get('mse', 0)
        prediction_mape = prediction_results.get('mape', 0)
        prediction_smape = prediction_results.get('smape', 0)

    if prediction_confidence_interval is None:
        prediction_confidence_interval = "-10000000000000000000000000,10000000000000000000000000"

        if prediction_value != 0 and prediction_confidence_interval:
            prediction_confidence_interval, prediction_value = sanitize_prediction_statistics(
                prediction_confidence_interval, float(prediction_value), attribute, application_state
            )
            prediction_valid = True
            print_with_time("The prediction for attribute " + attribute + " is " + str(
                prediction_value) + " and the confidence interval is " + prediction_confidence_interval)
    else:
        logging.info("There was an error during the calculation of the predicted value for " + str(attribute))

    # Create a Prediction object to store the results
    output_prediction = Prediction(
        prediction_value,
        prediction_confidence_interval,
        prediction_valid,
        prediction_mae,
        prediction_mse,
        prediction_mape,
        prediction_smape
    )
    return output_prediction


def predict_attributes(application_state, next_prediction_time):
    attributes = application_state.metrics_to_predict
    pool = multiprocessing.Pool(len(attributes))
    print_with_time("Prediction thread pool size set to " + str(len(attributes)))
    attribute_predictions = {}

    for attribute in attributes:
        print_with_time("Starting " + attribute + " prediction thread")
        start_time = time.time()
        attribute_predictions[attribute] = pool.apply_async(predict_attribute, args=[application_state, attribute,
                                                                                     LstmPredictorState.configuration_file_location,
                                                                                     str(next_prediction_time)])
        # attribute_predictions[attribute] = pool.apply_async(predict_attribute, args=[attribute, configuration_file_location,str(next_prediction_time)]).get()

    for attribute in attributes:
        attribute_predictions[attribute] = attribute_predictions[attribute].get()  # get the results of the processing
        attribute_predictions[attribute].set_last_prediction_time_needed(int(time.time() - start_time))
        # prediction_time_needed[attribute])

    pool.close()
    pool.join()
    return attribute_predictions


def update_prediction_time(epoch_start, prediction_horizon, maximum_time_for_prediction):
    current_time = time.time()
    prediction_intervals_since_epoch = ((current_time - epoch_start) // prediction_horizon)
    estimated_time_after_prediction = current_time + maximum_time_for_prediction
    earliest_time_to_predict_at = epoch_start + (
            prediction_intervals_since_epoch + 1) * prediction_horizon  # these predictions will concern the next prediction interval

    if (estimated_time_after_prediction > earliest_time_to_predict_at):
        future_prediction_time_factor = 1 + (
                estimated_time_after_prediction - earliest_time_to_predict_at) // prediction_horizon
        prediction_time = earliest_time_to_predict_at + future_prediction_time_factor * prediction_horizon
        print_with_time(
            "Due to slowness of the prediction, skipping next time point for prediction (prediction at " + str(
                earliest_time_to_predict_at - prediction_horizon) + " for " + str(
                earliest_time_to_predict_at) + ") and targeting " + str(
                future_prediction_time_factor) + " intervals ahead (prediction at time point " + str(
                prediction_time - prediction_horizon) + " for " + str(prediction_time) + ")")
    else:
        prediction_time = earliest_time_to_predict_at + prediction_horizon
    print_with_time(
        "Time is now " + str(current_time) + " and next prediction batch starts with prediction for time " + str(
            prediction_time))
    return prediction_time


def calculate_and_publish_predictions(application_state, maximum_time_required_for_prediction):
    start_forecasting = application_state.start_forecasting

    while start_forecasting:
        print_with_time("Using " + LstmPredictorState.configuration_file_location + " for configuration details...")
        application_state.next_prediction_time = update_prediction_time(application_state.epoch_start,
                                                                        application_state.prediction_horizon,
                                                                        maximum_time_required_for_prediction)

        for attribute in application_state.metrics_to_predict:
            if ((application_state.previous_prediction is not None) and (
                    application_state.previous_prediction[attribute] is not None) and (
                    application_state.previous_prediction[
                        attribute].last_prediction_time_needed > maximum_time_required_for_prediction)):
                maximum_time_required_for_prediction = application_state.previous_prediction[
                    attribute].last_prediction_time_needed

        # Below we subtract one reconfiguration interval, as we cannot send a prediction for a time point later than one prediction_horizon interval
        wait_time = application_state.next_prediction_time - application_state.prediction_horizon - time.time()
        print_with_time("Waiting for " + str(
            (int(wait_time * 100)) / 100) + " seconds, until time " + datetime.datetime.fromtimestamp(
            application_state.next_prediction_time - application_state.prediction_horizon).strftime(
            '%Y-%m-%d %H:%M:%S'))
        if (wait_time > 0):
            time.sleep(wait_time)
            if (not start_forecasting):
                break

        Utilities.load_configuration()
        application_state.update_monitoring_data()
        first_prediction = None
        for prediction_index in range(0, LstmPredictorState.total_time_intervals_to_predict):
            prediction_time = int(
                application_state.next_prediction_time) + prediction_index * application_state.prediction_horizon
            try:
                print_with_time("Initiating predictions for all metrics for next_prediction_time, which is " + str(
                    application_state.next_prediction_time))
                prediction = predict_attributes(application_state, prediction_time)
                if (prediction_time == int(application_state.next_prediction_time)):
                    first_prediction = prediction
            except Exception as e:
                print_with_time("Could not create a prediction for some or all of the metrics for time point " + str(
                    application_state.next_prediction_time) + ", proceeding to next prediction time. However, " + str(
                    prediction_index) + " predictions were produced (out of the configured " + str(
                    LstmPredictorState.total_time_intervals_to_predict) + "). The encountered exception trace follows:")
                print(traceback.format_exc())
                # continue was here, to continue while loop, replaced by break
                break
            for attribute in application_state.metrics_to_predict:
                if (not prediction[attribute].prediction_valid):
                    # continue was here, to continue while loop, replaced by break
                    break
                if (LstmPredictorState.disconnected or LstmPredictorState.check_stale_connection()):
                    logging.info("Possible problem due to disconnection or a stale connection")
                    # State.connection.connect()
                message_not_sent = True
                current_time = int(time.time())
                prediction_message_body = {
                    "metricValue": float(prediction[attribute].value),
                    "level": 3,
                    "timestamp": current_time,
                    "probability": 0.95,
                    # This is the default second parameter of the prediction intervals (first is 80%) created as part of the HoltWinters forecasting mode in R
                    "confidence_interval": [float(prediction[attribute].lower_confidence_interval_value), float(
                        prediction[attribute].upper_confidence_interval_value)],
                    "predictionTime": prediction_time,
                }
                training_models_message_body = {
                    "metrics": application_state.metrics_to_predict,
                    "forecasting_method": "lstm",
                    "timestamp": current_time,
                }
                while (message_not_sent):
                    try:
                        # for publisher in State.broker_publishers:
                        #    if publisher.
                        for publisher in LstmPredictorState.broker_publishers:
                            # if publisher.address=="eu.nebulouscloud.monitoring.preliminary_predicted.lstm"+attribute:

                            if publisher.key == "publisher_" + attribute:
                                publisher.send(prediction_message_body)

                        # State.connection.send_to_topic('intermediate_prediction.%s.%s' % (id, attribute), prediction_message_body)

                        # State.connection.send_to_topic('training_models',training_models_message_body)
                        message_not_sent = False
                        print_with_time(
                            "Successfully sent prediction message for %s to topic eu.nebulouscloud.preliminary_predicted.%s.%s:\n\n%s\n\n" % (
                                attribute, LstmPredictorState.forecaster_name, attribute, prediction_message_body))
                    except ConnectionError as exception:
                        # State.connection.disconnect()
                        # State.connection = messaging.morphemic.Connection('admin', 'admin')
                        # State.connection.connect()
                        logging.error("Error sending intermediate prediction" + str(exception))
                        LstmPredictorState.disconnected = False

        if (first_prediction is not None):
            application_state.previous_prediction = first_prediction  # first_prediction is the first of the batch of the predictions which are produced. The size of this batch is set by the State.total_time_intervals_to_predict (currently set to 8)

        # State.number_of_days_to_use_data_from = (prediction_horizon - State.prediction_processing_time_safety_margin_seconds) / (wait_time / State.number_of_days_to_use_data_from)
        # State.number_of_days_to_use_data_from = 1 + int(
        #    (prediction_horizon - State.prediction_processing_time_safety_margin_seconds) /
        #    (wait_time / State.number_of_days_to_use_data_from)
        # )


# class Listener(messaging.listener.MorphemicListener):
class BootStrap(ConnectorHandler):
    pass


class ConsumerHandler(Handler):
    prediction_thread = None

    def ready(self, context):
        if context.has_publisher('state'):
            context.publishers['state'].starting()
            context.publishers['state'].started()
            context.publishers['state'].custom('forecasting')
            context.publishers['state'].stopping()
            context.publishers['state'].stopped()

            # context.publishers['publisher_cpu_usage'].send({
            #     'hello': 'world'
            # })

    def on_message(self, key, address, body, context, **kwargs):
        address = address.replace("topic://" + LstmPredictorState.GENERAL_TOPIC_PREFIX, "")
        if (address).startswith(LstmPredictorState.MONITORING_DATA_PREFIX):
            address = address.replace(LstmPredictorState.MONITORING_DATA_PREFIX, "", 1)

            logging.info("New monitoring data arrived at topic " + address)
            if address == 'metric_list':
                application_name = body["name"]
                message_version = body["version"]
                application_state = None
                individual_application_state = {}
                application_already_defined = application_name in LstmPredictorState.individual_application_state
                if (application_already_defined and
                        (message_version == LstmPredictorState.individual_application_state[
                            application_state].message_version)
                ):
                    individual_application_state = LstmPredictorState.individual_application_state
                    application_state = individual_application_state[application_name]

                    print_with_time("Using existing application definition for " + application_name)
                else:
                    if (application_already_defined):
                        print_with_time(
                            "Updating application " + application_name + " based on new metrics list message")
                    else:
                        print_with_time("Creating new application " + application_name)
                    application_state = ApplicationState(application_name, message_version)
                metric_list_object = body["metric_list"]
                lower_bound_value = application_state.lower_bound_value
                upper_bound_value = application_state.upper_bound_value
                for metric_object in metric_list_object:
                    lower_bound_value[metric_object["name"]] = float(metric_object["lower_bound"])
                    upper_bound_value[metric_object["name"]] = float(metric_object["upper_bound"])

                    application_state.lower_bound_value.update(lower_bound_value)
                    application_state.upper_bound_value.update(upper_bound_value)

                application_state.initial_metric_list_received = True

                individual_application_state[application_name] = application_state
                LstmPredictorState.individual_application_state.update(individual_application_state)
                # body = json.loads(body)
                # for element in body:
                #    State.metrics_to_predict.append(element["metric"])


        elif (address).startswith(LstmPredictorState.FORECASTING_CONTROL_PREFIX):
            address = address.replace(LstmPredictorState.FORECASTING_CONTROL_PREFIX, "", 1)
            logging.info("The address is " + address)

            if address == 'test.lstm':
                LstmPredictorState.testing_prediction_functionality = True

            elif address == 'start_forecasting.lstm':
                try:
                    application_name = body["name"]
                    message_version = 0
                    if (not "version" in body):
                        logging.info(
                            "There was an issue in finding the message version in the body of the start forecasting message, assuming it is 1")
                        message_version = 1
                    else:
                        message_version = body["version"]
                    if (application_name in LstmPredictorState.individual_application_state) and (
                            message_version <= LstmPredictorState.individual_application_state[
                        application_name].message_version):
                        application_state = LstmPredictorState.individual_application_state[application_name]
                    else:
                        LstmPredictorState.individual_application_state[application_name] = ApplicationState(
                            application_name, message_version)
                        application_state = LstmPredictorState.individual_application_state[application_name]

                    if (not application_state.start_forecasting) or (
                            (application_state.metrics_to_predict is not None) and (
                            len(application_state.metrics_to_predict) <= len(body["metrics"]))):
                        application_state.metrics_to_predict = body["metrics"]
                        print_with_time("Received request to start predicting the following metrics: " + ",".join(
                            application_state.metrics_to_predict) + " for application " + application_name + ", proceeding with the prediction process")
                    else:
                        application_state.metrics_to_predict = body["metrics"]
                        print_with_time("Received request to start predicting the following metrics: " + body[
                            "metrics"] + " for application " + application_name + "but it was perceived as a duplicate")
                        return
                    application_state.broker_publishers = []
                    for metric in application_state.metrics_to_predict:
                        LstmPredictorState.broker_publishers.append(PredictionPublisher(application_name, metric))
                    LstmPredictorState.publishing_connector = connector.EXN(
                        'publishing_' + LstmPredictorState.forecaster_name + '-' + application_name,
                        handler=BootStrap(),
                        # consumers=list(State.broker_consumers),
                        consumers=[],
                        publishers=LstmPredictorState.broker_publishers,
                        url=LstmPredictorState.broker_address,
                        port=LstmPredictorState.broker_port,
                        username=LstmPredictorState.broker_username,
                        password=LstmPredictorState.broker_password
                    )
                    # LstmPredictorState.publishing_connector.start()
                    thread = threading.Thread(target=LstmPredictorState.publishing_connector.start, args=())
                    thread.start()

                except Exception as e:
                    print_with_time(
                        "Could not load json object to process the start forecasting message \n" + str(body))
                    print(traceback.format_exc())
                    return

                # if (not State.initial_metric_list_received):
                #    print_with_time("The initial metric list has not been received,
                # therefore no predictions are generated")
                #    return

                try:
                    application_state = LstmPredictorState.individual_application_state[application_name]
                    application_state.start_forecasting = True
                    application_state.epoch_start = body["epoch_start"]
                    application_state.prediction_horizon = int(body["prediction_horizon"])
                    application_state.next_prediction_time = update_prediction_time(application_state.epoch_start,
                                                                                    application_state.prediction_horizon,
                                                                                    LstmPredictorState.prediction_processing_time_safety_margin_seconds)  # State.next_prediction_time was assigned the value of State.epoch_start here, but this re-initializes targeted prediction times after each start_forecasting message, which is not desired necessarily
                    print_with_time(
                        "A start_forecasting message has been received, epoch start and prediction horizon are " + str(
                            application_state.epoch_start) + ", and " + str(
                            application_state.prediction_horizon) + " seconds respectively")
                except Exception as e:
                    print_with_time("Problem while retrieving epoch start and/or prediction_horizon")
                    print(traceback.format_exc())
                    return

                with open(LstmPredictorState.configuration_file_location, "r+b") as f:

                    LstmPredictorState.configuration_details.load(f, "utf-8")

                    # Do stuff with the p object...
                    initial_seconds_aggregation_value, metadata = LstmPredictorState.configuration_details[
                        "number_of_seconds_to_aggregate_on"]
                    initial_seconds_aggregation_value = int(initial_seconds_aggregation_value)

                    if (application_state.prediction_horizon < initial_seconds_aggregation_value):
                        print_with_time("Changing number_of_seconds_to_aggregate_on to " + str(
                            application_state.prediction_horizon) + " from its initial value " + str(
                            initial_seconds_aggregation_value))
                        LstmPredictorState.configuration_details["number_of_seconds_to_aggregate_on"] = str(
                            application_state.prediction_horizon)

                    f.seek(0)
                    f.truncate(0)
                    LstmPredictorState.configuration_details.store(f, encoding="utf-8")

                maximum_time_required_for_prediction = LstmPredictorState.prediction_processing_time_safety_margin_seconds  # initialization, assuming X seconds processing time to derive a first prediction
                if ((self.prediction_thread is None) or (not self.prediction_thread.is_alive())):
                    self.prediction_thread = threading.Thread(target=calculate_and_publish_predictions,
                                                              args=[application_state,
                                                                    maximum_time_required_for_prediction])
                    self.prediction_thread.start()

                # waitfor(first period)

            elif address == 'stop_forecasting.lstm':
                # waitfor(first period)
                application_name = body["name"]
                application_state = LstmPredictorState.individual_application_state[application_name]
                print_with_time("Received message to stop predicting some of the metrics")
                metrics_to_remove = json.loads(body)["metrics"]
                for metric in metrics_to_remove:
                    if (application_state.metrics_to_predict.__contains__(metric)):
                        print_with_time("Stopping generating predictions for metric " + metric)
                        application_state.metrics_to_predict.remove(metric)
                if len(application_state.metrics_to_predict) == 0:
                    LstmPredictorState.individual_application_state[application_name].start_forecasting = False
                    self.prediction_thread.join()

            else:
                print_with_time(
                    "The address was " + address + " and did not match metrics_to_predict/test.lstm/start_forecasting.lstm/stop_forecasting.lstm")
                #        logging.info(f"Received {key} => {address}")
        else:
            print_with_time("Received message " + body + " but could not handle it")


def get_dataset_file(attribute):
    pass


def main():
    # Ensure the configuration file location is provided as a command-line argument
    if len(sys.argv) < 2:
        print("Error: Configuration file location must be provided as an argument.")
        sys.exit(1)

    # Set the configuration file location from the command-line argument
    configuration_file_location = sys.argv[1]
    LstmPredictorState.configuration_file_location = configuration_file_location

    # Print the current directory contents for debugging
    print(os.listdir("."))


# Change to the appropriate directory for invoking the forecasting script
os.chdir("/app/lstm")  # Update the path to match the Docker container structure

# Load configurations
Utilities.load_configuration()
Utilities.update_influxdb_organization_id()
# Subscribe to retrieve the metrics which should be used


id = "lstm"
LstmPredictorState.disconnected = True

# while(True):
#    State.connection = messaging.morphemic.Connection('admin', 'admin')
#    State.connection.connect()
#    State.connection.set_listener(id, Listener())
#    State.connection.topic("test","helloid")
#    State.connection.send_to_topic("test","HELLO!!!")
# exit(100)

while True:
    topics_to_subscribe = ["eu.nebulouscloud.monitoring.metric_list", "eu.nebulouscloud.monitoring.realtime.>",
                           "eu.nebulouscloud.forecasting.start_forecasting.lstm",
                           "eu.nebulouscloud.forecasting.stop_forecasting.lstm"]
    current_consumers = []

    for topic in topics_to_subscribe:
        current_consumer = core.consumer.Consumer(key='monitoring_' + topic, address=topic, handler=ConsumerHandler(),
                                                  topic=True, fqdn=True)
        LstmPredictorState.broker_consumers.append(current_consumer)
        current_consumers.append(current_consumer)
    LstmPredictorState.subscribing_connector = connector.EXN(LstmPredictorState.forecaster_name, handler=BootStrap(),
                                                             # consumers=list(State.broker_consumers),
                                                             consumers=LstmPredictorState.broker_consumers,
                                                             url=LstmPredictorState.broker_address,
                                                             port=LstmPredictorState.broker_port,
                                                             username=LstmPredictorState.broker_username,
                                                             password=LstmPredictorState.broker_password
                                                             )

    # connector.start()
    thread = threading.Thread(target=LstmPredictorState.subscribing_connector.start, args=())
    thread.start()
    LstmPredictorState.disconnected = False;

    print_with_time("Checking (EMS) broker connectivity state, possibly ready to start")
    if (LstmPredictorState.disconnected or LstmPredictorState.check_stale_connection()):
        try:
            # State.connection.disconnect() #required to avoid the already connected exception
            # State.connection.connect()
            LstmPredictorState.disconnected = True
            print_with_time("Possible problem in the connection")
        except Exception as e:
            print_with_time("Encountered exception while trying to connect to broker")
            print(traceback.format_exc())
            LstmPredictorState.disconnected = True
            time.sleep(5)
            continue
    LstmPredictorState.disconnection_handler.acquire()
    LstmPredictorState.disconnection_handler.wait()
    LstmPredictorState.disconnection_handler.release()

# State.connector.stop()

if __name__ == "__main__":
    main()
