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
from proton import Message
from runtime.operational_status.ApplicationState import ApplicationState
from runtime.predictions.Prediction import Prediction
from runtime.utilities.PredictionPublisher import PredictionPublisher
from runtime.utilities.Utilities import Utilities

from runtime.operational_status.LstmPredictorState import LstmPredictorState

print_with_time = Utilities.print_with_time


def sanitize_prediction_statistics(
    prediction_confidence_interval,
    prediction_value,
    metric_name,
    lower_bound_value,
    upper_bound_value
):
    print_with_time(
        "Inside the sanitization process with an interval of "
        + prediction_confidence_interval
        + " and a prediction of "
        + str(prediction_value)
    )
    lower_value_prediction_confidence_interval = float(prediction_confidence_interval.split(",")[0])
    upper_value_prediction_confidence_interval = float(prediction_confidence_interval.split(",")[1])

    confidence_interval_modified = False
    new_prediction_confidence_interval = prediction_confidence_interval

    if (lower_bound_value is None and upper_bound_value is None):
        print_with_time(
            f"Lower value is unmodified - {lower_value_prediction_confidence_interval} and "
            f"upper value is unmodified - {upper_value_prediction_confidence_interval}"
        )
        return new_prediction_confidence_interval, prediction_value

    # If there is a lower bound
    if (lower_bound_value is not None):
        if (upper_value_prediction_confidence_interval < lower_bound_value):
            upper_value_prediction_confidence_interval = lower_bound_value
            lower_value_prediction_confidence_interval = lower_bound_value
            confidence_interval_modified = True
        elif (lower_value_prediction_confidence_interval < lower_bound_value):
            lower_value_prediction_confidence_interval = lower_bound_value
            confidence_interval_modified = True

    # If there is an upper bound
    if (upper_bound_value is not None):
        if (lower_value_prediction_confidence_interval > upper_bound_value):
            lower_value_prediction_confidence_interval = upper_bound_value
            upper_value_prediction_confidence_interval = upper_bound_value
            confidence_interval_modified = True
        elif (upper_value_prediction_confidence_interval > upper_bound_value):
            upper_value_prediction_confidence_interval = upper_bound_value
            confidence_interval_modified = True

    if confidence_interval_modified:
        new_prediction_confidence_interval = (
            str(lower_value_prediction_confidence_interval)
            + ","
            + str(upper_value_prediction_confidence_interval)
        )
        print_with_time(
            "The confidence interval "
            + prediction_confidence_interval
            + " was modified, becoming "
            + str(new_prediction_confidence_interval)
            + ", taking into account the values of the metric"
        )

    # Finally, clamp the prediction_value if out of bounds
    if (lower_bound_value is not None and prediction_value < lower_bound_value):
        print_with_time(
            "The prediction value of "
            + str(prediction_value)
            + " for metric "
            + metric_name
            + " was sanitized to "
            + str(lower_bound_value)
        )
        prediction_value = lower_bound_value
    elif (upper_bound_value is not None and prediction_value > upper_bound_value):
        print_with_time(
            "The prediction value of "
            + str(prediction_value)
            + " for metric "
            + metric_name
            + " was sanitized to "
            + str(upper_bound_value)
        )
        prediction_value = upper_bound_value

    return new_prediction_confidence_interval, prediction_value


def predict_attribute(
    data_filename,
    attribute,
    lower_bound_value,
    upper_bound_value,
    next_prediction_time
):
    """
    A wrapper that runs the LSTM predictor for a single attribute,
    handles missing data, and returns a Prediction or None.
    """
    # 1) Run LSTM
    if LstmPredictorState.testing_prediction_functionality:
        print_with_time(f"Running LSTM prediction in test mode for {attribute}.")
        prediction_results = predict_with_lstm(data_filename, attribute)
    else:
        print_with_time(f"Running LSTM prediction for {attribute} with next_prediction_time.")
        prediction_results = predict_with_lstm(
            data_filename,
            attribute,
            next_prediction_time=next_prediction_time
        )

    # If predict_with_lstm() returned a None or empty result, it usually means the metric's data was missing
    if not prediction_results:
        print_with_time(
            f"No results for {attribute} because data was missing or empty. Skipping."
        )
        return None

    # 2) Construct a default confidence interval if not in result
    prediction_confidence_interval = "-10000000000000000000000000,10000000000000000000000000"

    # 3) Extract results
    prediction_value = prediction_results.get('prediction_value', None)
    prediction_mae = prediction_results.get('mae', 0)
    prediction_mse = prediction_results.get('mse', 0)
    prediction_mape = prediction_results.get('mape', 0)
    prediction_smape = prediction_results.get('smape', 0)

    # If for some reason the returned value is None, skip
    if prediction_value is None:
        print_with_time(f"Prediction value for {attribute} is None. Skipping.")
        return None

    # 4) Sanitize bounds
    new_interval, new_value = sanitize_prediction_statistics(
        prediction_confidence_interval,
        float(prediction_value),
        attribute,
        lower_bound_value,
        upper_bound_value
    )

    # 5) Build the Prediction object
    output_prediction = Prediction(
        new_value,
        new_interval,
        True,  # Mark as valid if we got this far
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
    prediction_results = {}
    attribute_predictions = {}

    # Enqueue each attribute’s prediction in parallel
    for attribute in attributes:
        print_with_time("Starting " + attribute + " prediction thread")
        start_time = time.time()

        data_filename = application_state.get_prediction_data_filename(
            LstmPredictorState.configuration_file_location, attribute
        )
        # If you stored bounds as dict keyed by metric:
        lb_val = application_state.lower_bound_value.get(attribute)
        ub_val = application_state.upper_bound_value.get(attribute)

        prediction_results[attribute] = pool.apply_async(
            predict_attribute,
            args=[data_filename, attribute, lb_val, ub_val, str(next_prediction_time)]
        )

    # Gather results
    for attribute in attributes:
        attribute_predictions[attribute] = prediction_results[attribute].get()

        # If we got a valid prediction object, store the time used
        if attribute_predictions[attribute] is not None:
            # The logic for measuring time is per-attribute:
            # That means each loop’s start_time is overwritten in the queue above.
            # Typically, you'd measure it individually, but here is fine for demonstration.
            attribute_predictions[attribute].set_last_prediction_time_needed(
                int(time.time() - start_time)
            )

    pool.close()
    pool.join()
    return attribute_predictions


def update_prediction_time(epoch_start, prediction_horizon, maximum_time_for_prediction):
    current_time = time.time()
    prediction_intervals_since_epoch = ((current_time - epoch_start) // prediction_horizon)
    estimated_time_after_prediction = current_time + maximum_time_for_prediction
    earliest_time_to_predict_at = epoch_start + (
        prediction_intervals_since_epoch + 1
    ) * prediction_horizon

    if (estimated_time_after_prediction > earliest_time_to_predict_at):
        future_prediction_time_factor = 1 + (
            estimated_time_after_prediction - earliest_time_to_predict_at
        ) // prediction_horizon
        prediction_time = earliest_time_to_predict_at + future_prediction_time_factor * prediction_horizon
        print_with_time(
            f"Due to slowness of the prediction, skipping next time point for prediction and targeting "
            f"{future_prediction_time_factor} intervals ahead (prediction at time point {prediction_time})"
        )
    else:
        prediction_time = earliest_time_to_predict_at + prediction_horizon

    print_with_time(
        "Time is now " + str(current_time) +
        " and next prediction batch starts with prediction for time " + str(prediction_time)
    )
    return prediction_time


def calculate_and_publish_predictions(application_state, application_name, maximum_time_required_for_prediction):
    start_forecasting = application_state.start_forecasting

    while start_forecasting:
        print_with_time(
            f"Using {LstmPredictorState.configuration_file_location} "
            f"for configuration details related to forecasts of {application_state.application_name}..."
        )

        application_state.next_prediction_time = update_prediction_time(
            application_state.epoch_start,
            application_state.prediction_horizon,
            maximum_time_required_for_prediction
        )

        # -- FIX #1: Use `.get()` to avoid KeyError on missing attributes in previous_prediction
        for attribute in application_state.metrics_to_predict:
            prev_pred = (application_state.previous_prediction or {}).get(attribute)
            if prev_pred is not None:
                if prev_pred.last_prediction_time_needed > maximum_time_required_for_prediction:
                    maximum_time_required_for_prediction = prev_pred.last_prediction_time_needed

        # Calculate how long we sleep until we generate the next predictions
        wait_time = (
            application_state.next_prediction_time
            - application_state.prediction_horizon
            - time.time()
        )
        print_with_time(
            "Waiting for "
            + str((int(wait_time * 100)) / 100)
            + " seconds, until time "
            + datetime.datetime.fromtimestamp(
                application_state.next_prediction_time - application_state.prediction_horizon
            ).strftime('%Y-%m-%d %H:%M:%S')
        )

        if (wait_time > 0):
            time.sleep(wait_time)
            if (not start_forecasting):
                break

        Utilities.load_configuration()
        application_state.update_monitoring_data()
        first_prediction = None

        for prediction_index in range(0, LstmPredictorState.total_time_intervals_to_predict):
            prediction_time = int(application_state.next_prediction_time) + \
                              prediction_index * application_state.prediction_horizon
            try:
                print_with_time(
                    f"Initiating predictions for all metrics for next_prediction_time = {application_state.next_prediction_time}"
                )
                prediction = predict_attributes(application_state, prediction_time)

                # Save the first prediction for reference in application_state
                if (prediction_time == int(application_state.next_prediction_time)):
                    first_prediction = prediction

            except Exception as e:
                print_with_time(
                    f"Could not create a prediction for some/all metrics for time point "
                    f"{application_state.next_prediction_time}: {e}\n"
                    f"{traceback.format_exc()}"
                )
                break  # skip further intervals in this batch

            # Publish each metric’s prediction
            for attribute in application_state.metrics_to_predict:
                # If the metric’s prediction is missing or invalid, skip
                if (prediction[attribute] is None or not prediction[attribute].prediction_valid):
                    print_with_time(
                        f"Skipping {attribute}, no valid prediction or the prediction is None."
                    )
                    continue

                if (LstmPredictorState.disconnected or LstmPredictorState.check_stale_connection()):
                    logging.info("Possible problem due to disconnection or a stale connection")

                message_not_sent = True
                current_time = int(time.time())
                pred_obj = prediction[attribute]

                prediction_message_body = {
                    "metricValue": float(pred_obj.value),
                    "level": 3,
                    "timestamp": current_time,
                    "probability": 0.95,
                    "confidence_interval": [
                        float(pred_obj.lower_confidence_interval_value),
                        float(pred_obj.upper_confidence_interval_value)
                    ],
                    "predictionTime": prediction_time,
                }

                training_models_message_body = {
                    "metrics": application_state.metrics_to_predict,
                    "forecasting_method": "lstm",
                    "timestamp": current_time,
                }

                # Keep trying to send, in case we get a temporary broker issue
                while message_not_sent:
                    try:
                        for publisher in LstmPredictorState.broker_publishers:
                            if publisher.key == "publisher_" + application_name + "-" + attribute:
                                publisher.send(prediction_message_body, application_name)

                        message_not_sent = False
                        print_with_time(
                            f"Successfully sent prediction message for {attribute}:\n{prediction_message_body}\n"
                        )
                    except ConnectionError as exception:
                        logging.error("Error sending intermediate prediction: " + str(exception))
                        LstmPredictorState.disconnected = False

        # Update application_state with the first prediction from the batch
        if (first_prediction is not None):
            application_state.previous_prediction = first_prediction

    # If we exit the loop, forecasting is done for this thread.
    print_with_time(
        f"Terminating prediction loop for application: {application_name}. start_forecasting = {start_forecasting}"
    )


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

    def on_message(self, key, address, body, message: Message, context):
        address = address.replace("topic://" + LstmPredictorState.GENERAL_TOPIC_PREFIX, "")
        if address.startswith(LstmPredictorState.MONITORING_DATA_PREFIX):
            address = address.replace(LstmPredictorState.MONITORING_DATA_PREFIX, "", 1)

            if address == 'metric_list':
                application_name = body["name"]
                message_version = body["version"]
                application_state = None
                individual_application_state = {}

                # Decide if we update an existing ApplicationState or create a new one
                app_defined = application_name in LstmPredictorState.individual_application_state
                if app_defined and (
                    message_version == LstmPredictorState.individual_application_state[application_name].message_version
                ):
                    application_state = LstmPredictorState.individual_application_state[application_name]
                    print_with_time("Using existing application definition for " + application_name)
                else:
                    if app_defined:
                        print_with_time("Updating application " + application_name)
                    else:
                        print_with_time("Creating new application " + application_name)

                    application_state = ApplicationState(application_name, message_version)

                # Update metric bounds from the incoming metric_list
                metric_list_object = body["metric_list"]
                lb_val = application_state.lower_bound_value
                ub_val = application_state.upper_bound_value
                for metric_object in metric_list_object:
                    lb_val[metric_object["name"]] = float(metric_object["lower_bound"])
                    ub_val[metric_object["name"]] = float(metric_object["upper_bound"])

                application_state.lower_bound_value.update(lb_val)
                application_state.upper_bound_value.update(ub_val)
                application_state.initial_metric_list_received = True

                individual_application_state[application_name] = application_state
                LstmPredictorState.individual_application_state.update(individual_application_state)

        elif address.startswith(LstmPredictorState.FORECASTING_CONTROL_PREFIX):
            address = address.replace(LstmPredictorState.FORECASTING_CONTROL_PREFIX, "", 1)
            logging.info("The address is " + address)

            if address == 'test.lstm':
                LstmPredictorState.testing_prediction_functionality = True

            elif address == 'start_forecasting.lstm':
                try:
                    application_name = body["name"]
                    message_version = body.get("version", 1)

                    if (application_name in LstmPredictorState.individual_application_state
                        and message_version <= LstmPredictorState.individual_application_state[application_name].message_version):
                        application_state = LstmPredictorState.individual_application_state[application_name]
                    else:
                        LstmPredictorState.individual_application_state[application_name] = ApplicationState(
                            application_name, message_version
                        )
                        application_state = LstmPredictorState.individual_application_state[application_name]

                    # Only update if we are not already forecasting or the metrics changed
                    new_metrics = body["metrics"]
                    if (not application_state.start_forecasting) or (
                        set(application_state.metrics_to_predict) != set(new_metrics)
                    ):
                        application_state.metrics_to_predict = new_metrics
                        print_with_time(
                            "Received request to start predicting the following metrics: "
                            + ",".join(application_state.metrics_to_predict)
                            + " for application "
                            + application_name
                            + ", proceeding with the prediction process"
                        )
                        if not application_state.start_forecasting:
                            # Coarse initialization of metric bounds if no metric_list has arrived
                            for metric in application_state.metrics_to_predict:
                                if metric not in application_state.lower_bound_value:
                                    application_state.lower_bound_value[metric] = None
                                if metric not in application_state.upper_bound_value:
                                    application_state.upper_bound_value[metric] = None
                        else:
                            # If we already had some metrics, merge new ones
                            old_metrics = set(application_state.metrics_to_predict)
                            for metric in (set(new_metrics) - old_metrics):
                                application_state.lower_bound_value[metric] = None
                                application_state.upper_bound_value[metric] = None
                    else:
                        print_with_time(
                            "Received duplicate start_forecasting for same metrics: "
                            + ",".join(application_state.metrics_to_predict)
                            + " for application "
                            + application_name
                        )
                        return

                    # Prepare publishers for each metric
                    application_state.broker_publishers = []
                    for metric in application_state.metrics_to_predict:
                        LstmPredictorState.broker_publishers.append(PredictionPublisher(application_name, metric))

                    LstmPredictorState.publishing_connector = connector.EXN(
                        'publishing_' + LstmPredictorState.forecaster_name + '-' + application_name,
                        handler=BootStrap(),
                        consumers=[],
                        publishers=LstmPredictorState.broker_publishers,
                        url=LstmPredictorState.broker_address,
                        port=LstmPredictorState.broker_port,
                        username=LstmPredictorState.broker_username,
                        password=LstmPredictorState.broker_password
                    )
                    # Start the new connector in a thread
                    thread = threading.Thread(
                        target=LstmPredictorState.publishing_connector.start,
                        args=()
                    )
                    thread.start()

                except Exception as e:
                    print_with_time(
                        "Could not parse or process the 'start_forecasting' message \n" + str(body)
                    )
                    print(traceback.format_exc())
                    return

                # Mark this application’s forecasting as active
                try:
                    application_state = LstmPredictorState.individual_application_state[application_name]
                    application_state.start_forecasting = True
                    application_state.epoch_start = body["epoch_start"]
                    application_state.prediction_horizon = int(body["prediction_horizon"])
                    application_state.next_prediction_time = update_prediction_time(
                        application_state.epoch_start,
                        application_state.prediction_horizon,
                        LstmPredictorState.prediction_processing_time_safety_margin_seconds
                    )
                    print_with_time(
                        "start_forecasting message received, epoch_start = "
                        + str(application_state.epoch_start)
                        + ", horizon = "
                        + str(application_state.prediction_horizon)
                    )
                except Exception as e:
                    print_with_time("Problem while retrieving epoch_start/prediction_horizon")
                    print(traceback.format_exc())
                    return

                # Safety margin for the first iteration
                maximum_time_required_for_prediction = (
                    LstmPredictorState.prediction_processing_time_safety_margin_seconds
                )

                # -- FIX #2: Only start the forecasting thread if it’s not already running
                if (application_state.prediction_thread is None
                    or not application_state.prediction_thread.is_alive()
                ):
                    application_state.prediction_thread = threading.Thread(
                        target=calculate_and_publish_predictions,
                        args=[application_state, application_name, maximum_time_required_for_prediction]
                    )
                    application_state.prediction_thread.start()

            elif address == 'stop_forecasting.lstm':
                application_name = body["name"]
                if application_name not in LstmPredictorState.individual_application_state:
                    print_with_time("No known application: " + application_name)
                    return

                application_state = LstmPredictorState.individual_application_state[application_name]
                metrics_to_remove = body["metrics"]

                if not metrics_to_remove:
                    # If the message has an empty list, we stop forecasting altogether
                    print_with_time("No metrics specified => stop forecasting entirely.")
                    application_state.start_forecasting = False
                    if application_state.prediction_thread is not None:
                        application_state.prediction_thread.join()
                else:
                    for metric in metrics_to_remove:
                        if metric in application_state.metrics_to_predict:
                            print_with_time("Stopping predictions for metric " + metric)
                            application_state.metrics_to_predict.remove(metric)

                    # If no metrics left, stop everything
                    if len(application_state.metrics_to_predict) == 0:
                        application_state.start_forecasting = False
                        if application_state.prediction_thread is not None:
                            application_state.prediction_thread.join()

            else:
                print_with_time("Received message at " + address + " but could not handle it.")

        else:
            print_with_time("Received message " + str(body) + " but could not handle it")


def main():
    if len(sys.argv) < 2:
        print("Error: Configuration file location must be provided as an argument.")
        sys.exit(1)

    LstmPredictorState.configuration_file_location = sys.argv[1]
    print(os.listdir("."))


# Change to the appropriate directory for invoking the forecasting script
os.chdir("/app/lstm")  # Update the path to match the Docker container structure

Utilities.load_configuration()
Utilities.update_influxdb_organization_id()

logging.basicConfig(level=logging.INFO)
id = "lstm"
LstmPredictorState.disconnected = True

while True:
    topics_to_subscribe = [
        "eu.nebulouscloud.monitoring.metric_list",
        "eu.nebulouscloud.monitoring.realtime.>",
        "eu.nebulouscloud.forecasting.start_forecasting.lstm",
        "eu.nebulouscloud.forecasting.stop_forecasting.lstm"
    ]
    current_consumers = []

    for topic in topics_to_subscribe:
        current_consumer = core.consumer.Consumer(
            key='monitoring_' + topic,
            address=topic,
            handler=ConsumerHandler(),
            topic=True,
            fqdn=True
        )
        LstmPredictorState.broker_consumers.append(current_consumer)
        current_consumers.append(current_consumer)

    LstmPredictorState.subscribing_connector = connector.EXN(
        LstmPredictorState.forecaster_name,
        handler=BootStrap(),
        consumers=LstmPredictorState.broker_consumers,
        url=LstmPredictorState.broker_address,
        port=LstmPredictorState.broker_port,
        username=LstmPredictorState.broker_username,
        password=LstmPredictorState.broker_password
    )

    thread = threading.Thread(target=LstmPredictorState.subscribing_connector.start, args=())
    thread.start()
    LstmPredictorState.disconnected = False

    print_with_time("Checking (EMS) broker connectivity state...")

    if (LstmPredictorState.disconnected or LstmPredictorState.check_stale_connection()):
        try:
            LstmPredictorState.disconnected = True
            print_with_time("Possible problem in the connection")
        except Exception as e:
            print_with_time("Encountered exception while connecting to broker")
            print(traceback.format_exc())
            LstmPredictorState.disconnected = True
            time.sleep(5)
            continue

    LstmPredictorState.disconnection_handler.acquire()
    LstmPredictorState.disconnection_handler.wait()
    LstmPredictorState.disconnection_handler.release()
