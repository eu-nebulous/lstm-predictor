from exn.core.publisher import Publisher


class PredictionPublisher(Publisher):
    metric_name = ""
    def __init__(self,application_name,metric_name):
        super().__init__('publisher_'+application_name+'-'+metric_name, 'eu.nebulouscloud.preliminary_predicted.lstm.'+metric_name, True,True)
        self.metric_name = metric_name

    def send(self, body={}, application=""):
        super(PredictionPublisher, self).send(body, application)
