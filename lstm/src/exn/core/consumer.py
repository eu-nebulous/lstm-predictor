import logging

from proton import Event
from .handler import Handler

from . import link

from proton.handlers import MessagingHandler

_logger = logging.getLogger(__name__)
_logger.setLevel(level=logging.INFO)


class Consumer(link.Link, MessagingHandler):
    application = None

    def __init__(self, key, address, handler: Handler, application=None, topic=False, fqdn=False):
        super(Consumer, self).__init__(key, address, topic, fqdn)
        self.application = application
        self.handler = handler
        self.handler._consumer = self

    def should_handle(self, event: Event):

        should = event.link.name == self._link.name and \
            (self.application is None or event.message.subject == self.application)

        _logger.debug(f"[{self.key}] checking if link is the same {event.link.name}={self._link.name}  "
                      f" and application {self.application}={event.message.subject}  == {should}")

        return should

    def on_start(self, event: Event) -> None:
        _logger.debug(f"[{self.key}]  on_start")

    def on_message(self, event):
        _logger.debug(f"[{self.key}]  handling event with  address => {event.message.address}")
        try:
            if self.should_handle(event):
                self.handler.on_message(self.key, event.message.address, event.message.body, event.message)

        except Exception as e:
            _logger.error(f"Received message: {e}")
