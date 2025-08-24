from abc import ABC, abstractmethod
from typing import List

class Notification(ABC):
    @abstractmethod
    def send(self, message: str) -> None:
        pass

class EmailNotification(Notification):
    def send(self, message: str) -> None:
        print(f"Email: {message}")

class SMSNotification(Notification):
    def send(self, message: str) -> None:
        print(f"SMS: {message}")

class NotificationFactory:
    @staticmethod
    def create(channel: str) -> Notification:
        channels = {
            'email': EmailNotification,
            'sms': SMSNotification,
        }
        return channels[channel]()

class NotificationManager:
    def __init__(self):
        self.channels: List[Notification] = []
    
    def add_channel(self, channel: str):
        notification = NotificationFactory.create(channel)
        self.channels.append(notification)
    
    def notify_all(self, message: str):
        for channel in self.channels:
            channel.send(message)

# Usage
manager = NotificationManager()
manager.add_channel('email')
manager.add_channel('sms')
manager.notify_all("System update completed")
