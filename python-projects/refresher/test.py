from datetime import datetime
from typing import List, Optional

class BankAccount:
    """Professional bank account implementation."""
    
    bank_name = "Python Bank"  # Class attribute
    
    def __init__(self, owner: str, initial_balance: float = 0.0):
        self.owner = owner
        self._balance = initial_balance  # Protected attribute
        self.__pin = None  # Private attribute
        self.transactions: List[dict] = []
        self._account_type = ""  # Protected attribute
        self._log_transaction("Account created", initial_balance)
    
    @property
    def balance(self) -> float:
        """Protected balance with property decorator."""
        return self._balance
    
    @balance.setter
    def balance(self, amount: float):
        if amount < 0:
            raise ValueError("Balance cannot be negative")
        self._balance = amount
    
    def deposit(self, amount: float) -> None:
        """Add funds to account."""
        if amount <= 0:
            raise ValueError("Deposit must be positive")
        self._balance += amount
        self._log_transaction("Deposit", amount)
    
    def _log_transaction(self, type: str, amount: float):
        """Private method for logging."""
        self.transactions.append({
            'timestamp': datetime.now(),
            'type': type,
            'amount': amount,
            'balance': self._balance
        })
        
    
    @classmethod
    def create_savings_account(cls, owner: str):
        """Factory method for savings accounts."""
        account = cls(owner, 100.0)  # Minimum balance
        account.account_type = "Savings"  # Set account type
        return account
    
    @staticmethod
    def calculate_interest(principal: float, rate: float, years: int) -> float:
        """Utility method for interest calculation."""
        return principal * (1 + rate) ** years
    
    @property
    def account_type(self) -> str:
        """Property for account type."""
        return self._account_type

    @account_type.setter
    def account_type(self, value: str):
        # Optional: validate allowed types
        allowed_types = {"Savings", "Checking", ""}
        if value not in allowed_types:
            raise ValueError(f"Invalid account type: {value}")
        self._account_type = value