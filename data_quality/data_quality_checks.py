from abc import ABC, abstractmethod


class DataQualityChecks(ABC):
    """
    Abstract base class for data quality checks.
    """

    @abstractmethod
    def metrics(self, *args, **kwargs):
        """
        Abstract method for calculating metrics related to a data quality dimension.
        """
        pass

    @abstractmethod
    def rules(self, *args, **kwargs):
        """
        Abstract method for applying rules to check data quality.
        """
        pass