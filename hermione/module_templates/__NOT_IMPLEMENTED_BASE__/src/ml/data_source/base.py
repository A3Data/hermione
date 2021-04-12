from abc import ABC, abstractmethod
import pandas as pd

class DataSource(ABC):
    
    @abstractmethod
    def get_data(self) -> pd.DataFrame:
        """
        Abstract method that is implemented in classes that inherit it
        """
        pass
         
