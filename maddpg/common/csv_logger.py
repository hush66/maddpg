import csv
import threading

class CSVLogger(object):
    def __init__(self, output_file):
        """CSV Logger which can write dict result to csv file.
        Args:
            output_file(str): filename of the csv file.
        Examples:
            ```python
            from parl.utils import CSVLogger
            csv_logger = CSVLogger("result.csv")
            csv_logger.log_dict({"loss": 1, "reward": 2})
            csv_logger.log_dict({"loss": 3, "reward": 4})
            ```
            The content of the `result.csv`:
            loss,reward
            1,2
            3,4
        """

        self.output_file = open(output_file, "w", newline="")

        self.csv_writer = None
        self.lock = threading.Lock()
        self.keys_set = None

    def log_dict(self, result):
        """Ouput result to the csv file.
        Will create the header of the csv file automatically when the function is called for the first time.
        Ususally, the keys of the result should be the same every time you call the function.
        Args:
            result(dict)
        """
        with self.lock:
            assert isinstance(result, dict), "the input should be a dict."
            if self.csv_writer is None:
                self.csv_writer = csv.DictWriter(self.output_file, result.keys())
                self.csv_writer.writeheader()
                self.keys_set = set(result.keys())

            assert set(
                result.keys()
            ) == self.keys_set, "The keys of the dict should be the same as before."

            self.csv_writer.writerow({
                k: v
                for k, v in result.items() if k in self.csv_writer.fieldnames
            })

    def flush(self):
        # flush buffer, write data in the buffer immediately and empty it.
        with self.lock:
            self.output_file.flush()

    def close(self):
        with self.lock:
            if not self.output_file.closed:
                self.output_file.close()

    def __del__(self):
        with self.lock:
            if not self.output_file.closed:
                self.output_file.close()