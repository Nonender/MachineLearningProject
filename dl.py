import os
import pandas as pd
from tradingcalendar import Calendar


class MeowDataLoader:
    def __init__(self, h5dir):
        self.h5dir = h5dir
        self.calendar = Calendar()
        self._available = None  # cached set of available dates

    def available_dates(self):
        """Return set of dates that have corresponding HDF5 files."""
        if self._available is None:
            if not os.path.isdir(self.h5dir):
                self._available = set()
            else:
                self._available = {
                    int(f.replace(".h5", ""))
                    for f in os.listdir(self.h5dir)
                    if f.endswith(".h5")
                }
        return self._available

    def has_date(self, date):
        """Check if data exists for a given date."""
        return date in self.available_dates()

    def load_date(self, date):
        if not self.calendar.is_trading_day(date):
            raise ValueError("Not a trading day: {}".format(date))
        h5_file = os.path.join(self.h5dir, "{}.h5".format(date))
        df = pd.read_hdf(h5_file)
        df.loc[:, "date"] = date
        precols = ["symbol", "interval", "date"]
        df = df[precols + [x for x in df.columns if x not in precols]]
        return df
