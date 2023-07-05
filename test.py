import pandas as pd
import numpy as np
import time
import pathlib
import json
from datetime import datetime
from time import mktime
from tqdm import tqdm
# from tqdm.notebook import tqdm

alarm_file_path = "data/alarm_history_processed_removed_unsed.xlsx"
alarm_df = pd.read_excel(alarm_file_path)


def parse_time(x):
    time_struct = time.strptime(x[:-2], "%Y-%m-%dT%H:%M:%S.%fZ")
    return datetime.fromtimestamp(mktime(time_struct))

def get_error_code(x):
    text = x[:1] + '"' + x[1:]
    text = text.replace(";", ',')[:-1]
    json_file = json.loads(text, )
    return json_file['errorCode']

def acor_score(a_count, b_count, ab_count):
    return (ab_count / a_count) / ( 2 - ab_count / b_count)



alarm_df['time'] = alarm_df['alarmRaisedTime'].apply(parse_time)
# alarm_df['errorCode'] = alarm_df['metadata'].apply(get_error_code)

alarm_cr_df = alarm_df[['time', 'errorCode', 'subObjectInstanceNameNocPro', 'objectInstanceName','perceivedSeverity']]


from pydantic import BaseModel

class AlarmSample(BaseModel):
    time: datetime
    errorCode: str
    subObjectInstanceNameNocPro: str
    objectInstanceName: str
    perceivedSeverity: str
    
    def is_together(self, other, mins=5):
        # time delta < 5 min and subObjectInstanceNameNocPro different
        if abs((self.time - other.time).total_seconds()) > mins * 60:
            return False
        return True
    
class ObjectInstance:
    def __init__(self, df : pd.DataFrame):
        """ historical alarm data : get dict of alarm type, each alarm type is list of sample sort by time"""
        error_code_group = df.groupby('errorCode')
        self.historical_alarm_data = {}
        for error_code, group in error_code_group:
            records = group.sort_values(by=['time']).to_dict('records')
            records = [AlarmSample(**record) for record in records]
            # sort by time
            records = sorted(records, key=lambda x: x.time)
            self.historical_alarm_data[error_code] = records
        
        self._alarm_type = list(self.historical_alarm_data.keys())
        
    @property
    def alarm_type(self):
        return self._alarm_type
    
    def get_number_alarm(self, alarm_type):
        if alarm_type not in self._alarm_type:
            return 0
        return len(self.historical_alarm_data[alarm_type])

class AlarmCorrelation:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.object_instance_group = df.groupby('objectInstanceName')
        self.object_instances = {}
        for object_instance_name, group in self.object_instance_group:
            self.object_instances[object_instance_name] = ObjectInstance(group)
        
        self.alarm_type = list(set([alarm_type for object_instance in self.object_instances.values() for alarm_type in object_instance.alarm_type]))
        self._alarm_correlation_matrix = np.zeros((len(self.alarm_type), len(self.alarm_type)))
        self._ab_count = np.zeros((len(self.alarm_type), len(self.alarm_type)))
        self._alarm_list_dict = {}
        self._calculate_correlation_matrix()
            
    def _get_alarm_correlation(self, alarm_type_a, alarm_type_b):
        self._alarm_list_dict[(alarm_type_a, alarm_type_b)] = []
        my_list = self._alarm_list_dict[(alarm_type_a, alarm_type_b)]
        if alarm_type_a == alarm_type_b:
            return 0, 0
        assert alarm_type_a in self.alarm_type and alarm_type_b in self.alarm_type, f"alarm type {alarm_type_a} or {alarm_type_b} not in alarm type list"
        a_count = 0
        b_count = 0
        ab_count = 0
        for object_instance in self.object_instances.values():
            len_a = object_instance.get_number_alarm(alarm_type_a)
            len_b = object_instance.get_number_alarm(alarm_type_b)
            a_count += len_a
            b_count += len_b
            i, j = 0, 0
            while i < len_a and j < len_b:
                instance_alarm_a_i = object_instance.historical_alarm_data[alarm_type_a][i]
                instance_alarm_b_j = object_instance.historical_alarm_data[alarm_type_b][j]
                if instance_alarm_a_i.is_together(instance_alarm_b_j) :
                    i += 1
                    j += 1
                    if instance_alarm_a_i.subObjectInstanceNameNocPro != instance_alarm_b_j.subObjectInstanceNameNocPro:
                        if len(my_list) < 100:
                            my_list.append((instance_alarm_a_i, instance_alarm_b_j))
                        ab_count += 1
                elif instance_alarm_a_i.time < instance_alarm_b_j.time:
                    i += 1
                elif instance_alarm_b_j.time < instance_alarm_a_i.time:
                    j += 1
                else:
                    raise Exception("error")
        
        return acor_score(a_count, b_count, ab_count) , ab_count
    
    def _calculate_correlation_matrix(self):
        for i, alarm_type_a in tqdm(enumerate(self.alarm_type)):
            for j, alarm_type_b in enumerate(self.alarm_type):
                t = self._get_alarm_correlation(alarm_type_a, alarm_type_b)
                self._alarm_correlation_matrix[i, j], self._ab_count[i,j] = t[0], t[1]
    
    @property
    def alarm_correlation_matrix(self):
        return self._alarm_correlation_matrix
    
    
    def top_k_correlation(self, k=5):
        idx = np.argpartition(-self._alarm_correlation_matrix.ravel(),k)[:k]
        idx_stack = np.column_stack(np.unravel_index(idx, self._alarm_correlation_matrix.shape))
        return [(self.alarm_type[i], self.alarm_type[j], self._alarm_correlation_matrix[i, j], self._ab_count[i,j], self._alarm_list_dict[(self.alarm_type[i], self.alarm_type[j])]) for i, j in idx_stack]


alarm_correlation = AlarmCorrelation(alarm_cr_df)  

top_200 = alarm_correlation.top_k_correlation(200)
top_200_df = pd.DataFrame(top_200, columns=['alarm_type_a', 'alarm_type_b', 'correlation', 'ab_count', 'alarm_list'])
top_200_df.to_excel('top_200.xlsx')