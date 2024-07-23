from datetime import datetime

import numpy as np


def serialize_datetime(obj: any):
    """
    Serialize a datetime object to a string.
    @param obj: datetime object
    @return: string serialization
    """
    if isinstance(obj, datetime):
        return obj.isoformat(sep=' ', timespec='milliseconds')
    return obj


def deserialize_datetime(obj: any):
    """
    Deserialize a datetime string representation to a datetime object.
    @param obj: datetime string
    @return: datetime object
    """
    if isinstance(obj, str):
        try:
            return datetime.fromisoformat(obj)
        except ValueError:
            pass
    return obj


def serialize_dict(data: any):
    """
    Serialize the content of a dictionary
    @param data: dictionary to serialize
    @return: serialized dictionary
    """
    if isinstance(data, dict):
        converted_data = {}
        for key, value in data.items():
            converted_data[key] = serialize_dict(value)
        return converted_data
    elif isinstance(data, list):
        return [serialize_dict(item) for item in data]
    elif isinstance(data, np.int64):
        return int(data)
    elif isinstance(data, datetime):
        return serialize_datetime(data)
    else:
        return data


def deserialize_dict(data: any):
    """
    Deserialize the content of a dictionary
    @param data: dictionary to deserialize
    @return: deserialized dictionary
    """
    if isinstance(data, dict):
        converted_data = {}
        for key, value in data.items():
            converted_data[key] = deserialize_dict(value)
        return converted_data
    elif isinstance(data, list):
        return [deserialize_dict(item) for item in data]
    elif isinstance(data, int):
        return np.int64(data)
    elif isinstance(data, str):
        return deserialize_datetime(data)
    else:
        return data
