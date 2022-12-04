from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pickle
from collections import namedtuple
import re
import pandas as pd
import numpy as np
from typing import Union
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

app = FastAPI()


class Item(BaseModel):
    name: str
    year: int
    selling_price: Union[int, None] = None
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float


class Items(BaseModel):
    objects: List[Item]


VehiclePriceModel = namedtuple('VehiclePriceModel', ['model', 'normalizer', 'one_hot_encoder'])

with open('VehiclePriceModel.pickle', 'rb') as f:
    model = VehiclePriceModel(**pickle.load(f))


def isfloat(string: str) -> bool:
    string = str(string)
    if string.replace('.', '', 1).isdigit():
        return True
    return False


def regex_findall(string: str, regex_str: str) -> list:
    regex = re.compile(regex_str)
    try:
        f_list = regex.findall(string)[0]
    except IndexError:
        return []
    return f_list


def float_from_first_list_el(lst: list) -> Union[float, None]:
    if len(lst) <= 1:
        return None
    if isfloat(lst[0]):
        return float(lst[0])
    else:
        return None


def parse_one_val_and_unit(string, regex_str):
    string = str(string).strip().lower()
    if isfloat(string):
        val = float(string)
        unit = None
    else:
        val_f = regex_findall(string, regex_str)
        val = float_from_first_list_el(val_f)
        unit = val_f[1] if len(val_f) == 2 else None
    return val, unit


def parse_torque(string):
    string = str(string).replace(',', '')
    torque, unit_t = parse_one_val_and_unit(string, r"(?:(\d+\.?\d*)(@|\s?[^(]?[kK][gG][mM]|\s?[nN][mM]))")
    rpm, unit_r = parse_one_val_and_unit(string, r"(?:(\d+,\d+|\d+)(\s?\(\D+\)|\s?\s?(?:rpm|RPM)))")
    if isinstance(torque, float) and isinstance(unit_t, str) and \
            (unit_t == 'kgm'
             or str(unit_r).replace(' ', '') == '(kgm@rpm)'):
        torque *= 9.8067
    rpm = float(rpm) if isinstance(rpm, float) else None
    return torque, rpm


def parse_mileage(string):
    mileage, unit = parse_one_val_and_unit(string, r'(\d+\.?\d*)(?: ?(\D+))?')
    if isinstance(mileage, float) and unit == 'km/kg':
        mileage /= 0.8
    return mileage


def parse_engine(string):
    engine, unit = parse_one_val_and_unit(string, r'(\d+\.?\d*)(?: ?(\D+))?')
    return engine


def parse_max_power(string):
    max_power, unit = parse_one_val_and_unit(string, r'(\d+\.?\d*)(?: ?(\D+))?')
    return max_power


def get_df_with_parsed_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['mileage'] = df['mileage'].apply(parse_mileage)
    df['engine'] = df['engine'].apply(parse_engine)
    df['max_power'] = df['max_power'].apply(parse_max_power)
    df.insert(df.columns.get_loc('torque')+1, 'max_torque_rpm', None)
    df['max_torque_rpm'] = df['torque'].apply(lambda x: parse_torque(x)[1])
    df['torque'] = df['torque'].apply(lambda x: parse_torque(x)[0])
    return df


@app.post("/predict_item")
def predict_item(item: Item) -> float:
    df = pd.DataFrame([item.dict()])
    df = df.astype({'fuel': 'category', 'seller_type': 'category',
                    'transmission': 'category', 'owner': 'category',
                    'seats': 'category'})
    df = get_df_with_parsed_features(df)
    df_num = df.select_dtypes(exclude=["category"]).drop(['selling_price', 'name'], axis=1)
    df_cat = df.select_dtypes(include=["category"])
    item_ohe = model.one_hot_encoder.transform(df_cat).toarray()
    item_ohe = np.concatenate([df_num, item_ohe], axis=1)
    item_ohe_scaled = model.normalizer.transform(item_ohe)
    return model.model.predict(item_ohe_scaled)[0]

@app.post("/predict_items")
def predict_items(items: List[Item]) -> List[float]:
    return ...