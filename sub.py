from datetime import datetime
from decimal import Decimal
from openpyxl import load_workbook
import pandas as pd
import requests
import config


# МЕТОДЫ КОНВЕРТАЦИИ
def price(sender):  # цена
    return sender.units + sender.nano / 1e9


def no_timestamp(sender):  # время
    # return datetime.fromtimestamp(sender)
    return f"{datetime.fromtimestamp(sender):%m.%d %H:%M}"


def nano_price(value, i):
    # Преобразуем число в формат Decimal
    decimal_value = Decimal(str(value))

    # Разбиваем число на целую и дробную части
    units = int(decimal_value)
    nano = int((decimal_value - Decimal(units)) * Decimal('1e9'))
    if i == 'u':
        return units
    if i == 'n':
        return nano


def check_match(row, arr):
    if row['time'] in arr:
        return '🌟'
    else:
        return ''


def load_excel():
    url = config.excel_url
    response = requests.get(url)
    if response.status_code == 200:
        with open('test.xlsx', 'wb') as file:
            file.write(response.content)
            workbook = load_workbook('test.xlsx')
            print(f"файл обновлен, проверочное слово: {workbook['Лист1']['B1'].value}")
            return pd.read_excel('test.xlsx', skiprows=1)
    else:
        print('Ошибка при скачивании файла.')


