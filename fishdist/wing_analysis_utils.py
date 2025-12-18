# TODO maybe not sure --> extend the hinge_blade_sep to the top and bottom of the image and force apply it as a cut of the image to be sure the cells are not connected through the hinge !!!

import numpy as np
from batoolset.ta.database.sql import TAsql, combine_single_file_queries, query_db_and_get_results
import traceback
from skimage.filters import threshold_yen
from batoolset.tools.logger import TA_logger

logger = TA_logger()


def compute_weighted_centroid(coords, intensity_image):
    """
    Computes the weighted centroid of the given coordinates in an intensity image.

    Args:
        coords (numpy.ndarray): The coordinates.
        intensity_image (numpy.ndarray): The intensity image.

    Returns:
        numpy.ndarray: The weighted centroid.

    """
    intensities = intensity_image[tuple(coords.T)]
    intensities = intensities.astype(float)
    coords = coords.astype(float)
    return np.sum(coords * intensities[..., np.newaxis], axis=0) / np.sum(intensities)


def get_q1_q3(x):
    """
    Computes the 1st and 3rd quartiles of the given data.

    Args:
        x (numpy.ndarray): The data.

    Returns:
        numpy.ndarray: The 1st and 3rd quartiles.

    """
    return np.percentile(x, [25, 75])

def compute_iqr(x):
    """
    Computes the interquartile range (third quartile - first quartile), a distribution measure not sensitive to outliers.

    Args:
        x (numpy.ndarray): The data.

    Returns:
        float: The interquartile range.

    """
    q1, q3 = get_q1_q3(x)
    return q3 - q1



def add_to_db_sql(sql_file, table_name, headers, data_rows):
    """
    Adds data to a SQL database table.

    Args:
        sql_file (str): The path to the SQL file.
        table_name (str): The name of the table.
        headers (list): The headers of the table.
        data_rows (list): The data rows to be added.

    Returns:
        dict: The formatted table containing the data.

    """
    if not isinstance(data_rows, np.ndarray):
        data_rows = np.asarray(data_rows, dtype=object)
    data_rows = data_rows.T
    finally_formatted_table = {}

    try:
        for iii, header in enumerate(headers):
            try:
                finally_formatted_table[header] = data_rows[iii].tolist()
            except:
                logger.warning('no data to be added --> the column will be empty')
                finally_formatted_table[header] = []
    except:
        logger.warning('Something went wrong during table creation')

    if table_name is None or sql_file is None:
        return finally_formatted_table

    append_to_file(sql_file, table_name, finally_formatted_table)


def append_to_file(sql_file, table_name, finally_formatted_table):
    """
    Appends a formatted table to a SQL database.

    Args:
        sql_file (str): The path to the SQL file.
        table_name (str): The name of the table.
        finally_formatted_table (dict): The formatted table data to be appended.

    """
    db = None
    try:
        db = TAsql(sql_file)
        db.create_and_append_table(table_name=table_name, datas=finally_formatted_table)
    except:
        traceback.print_exc()
        logger.error('Something went wrong, DB could not be created')
    finally:
        if db is not None:
            try:
                db.close()
            except:
                traceback.print_exc()

