import traceback

from batoolset.ta.database.sql import TAsql, save_data_to_csv, prepend_to_content
from batoolset.tools.logger import TA_logger  # logging
from batoolset.files.tools import smart_name_parser

logger = TA_logger()



def combine_single_file_queries(lst, sql_command, table_names=None, db_name='pyTA.db', return_header=False, prepend_frame_nb=True, prepend_file_name=True, output_filename=None ): # , save_master_db=False
    """
    Combine the results of single file queries into a master data set.

    Args:
        lst (list): List of file paths to query.
        sql_command (str): SQL command to execute for each file.
        table_names (str or list, optional): Names of tables to include in the query. Defaults to None.
        db_name (str, optional): Name of the database file. Defaults to 'pyTA.db'.
        return_header (bool, optional): Flag to indicate whether to return the header. Defaults to False.
        prepend_frame_nb (bool, optional): Flag to indicate whether to prepend the frame number to the data. Defaults to True.
        prepend_file_name (bool, optional): Flag to indicate whether to prepend the file name to the data. Defaults to True.
        output_filename (str, optional): Name of the output file to save the combined data. Defaults to None.

    Returns:
        tuple or list: If `output_filename` is None and `return_header` is False, returns the combined data as a list.
                      If `output_filename` is None and `return_header` is True, returns the header and combined data as a tuple.
                      If `output_filename` is provided, saves the combined data to the specified file and returns None.

    # Examples:
    #     >>> lst = ['file1.db', 'file2.db', 'file3.db']
    #     >>> sql_command = 'SELECT * FROM data'
    #     >>> combine_single_file_queries(lst, sql_command, table_names='my_table', output_filename='combined_data.csv')
    #
    #     >>> lst = ['file1.db', 'file2.db', 'file3.db']
    #     >>> sql_command = 'SELECT * FROM data WHERE value > 10'
    #     >>> header, data = combine_single_file_queries(lst, sql_command, return_header=True)
    #     >>> print(header)
    #     >>> print(data)
    """

    if lst is not None and lst and isinstance(lst, list):
        # merged output
        master_data = []
        header = None
        for iii, file in enumerate(lst):
            if db_name:
                # Parse the database file path
                db_path = smart_name_parser(file, db_name)
            else:
                db_path = file
            # Create a TAsql instance for the database
            db = TAsql(db_path)
            try:
                final_command = sql_command

                if table_names is not None:
                    if isinstance(table_names, list):
                        # Iterate over the table names and check if they exist in the database
                        for table_name in table_names:
                            if db.exists(table_name):
                                final_command += table_name
                                break
                    else:
                        final_command += table_names

                # Execute the SQL command and get the results
                # try:
                out = db.run_SQL_command_and_get_results(final_command, return_header=((iii == 0) or header is None)) # bug fix for header error if first table misses the appropriate table --> this is a hack but I have cases where I may have missing tables for some files and this is not an error this is intended
                # except:
                #     traceback.print_exc()
                #     print(sql_command, 'returned empty data for', file, '--> ignoring')
                #     out = None, None

                if isinstance(out, tuple):
                    header, data = out
                    if prepend_file_name:
                        header = prepend_to_content(header, 'filename')
                    if prepend_frame_nb:
                        header = prepend_to_content(header, 'frame_nb')
                else:
                    data = out
                if prepend_file_name:
                    data = prepend_to_content(data, file)
                if prepend_frame_nb:
                    data = prepend_to_content(data, iii)

                if data is not None:
                    # Append the data to the master data list
                    master_data.extend(data)
                else:
                    print(sql_command, 'returned empty data for',file,'--> ignoring')
            except:
                traceback.print_exc()
            finally:
                # if save_master_db:
                #     if isinstance(save_master_db, bool):
                #         save_master_db = smart_name_parser(output_filename, 'full_no_ext')+'_master.db'
                #
                # Close the database connection
                db.close()

        if not master_data:
            master_data = None
        if output_filename is not None:
            # Save the combined data to the output file
            save_data_to_csv(output_filename, header, master_data)
        else:
            if return_header:
                return header, master_data
            return master_data
    else:
        logger.error('No input list -> nothing to do...')
        if return_header:
            return None, None
        return None
