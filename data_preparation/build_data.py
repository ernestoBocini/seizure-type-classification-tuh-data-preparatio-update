import os
import sys
print('Current working path is %s' % str(os.getcwd()))
sys.path.insert(0, os.getcwd())

import platform
import argparse
import pandas as pd
import numpy as np
import math
import collections
from tabulate import tabulate
import pyedflib
import re
from scipy.signal import resample
import pickle
import h5py
import progressbar
from time import sleep
import glob

parameters = pd.read_csv('/home/ebocini/repos/seizure-type-classification-tuh-data-preparation-update/data_preparation/parameters.csv', index_col=['parameter'])
seizure_type_data = collections.namedtuple('seizure_type_data', ['segment_id','seizure_type', 'data'])

def generate_data_dict(xlsx_file_name, sheet_name):

    """
    Generate a data dictionary from an Excel file containing seizure information.

    Args:
        xlsx_file_name (str): Path to the Excel file with seizure annotations.
        sheet_name (str): The sheet name within the Excel file to read.

    Returns:
        data_dict (dict): A dictionary with seizure types as keys and lists of seizure_info namedtuples as values.
    """
    seizure_info = collections.namedtuple('seizure_info', ['segment_id','start_time', 'end_time', 'label', 'confidence', 'data_type'])
    data_dict = collections.defaultdict(list)

    data_type = xlsx_file_name.split('/')[-2]  # Assumes the second last element in the path is 'train', 'dev', or 'eval'

     # Load the Excel file
    xls = pd.ExcelFile(xlsx_file_name)
    
    for sheet_name in xls.sheet_names:
        # print(f"Processing sheet: {sheet_name}")
        # Read the Excel sheet
        data = pd.read_excel(xls, sheet_name=sheet_name)
        # print(f"Columns in the sheet: {data.columns.tolist()}")

        # Assuming the columns in the Excel file are consistent with what the function originally expected
        for index, row in data.iterrows():
            segment_id = sheet_name  # Using the sheet name as the segment identifier
            start_time = row['start_time']
            end_time = row['stop_time']
            label = row['label']
            confidence = row['confidence']

            v = seizure_info(segment_id=segment_id, start_time=start_time, end_time=end_time, label=label, confidence=confidence, data_type=data_type)
            data_dict[label].append(v)

    return data_dict

def print_type_information(data_dict):

    l = []
    for szr_type, szr_info_list in data_dict.items():
        # how many different patient id for seizure K?
        patient_id_list = [szr_info.segment_id for szr_info in szr_info_list]
        unique_patient_id_list,counts = np.unique(patient_id_list,return_counts=True)

        dur_list = [szr_info.end_time-szr_info.start_time for szr_info in szr_info_list]
        total_dur = sum(dur_list)
        # l.append([szr_type, str(len(szr_info_list)), str(len(unique_patient_id_list)), str(total_dur)])
        l.append([szr_type, (len(szr_info_list)), (len(unique_patient_id_list)), (total_dur)])

        #  numpy.asarray((unique, counts)).T
        '''
        if szr_type=='TNSZ':
            print('TNSZ Patient ID list:')
            print(np.asarray((unique_patient_id_list, counts)).T)
        if szr_type=='SPSZ':
            print('SPSZ Patient ID list:')
            print(np.asarray((unique_patient_id_list, counts)).T)
        '''

    sorted_by_szr_num = sorted(l, key=lambda tup: tup[1], reverse=True)
    print(tabulate(sorted_by_szr_num, headers=['Seizure Type', 'Seizure Num','Patient Num','Duration(Sec)']))

def merge_train_test(train_data_dict, dev_test_data_dict):

    merged_dict = collections.defaultdict(list)
    for item in train_data_dict:
        merged_dict[item] = train_data_dict[item] + dev_test_data_dict[item]

    return merged_dict

def extract_signal(f, signal_labels, electrode_name, start, stop):

    tuh_label = [s for s in signal_labels if 'EEG ' + electrode_name + '-' in s]

    if len(tuh_label) > 1:
        print(tuh_label)
        exit('Multiple electrodes found with the same string! Abort')

    channel = signal_labels.index(tuh_label[0])
    signal = np.array(f.readSignal(channel))

    start, stop = float(start), float(stop)
    original_sample_frequency = f.getSampleFrequency(channel)
    original_start_index = int(np.floor(start * float(original_sample_frequency)))
    original_stop_index = int(np.floor(stop * float(original_sample_frequency)))

    seizure_signal = signal[original_start_index:original_stop_index]

    new_sample_frequency = int(parameters.loc['sampling_frequency']['value'])
    new_num_time_points = int(np.floor((stop - start) * new_sample_frequency))
    seizure_signal_resampled = resample(seizure_signal, new_num_time_points)

    return seizure_signal_resampled

def read_edfs_and_extract(edf_path, edf_start, edf_stop):

    f = pyedflib.EdfReader(edf_path)

    montage = str(parameters.loc['montage']['value'])
    montage_list = re.split(';', montage)
    signal_labels = f.getSignalLabels()
    x_data = []

    for i in montage_list:
        electrode_list = re.split('-', i)
        electrode_1 = electrode_list[0]
        extracted_signal_from_electrode_1 = extract_signal(f, signal_labels, electrode_name=electrode_1, start=edf_start, stop=edf_stop)
        electrode_2 = electrode_list[1]
        extracted_signal_from_electrode_2 = extract_signal(f, signal_labels, electrode_name=electrode_2, start=edf_start, stop=edf_stop)
        this_differential_output = extracted_signal_from_electrode_1-extracted_signal_from_electrode_2
        x_data.append(this_differential_output)

    f._close()
    del f

    x_data = np.array(x_data)

    return x_data

def load_edf_extract_seizures_v140(base_dir, save_data_dir, data_dict):

    seizure_data_dict = collections.defaultdict(list)

    count = 0
    bar = progressbar.ProgressBar(maxval=sum(len(v) for k, v in data_dict.items()),
                                  widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    for seizure_type, seizures in data_dict.items():
        for seizure in seizures:
            rel_file_location = seizure.filename.replace('.tse', '.edf').replace('./', '')
            patient_id = seizure.patient_id
            abs_file_location = os.path.join(base_dir,rel_file_location)
            temp = seizure_type_data(patient_id = patient_id, seizure_type = seizure_type, data = read_edfs_and_extract(abs_file_location, seizure.start_time, seizure.end_time))
            with open(os.path.join(save_data_dir, 'szr_' + str(count) + '_pid_' + patient_id + '_type_' + seizure_type + '.pkl'), 'wb') as fseiz:
                pickle.dump(temp, fseiz)
            count += 1
            bar.update(count)
    bar.finish()

    return seizure_data_dict


def load_edf_extract_seizures_v203(base_dir, save_data_dir, data_dict):
    seizure_data_dict = collections.defaultdict(list)
    count = 0
    bar = progressbar.ProgressBar(maxval=sum(len(v) for k, v in data_dict.items()),
                                  widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()

    for seizure_type, seizures in data_dict.items():
        for seizure in seizures:
            # Construct the search pattern based on the segment_id
            parts = seizure.segment_id.split('_')
            patient_id_idx = 3
            patient_id = parts[patient_id_idx]
            if len(patient_id) < 8:
                patient_id_idx += 1
                patient_id = parts[patient_id_idx]
            session = parts[patient_id_idx+1]
            segment = parts[patient_id_idx+2]
            data_type = seizure.data_type  # Ensure this is correctly populated from your data structure

            # Construct the search pattern to find the .edf file
            search_pattern = os.path.join(base_dir, data_type, patient_id, f'{session}*', '**', f'{patient_id}_{session}_{segment}.edf')
            edf_files = glob.glob(search_pattern, recursive=True)

            if not edf_files:
                import pdb; pdb.set_trace()
                print(f"No .edf file found for pattern: {search_pattern}")
                continue

            edf_file_path = edf_files[0]  # Assume the first match is correct

            try:
                data = read_edfs_and_extract(edf_file_path, seizure.start_time, seizure.end_time)
                temp = seizure_type_data(segment_id=seizure.segment_id, seizure_type=seizure_type, data=data)
                
                output_file_path = os.path.join(save_data_dir, f'szr_{count}_{seizure.segment_id}_{seizure_type}.pkl')
                with open(output_file_path, 'wb') as fseiz:
                    pickle.dump(temp, fseiz)
                
                count += 1
            except Exception as e:
                print(e)
                print(f"Error processing file: {edf_file_path}")

            bar.update(count)

    bar.finish()
    return seizure_data_dict


# to convert raw edf data into pkl format raw data
def gen_raw_seizure_pkl(args,tuh_eeg_szr_ver):
    base_dir = args.base_dir

    save_data_dir = os.path.join(args.save_data_dir, tuh_eeg_szr_ver, 'raw_seizures')
    if not os.path.exists(save_data_dir):
        os.makedirs(save_data_dir)

    #raw_data_base_dir = os.path.join(base_dir, tuh_eeg_szr_ver)
    #szr_annotation_file = os.path.join(raw_data_base_dir, '_DOCS', anno_file)
    # Define the path to the combined Excel files
    train_xlsx_file = os.path.join(base_dir, 'train', 'train_combined_data.xlsx')
    eval_xlsx_file = os.path.join(base_dir, 'eval', 'eval_combined_data.xlsx')
    dev_xlsx_file = os.path.join(base_dir, 'dev', 'dev_combined_data.xlsx') 

    # For training files
    print('Parsing the seizures of the training set...\n')
    train_data_dict = generate_data_dict(train_xlsx_file, 'train')
    print('Number of seizures by type in the training set...\n')
    print_type_information(train_data_dict)
    print('\n\n')

    # For dev files
    print('Parsing the seizures of the dev set...\n')
    dev_data_dict = generate_data_dict(dev_xlsx_file, 'dev')
    print_type_information(dev_data_dict)


     # For eval files
    print('Parsing the seizures of the eval set...\n')
    eval_data_dict = generate_data_dict(eval_xlsx_file, 'eval')
    print_type_information(eval_data_dict)
    print('\n\n')

    # Now we combine both
    print('Combining the training and validation set...\n')
    merged_dict = merge_train_test(dev_data_dict, train_data_dict)
    # merged_dict = merge_train_test(train_data_dict,dev_test_data_dict)
    print('Number of seizures by type in the combined set...\n')
    print_type_information(merged_dict)
    print('\n\n')

    # Process and extract the seizures from the EDF files
    if tuh_eeg_szr_ver == 'v2.0.3':
        print('Processing v2.0.3 data for combined training and dev datasets...\n')
        seizure_data_dict = load_edf_extract_seizures_v203(base_dir, save_data_dir, merged_dict)
        print_type_information(seizure_data_dict)
    else:
        exit(f'Unsupported version {tuh_eeg_szr_ver}')
    print('\n\n')


def main():
    parser = argparse.ArgumentParser(description='Build data for TUH EEG data')

    # Set default base directory based on the operating system
    default_base_dir = '/home/ebocini/repos/tuh-seizure-data'
    default_save_dir = '/home/ebocini/repos/tuh-seizure-data/processed_data'

    parser.add_argument('--base_dir', default=default_base_dir, help='path to raw seizure dataset')
    parser.add_argument('--save_data_dir', default=default_save_dir, help='path to save processed data')
    parser.add_argument('--tuh_eeg_szr_ver', choices=['v1.4.0', 'v1.5.2', 'v2.0.3'], default='v2.0.3', help='version of TUH seizure dataset')

    args = parser.parse_args()

    # Define the Excel file paths based on the provided base directory
    train_xlsx_file = os.path.join(args.base_dir, 'train', 'train_combined_data.xlsx')
    eval_xlsx_file = os.path.join(args.base_dir, 'eval', 'eval_combined_data.xlsx')

    # Call the function to process the raw seizure data
    gen_raw_seizure_pkl(args, args.tuh_eeg_szr_ver)


if __name__ == '__main__':
    main()
