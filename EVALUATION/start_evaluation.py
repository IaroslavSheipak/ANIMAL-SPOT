#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module: start_evaluation.py
Authors: Christian Bergler
Institution: Friedrich-Alexander-University Erlangen-Nuremberg, Department of Computer Science, Pattern Recognition Lab
Last Access: 27.12.2022
"""


import os
import re
import ast
import sys
import logging
import platform

from collections import OrderedDict


def get_dict_key_ignorecase(dict_in, key_t):
    dict_lc = dict()
    for key in dict_in:
        dict_lc[key.lower()] = dict_in.get(key)
    key_t = key_t.lower()
    if dict_lc.get(key_t) is not None:
        return dict_lc.get(key_t)
    else:
        return None


class setup_evaluator(object):

    def __init__(self, config, log_level="debug"):
        self.logger = None
        self.config_path = config
        self.config_data = dict()
        self.log_level = log_level
        self.loglevels = ["debug", "error", "warning", "info"]
        self.classes_label_idx = None
        self.classes_idx_label = None
        self.duration = None
        self.sequence_len = None
        self.hop = None
        self.data_opts = None
        self.needed_annotation_columns = ["Selection", "View", "Channel", "Begin time (s)", "End time (s)",
                                          "Low Freq (Hz)", "High Freq (Hz)", "Sound type", "Comments"]
        # New config options with defaults
        self.use_probability_priority = True
        self.min_annotation_duration = 0.1
        self.merge_gap = 0.15
    def init_logger(self):
        self.logger = logging.getLogger('training animal-spot')
        stream_handler = logging.StreamHandler()
        if self.log_level.lower() == self.loglevels[0]:
            self.logger.setLevel(logging.DEBUG)
            stream_handler.setLevel(logging.DEBUG)
        elif self.log_level.lower() == self.loglevels[1]:
            self.logger.setLevel(logging.ERROR)
            stream_handler.setLevel(logging.ERROR)
        elif self.log_level.lower() == self.loglevels[2]:
            self.logger.setLevel(logging.WARNING)
            stream_handler.setLevel(logging.WARNING)
        elif self.log_level.lower() == self.loglevels[3]:
            self.logger.setLevel(logging.INFO)
            stream_handler.setLevel(logging.INFO)
        else:
            self.logger.setLevel(logging.DEBUG)
            stream_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        stream_handler.setFormatter(formatter)
        self.logger.addHandler(stream_handler)
    
    def read_config(self):
        config_file = open(self.config_path, 'r')
        for element in config_file:
            if element[0] != "#" and element[0] != " " and element[0] != "" and element[0] != "\n":
                key = element.strip().split("=")[0]
                value = element.strip().split("=")[1]
                if value == "true":
                    self.config_data[key] = ""
                elif value == "false":
                    continue
                else:
                    self.config_data[key] = value

        # Parse new config options
        if "use_probability_priority" in self.config_data:
            self.use_probability_priority = True
        else:
            # Check if explicitly set to false in file
            config_file.seek(0)
            content = config_file.read()
            if "use_probability_priority=false" in content:
                self.use_probability_priority = False
            else:
                self.use_probability_priority = True  # Default

        if "min_annotation_duration" in self.config_data:
            try:
                self.min_annotation_duration = float(self.config_data["min_annotation_duration"])
            except ValueError:
                self.min_annotation_duration = 0.1

        if "merge_gap" in self.config_data:
            try:
                self.merge_gap = float(self.config_data["merge_gap"])
            except ValueError:
                self.merge_gap = 0.15

        config_file.close()
        self.logger.info("Config Data: " + str(self.config_data))
        self.logger.info(f"Use probability priority: {self.use_probability_priority}")
        self.logger.info(f"Min annotation duration: {self.min_annotation_duration}s")
        self.logger.info(f"Merge gap: {self.merge_gap}s")
                
    def read_prediction_file(self, path_prediction_file):
        time_prob = []
        prediction_file = open(path_prediction_file, "r")
        for line in prediction_file:
            print(line)
            if line.find("pred=") != -1 and line.find("prob=") != -1:
                if len(self.classes_idx_label) <= 2:
                    if line.strip().find("|INFO|") != -1:
                        self.logger.debug(line.strip().split("|INFO|")[1])
                        time_prob.append((line.strip().split("|INFO|")[1].split(",")[0].strip().split("time=")[1], line.strip().split("|INFO|")[1].split("prob=")[1].strip(), None))
                        self.duration = line.strip().split("|INFO|")[1].split(",")[0].strip().split("time=")[1].split("-")[1]
                    else:
                        self.logger.debug(line.strip().split("|I|")[1])
                        time_prob.append((line.strip().split("|I|")[1].split(",")[0].strip().split("time=")[1], line.strip().split("|I|")[1].split("prob=")[1].strip(), None))
                        self.duration = line.strip().split("|I|")[1].split(",")[0].strip().split("time=")[1].split("-")[1]
                else:
                    if line.strip().find("|INFO|") != -1:
                        self.logger.debug(line.strip().split("|INFO|")[1])
                        time_prob.append((line.strip().split("|INFO|")[1].split(",")[0].strip().split("time=")[1], line.strip().split("|INFO|")[1].split("prob=")[1].strip(), line.strip().split("|INFO|")[1].split("pred_class=")[1].strip().split(",")[0].strip().split("-")[1].strip()))
                        self.duration = line.strip().split("|INFO|")[1].split(",")[0].strip().split("time=")[1].split("-")[1]
                    else:
                        self.logger.debug(line.strip().split("|I|")[1])
                        time_prob.append((line.strip().split("|I|")[1].split(",")[0].strip().split("time=")[1], line.strip().split("|I|")[1].split("prob=")[1].strip(), line.strip().split("|I|")[1].split("pred_class=")[1].strip().split(",")[0].strip().split("-")[1].strip()))
                        self.duration = line.strip().split("|I|")[1].split(",")[0].strip().split("time=")[1].split("-")[1]
            elif line.find("prediction length") != -1 and line.find("hop length") != -1:
                self.sequence_len = line.split(",")[0].split(" ")[-1].split("[")[0]
                self.hop = line.split(",")[1].split(" ")[-1].split("[")[0]
            elif line.find("class distribution") != -1:
                self.classes_label_idx = ast.literal_eval(line.split("class distribution:")[-1].strip())
                self.classes_idx_label = dict(zip(self.classes_label_idx.values(), self.classes_label_idx.keys()))
            elif line.find("dataOpts") != -1:
                self.data_opts = ast.literal_eval(line.split("dataOpts:")[-1].strip())
        self.logger.info("duration of tape="+self.duration)
        self.logger.info("prediction file has been processed successfully")
        return time_prob

    def get_all_network_prediction_data(self, time_prob, threshold):
        threshold_depended_pred_data = OrderedDict()
        for time, prob, pred_class_idx in time_prob:
            if pred_class_idx is None:
                if float(prob) >= float(threshold):
                    threshold_depended_pred_data[time] = [1, str(prob)]
                    self.logger.debug("time="+time+", pred=1, prob="+str(prob)+"\n")
                else:
                    threshold_depended_pred_data[time]=[0, str(prob)]
                    self.logger.debug("time="+time+", pred=0, prob="+str(prob)+"\n")
            else:
                if float(prob) >= float(threshold):
                    threshold_depended_pred_data[time] = [pred_class_idx, str(prob)]
                    self.logger.debug("time="+time+", pred=1, prob="+str(prob)+"\n")
                else:
                    threshold_depended_pred_data[time] = [str(get_dict_key_ignorecase(self.classes_label_idx, "noise")), str(prob)]
                    self.logger.debug("time="+time+", pred=0, prob="+str(prob)+"\n")
        if self.use_probability_priority:
            labeled_data, duration = self.get_time_info(threshold_depended_pred_data)
        else:
            labeled_data, duration = self.get_time_info_legacy(threshold_depended_pred_data)
        return labeled_data, duration

    def overlapped_times_to_pred(self, time_info, prediction_data):
        overlapped_pred_data = OrderedDict()
        for timestamp in prediction_data:
            overlapped_pred_data[timestamp] = [0, prediction_data.get(timestamp)[1]]
        for time, label in time_info:
            timestamps = self.generate_timestamps(time[0], time[1], round(float(self.sequence_len), 3), round(float(self.hop), 3))
            for update_timestamp in timestamps:
                overlapped_pred_data[update_timestamp] = [1, overlapped_pred_data.get(update_timestamp)[1]]
        return overlapped_pred_data

    def generate_timestamps(self, start, end, sequence_len, hop_size, human=False):
        if round(start, 3) == round(end, 3):
            return []
        else:
            if human:
                start = start-sequence_len+hop_size
                end = end+sequence_len
                if start < 0:
                    start = 0
                if end > round(float(self.duration), 3):
                    end = round(float(self.duration), 3)
            tmp_end=0.0
            timestamps = []
            while round(tmp_end,3) < round(end, 3):
                tmp_end=start+sequence_len
                if round(tmp_end,3) > round(float(self.duration), 3):
                    timestamps.append(str(round(float(self.duration)-sequence_len, 3))+"-"+str(round(float(self.duration), 3)))
                    return timestamps
                timestamps.append(str(round(start, 3))+"-"+str(round(tmp_end, 3)))
                start+=hop_size
            return timestamps

    def get_time_info(self, prediction_data):
        """
        Convert prediction data to time annotations, resolving overlapping chunks
        by selecting the prediction with the HIGHEST PROBABILITY (not earliest time).

        This fixes the issue where earlier chunks would incorrectly override
        later chunks with higher confidence scores.
        """
        time_info = []
        duration = float(list(prediction_data.keys())[-1].split("-")[1])
        noise_label = str(get_dict_key_ignorecase(self.classes_label_idx, "noise"))

        # Step 1: Build a list of all chunks with their times, predictions, and probabilities
        chunks = []
        for timestamp in prediction_data:
            time = timestamp.replace("-", " ").split(" ")
            start_time = float(time[0])
            end_time = float(time[1])
            pred = prediction_data.get(timestamp)[0]
            prob = float(prediction_data.get(timestamp)[1])

            # Skip noise predictions
            if str(pred) != noise_label:
                chunks.append({
                    'start': round(start_time, 3),
                    'end': round(end_time, 3),
                    'pred': pred,
                    'prob': prob
                })

        if not chunks:
            return time_info, duration

        # Step 2: Create a timeline with the best prediction for each time point
        # Use a fine-grained approach: sample at hop intervals
        hop = float(self.hop) if self.hop else 0.1
        time_resolution = min(hop / 2, 0.05)  # Use half the hop or 50ms, whichever is smaller

        # Create time points
        max_time = max(c['end'] for c in chunks)
        min_time = min(c['start'] for c in chunks)

        # For each time point, find the chunk with highest probability that covers it
        current_start = None
        current_pred = None
        current_end = None

        t = min_time
        while t <= max_time:
            # Find all chunks that cover this time point
            covering_chunks = [c for c in chunks if c['start'] <= t < c['end']]

            if covering_chunks:
                # Select the chunk with highest probability
                best_chunk = max(covering_chunks, key=lambda c: c['prob'])
                best_pred = best_chunk['pred']

                if current_pred is None:
                    # Start a new annotation
                    current_start = t
                    current_pred = best_pred
                    current_end = t + time_resolution
                elif str(best_pred) == str(current_pred):
                    # Extend current annotation
                    current_end = t + time_resolution
                else:
                    # Different prediction - save current and start new
                    time_info.append(([round(current_start, 3), round(current_end, 3)], current_pred))
                    current_start = t
                    current_pred = best_pred
                    current_end = t + time_resolution
            else:
                # No chunk covers this point - save current annotation if exists
                if current_pred is not None:
                    time_info.append(([round(current_start, 3), round(current_end, 3)], current_pred))
                    current_pred = None
                    current_start = None
                    current_end = None

            t += time_resolution

        # Don't forget the last annotation
        if current_pred is not None:
            time_info.append(([round(current_start, 3), round(current_end, 3)], current_pred))

        # Step 3: Filter by minimum duration
        if self.min_annotation_duration > 0:
            time_info = [
                ann for ann in time_info
                if (ann[0][1] - ann[0][0]) >= self.min_annotation_duration
            ]
            self.logger.debug(f"After min duration filter: {len(time_info)} annotations")

        # Step 4: Merge adjacent annotations with the same prediction
        merge_threshold = self.merge_gap if self.merge_gap else float(self.hop if self.hop else 0.1)
        if time_info:
            merged_info = [time_info[0]]
            for i in range(1, len(time_info)):
                prev = merged_info[-1]
                curr = time_info[i]
                # If same prediction and close in time (within merge_gap), merge
                if str(prev[1]) == str(curr[1]) and (curr[0][0] - prev[0][1]) < merge_threshold:
                    merged_info[-1] = ([prev[0][0], curr[0][1]], prev[1])
                else:
                    merged_info.append(curr)
            time_info = merged_info

        self.logger.info(f"Created {len(time_info)} annotations using probability-based selection")
        return time_info, duration

    def get_time_info_legacy(self, prediction_data):
        """
        Legacy method for converting prediction data to time annotations.
        This uses the original behavior where earlier chunks take priority
        over later chunks, regardless of probability.

        Kept for backwards compatibility - use get_time_info() for improved behavior.
        """
        time_info = []
        first_el = True
        duration = float(list(prediction_data.keys())[-1].split("-")[1])
        for timestamp in prediction_data:
            time = timestamp.replace("-", " ").split(" ")
            time = [float(item) for item in time]
            pred = prediction_data.get(timestamp)[0]
            if first_el:
                first_el = False
            if str(pred) != str(get_dict_key_ignorecase(self.classes_label_idx, "noise")):
                if time_info:
                    if round(time_info[-1][0][1], 3) >= round(time[0], 3) and (str(time_info[-1][1]) == str(pred)):
                        time_info[-1] = (([round(time_info[-1][0][0], 3), round(time[1], 3)], pred))
                    elif round(time_info[-1][0][1], 3) >= round(time[0], 3) and (str(time_info[-1][1]) != str(pred)):
                        time_info.append(([time_info[-1][0][1], round(time[1], 3)], pred))
                    else:
                        time_info.append(([round(time[0], 3), round(time[1], 3)], pred))
                else:
                    time_info.append(([round(time[0], 3), round(time[1], 3)], pred))

        self.logger.info(f"Created {len(time_info)} annotations using legacy (time-based) selection")
        return time_info, duration

    def get_time_info_former(self, prediction_data):
        end = 0.0
        start = 0.0
        target = str(get_dict_key_ignorecase(self.classes_label_idx, "noise"))
        time_info = []
        first_el = True
        pred = None
        duration = float(list(prediction_data.keys())[-1].split("-")[1])
        for timestamp in prediction_data:
            time = timestamp.replace("-", " ").split(" ")
            prev_pred = pred
            pred = prediction_data.get(timestamp)[0]
            if first_el:
                prev_pred = pred
                first_el = False
            if (str(pred) != str(get_dict_key_ignorecase(self.classes_label_idx, "noise")) and ((str(pred) == str(prev_pred)))) or (str(prev_pred) == str(
                        get_dict_key_ignorecase(self.classes_label_idx, "noise"))) :
                if target == str(get_dict_key_ignorecase(self.classes_label_idx, "noise")):
                    start = float(time[0])
                    end = float(time[1])
                    target = pred
                else:
                    end += float(self.hop)
            else:
                if str(target) != str(get_dict_key_ignorecase(self.classes_label_idx, "noise")):
                    if time_info:
                        if round(time_info[-1][0][1], 3) >= round(start, 3) and (str(time_info[-1][1]) == target):
                            time_info[-1] = (([round(time_info[-1][0][0], 3), round(end, 3)], target))
                        elif round(time_info[-1][0][1], 3) >= round(start, 3) and (str(time_info[-1][1]) != target):
                            time_info.append(([time_info[-1][0][1], round(end, 3)], target))
                        else:
                            time_info.append(([round(start, 3), round(end, 3)], target))

                        if str(pred) != str(get_dict_key_ignorecase(self.classes_label_idx, "noise")):
                            time_info.append(([time_info[-1][0][1], round(float(time[-1]), 3)], pred))

                    else:
                        time_info.append(([round(start, 3), round(end, 3)], target))
                    target = str(get_dict_key_ignorecase(self.classes_label_idx, "noise"))
                    
        if str(target) != str(get_dict_key_ignorecase(self.classes_label_idx, "noise")):
            if time_info:
                if round(time_info[-1][0][1], 3) >= round(start, 3):
                    time_info[-1] = (([round(time_info[-1][0][0], 3), round(end, 3)], target))
                else:
                    time_info.append(([round(start, 3), round(end, 3)], target))
            else:
                time_info.append(([round(start, 3), round(end, 3)], target))
        return time_info, duration

    def get_time_info_orig(self, prediction_data):
        end = 0.0
        start = 0.0
        target = False
        time_info = []
        duration = float(list(prediction_data.keys())[-1].split("-")[1])
        for timestamp in prediction_data:
            time = timestamp.replace("-", " ").split(" ")
            pred = prediction_data.get(timestamp)[0]
            if pred:
                if not target:
                    start = float(time[0])
                    end = float(time[1])
                    target = True
                else:
                    end+=float(self.hop)
            else:
                if target:
                    if time_info:
                        if round(time_info[-1][0][1],3) >= round(start,3):
                            time_info[-1] = (([round(time_info[-1][0][0],3), round(end,3)], 1))
                        else:
                            time_info.append(([round(start,3), round(end,3)], 1))
                    else:
                        time_info.append(([round(start,3), round(end,3)], 1))
                    target = False
        if target:
            if time_info:
                if round(time_info[-1][0][1],3) >= round(start,3):
                    time_info[-1] = (([round(time_info[-1][0][0],3), round(end,3)], 1))
                else:
                    time_info.append(([round(start,3), round(end,3)], 1))
            else:
                time_info.append(([round(start,3), round(end,3)], 1))
        return time_info, duration

    def add_noise_parts_to_time_info(self, time_info, duration):
        noise_time_info = []
        if not time_info:
            noise_time_info.append(([0.0, duration], get_dict_key_ignorecase(self.classes_label_idx, "noise")))
            return noise_time_info
        else:
            for index in range(len(time_info)):
                if index < len(time_info)-1:
                    noise_time_info.append(time_info[index])
                    if time_info[index][0][1] != time_info[index+1][0][0]:
                        noise_time_info.append(([time_info[index][0][1], time_info[index+1][0][0]], get_dict_key_ignorecase(self.classes_label_idx, "noise")))
                else:
                    noise_time_info.append(time_info[index])
            if noise_time_info[0][0][0] != 0.0:
                noise_time_info.insert(0, ([0.0, noise_time_info[0][0][0]], get_dict_key_ignorecase(self.classes_label_idx, "noise")))
            if noise_time_info[-1][0][1] != duration:
                noise_time_info.append(([noise_time_info[-1][0][1], duration], get_dict_key_ignorecase(self.classes_label_idx, "noise")))
            return noise_time_info

    def get_overlapped_threshold_dependend_annotation_data(self, labeled_data, duration):
        threshold_depended_annotation_data_overlapped = []
        noise_time_info = self.add_noise_parts_to_time_info(labeled_data, duration)
        for index in range(len(noise_time_info)):
            threshold_depended_annotation_data_overlapped.append((self.classes_idx_label.get(int(noise_time_info[index][1])), -1, noise_time_info[index][0][0], noise_time_info[index][0][1], False))
        return threshold_depended_annotation_data_overlapped

    def write_annotation_file(self, path_annotation_file, annotation_data):
        iterator = 1
        annotated_prediction_file = open(path_annotation_file, 'w')
        annotated_prediction_file.write("\t".join(self.needed_annotation_columns)+"\n")
        for label, __, start, end, __ in annotation_data:
            if label == "noise" and self.config_data.get("noise_in_anno") == None:
                continue
            else:
                annotated_prediction_file.write(str(iterator)+"\tSpectrogram_1\t1\t"+str(start)+"\t"+str(end)+"\t"+str(self.data_opts.get("fmin"))+"\t"+str(self.data_opts.get("fmax"))+"\t"+label+"\t \n")
            iterator += 1
        annotated_prediction_file.close()

    def list_all_files_in_dir(self, path):
        onlyfiles = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        return onlyfiles

    def process(self):
        path = self.config_data.get("prediction_dir")
        prediction_files = self.list_all_files_in_dir(path)
        for pred_file in prediction_files:
            if pred_file.find("_predict_output") != -1:
                time_prob = self.read_prediction_file(path+"/"+pred_file)
                labeled_data, duration = self.get_all_network_prediction_data(time_prob, self.config_data.get("threshold"))
                threshold_depended_annotation_data_overlapped = self.get_overlapped_threshold_dependend_annotation_data(labeled_data, duration)
                if platform.system() == "Windows":
                    self.write_annotation_file(self.config_data["output_dir"] + "\\" + pred_file + ".annotation.result.txt", threshold_depended_annotation_data_overlapped)
                else:
                    self.write_annotation_file(self.config_data["output_dir"] + "/" + pred_file + ".annotation.result.txt", threshold_depended_annotation_data_overlapped)
                self.logger.info("annotation file=" + str() + " has been successfully created!")

if __name__ == '__main__':
    if len(sys.argv) == 2:
        evaluator = setup_evaluator(config=sys.argv[1])
        evaluator.init_logger()
        evaluator.read_config()
        evaluator.process()
    else:
        raise Exception('Invalid Number of Cmd Parameter. Only one argument -> path of config file')
