# <Copyright 2019, Argo AI, LLC. Released under the MIT license.>
import argparse
import glob
import json
import logging
import os
import pathlib
import pickle
import sys
from typing import Any, Dict, List, TextIO, Tuple, Union

import motmetrics as mm
import numpy as np
from shapely.geometry.polygon import Polygon

from argoverse.evaluation.eval_utils import get_pc_inside_bbox, label_to_bbox, leave_only_roi_region
from argoverse.utils.json_utils import read_json_file
from argoverse.utils.ply_loader import load_ply
from argoverse.utils.se3 import SE3
from argoverse.utils.transform import quat2rotmat
from collections import defaultdict


import os
import sys

import numpy as np
from transform_utils import convert_3dbox_to_8corner
from iou_utils import iou3d
from sklearn.utils.linear_assignment_ import linear_assignment

from pyquaternion import Quaternion

# import argparse

tracking_names = ["VEHICLE", "PEDESTRIAN"]


def rotation_to_positive_z_angle(rotation):
  q = Quaternion(rotation)
  angle = q.angle if q.axis[2] > 0 else -q.angle
  return angle


def get_mean():
	# 1. read tracked files from ground truth
	# 2. extract x, y, z, h, w, l, rot -> angle

	path_datasets = glob.glob(os.path.join(args.path_dataset_root, "*"))
	

	gt_trajectory_map = {tracking_name: defaultdict(dict) for tracking_name in tracking_names}

	# store every detection data to compute mean and variance
	
	gt_box_data = {tracking_name: [] for tracking_name in tracking_names}
	
	for path_dataset in path_datasets:  # path_tracker_output, path_dataset in zip(path_tracker_outputs, path_datasets):

		log_id = pathlib.Path(path_dataset).name
		if len(log_id) == 0 or log_id.startswith("_"):
			continue

		# path_tracker_output = os.path.join(path_tracker_output_root, log_id)

		# print('\npath_tracker_output: ', path_tracker_output) # read tracked log files ALL, worked
		# /media/basic/Transcend/argoai_tracking/argoverse-tracking/val_output/val-split-track-preds-maxage15-minhits5-conf0.3/033669d3-3d6b-3d3d-bd93-7985d86653ea


		path_track_data = sorted(
			glob.glob(os.path.join(os.fspath(path_dataset), \
				"per_sweep_annotations_amodal", "*"))
		)

		# logger.info("log_id = %s", log_id)

		# city_info_fpath = f"{path_dataset}/city_info.json"
		# city_info = read_json_file(city_info_fpath)
		# city_name = city_info["city_name"]
		# logger.info("city name = %s", city_name)

		for ind_frame in range(len(path_track_data)):

			if ind_frame % 50 == 0:
				# print("%d/%d" % (ind_frame, len(path_track_data)))
				print("%d/%d" % (ind_frame, len(path_track_data)))

			timestamp_lidar = int(path_track_data[ind_frame].split("/")[-1].split("_")[-1].split(".")[0])
			

			# print('\ntimestamp: ', timestamp_lidar)

			path_gt = os.path.join(
				path_dataset, "per_sweep_annotations_amodal", f"tracked_object_labels_{timestamp_lidar}.json"
			)

			# print('\npath_dataset gt: ', path_gt) # corrected for reading val, all log files
			#/media/basic/Transcend/argoai_tracking/argoverse-tracking/val/00c561b9-2057-358d-82c6-5b06d76cebcf/per_sweep_annotations_amodal/tracked_object_labels_315969629019741000.json

			# gt_data = read_json_file(path_gt)
			
		   


			if not os.path.exists(path_gt):
				# logger.warning("Missing ", path_gt)
				continue

			gt_data = read_json_file(path_gt)

			for i in range(len(gt_data)):

				if gt_data[i]["label_class"] not in tracking_names:
					print('\nignored: ', gt_data[i]["label_class"])

					continue

				bbox, orientation = label_to_bbox(gt_data[i])

				x, y, z = gt_data[i]["center"]["x"], \
					gt_data[i]["center"]["y"], \
					gt_data[i]["center"]["z"]
	
				center = np.array([gt_data[i]["center"]["x"], gt_data[i]["center"]["y"], gt_data[i]["center"]["z"]])

				w = gt_data[i]['width']
				l = gt_data[i]["length"]
				h = gt_data[i]["height"]

				rotation = [gt_data[i]['rotation']['x'],
							gt_data[i]['rotation']['y'],
							gt_data[i]['rotation']['z'],
							gt_data[i]['rotation']['w']]

				# print('\nx,y,z, w, l, h: ', x, y, z, w, l, h)

				z_angle = rotation_to_positive_z_angle(rotation)

				# print('\nz_angle: ', z_angle)

				box_data = np.array([					
					h, w, l,
					x, y, z,
					z_angle,
					0, 0, 0, 0 # x_dot, y_dot, z_dot, a_dot
					])

				# print('\nbox_data: ', box_data)
				track_label_uuid = gt_data[i]["track_label_uuid"]
				cat = gt_data[i]["label_class"]
				gt_trajectory_map[cat][track_label_uuid][ind_frame] = box_data


				# compute x_dot, y_dot, z_dot
				# if we can find the same object in the previous frame, get the velocity
				if track_label_uuid in gt_trajectory_map[cat] \
					and ind_frame - 1 \
						in gt_trajectory_map[cat][track_label_uuid]:

					residual_vel = box_data[3:7] - \
						gt_trajectory_map[cat][track_label_uuid][ind_frame-1][3:7]
					
					box_data[7:11] = residual_vel

					gt_trajectory_map[cat][track_label_uuid][ind_frame] = box_data

					# back fill
					if gt_trajectory_map[cat][track_label_uuid][ind_frame-1][7] == 0:
						gt_trajectory_map[cat][track_label_uuid][ind_frame-1][7:11] = residual_vel


				gt_box_data[gt_data[i]["label_class"]].append(box_data)

	gt_box_data = {tracking_name: np.stack(gt_box_data[tracking_name], axis=0) for tracking_name in tracking_names}

	mean = {tracking_name: np.mean(gt_box_data[tracking_name], axis=0) for tracking_name in tracking_names}
	std = {tracking_name: np.std(gt_box_data[tracking_name], axis=0) for tracking_name in tracking_names}
	var = {tracking_name: np.var(gt_box_data[tracking_name], axis=0) for tracking_name in tracking_names}
	print('\nh, w, l, x, y, z, a, x_dot, y_dot, z_dot, a_dot\n')
	print('\nmean: ', mean, '\n'
		'\nstd: ', std, '\n',
		'\nvar: ', var) #Q



	# return mean, std, var #Q

# for R, H
def matching_and_get_diff_stats():
	
	# gt_path = args.path_dataset_root
	# pr_path = args.path_tracker_output

	tracking_names = ["VEHICLE", "PEDESTRIAN"]

	diff = {tracking_name: [] for tracking_name in tracking_names} # [h, w, l, x, y, z, a]
	diff_vel = {tracking_name: [] for tracking_name in tracking_names} # [x_dot, y_dot, z_dot, a_dot]
	match_diff_t_map = {tracking_name: {} for tracking_name in tracking_names}

	# similar to main.py class AB3DMOT update()
	reorder = [3, 4, 5, 6, 2, 1, 0]
	reorder_back = [6, 5, 4, 0, 1, 2, 3]

	# gt_all = {tracking_name: defaultdict(dict) for tracking_name in tracking_names}
	# pr_all = {tracking_name: defaultdict(dict) for tracking_name in tracking_names}
	
	gt_trajectory_map = {tracking_name: defaultdict(dict) for tracking_name in tracking_names}
	pr_trajectory_map = {tracking_name: defaultdict(dict) for tracking_name in tracking_names}
	gts_ids = list()
	prs_ids = list()
	tmp_prs = list()
	tmp_gts = list()
	path_datasets = glob.glob(os.path.join(args.path_dataset_root, "*"))
	
	for path_dataset in path_datasets:  # path_tracker_output, path_dataset in zip(path_tracker_outputs, path_datasets):

		log_id = pathlib.Path(path_dataset).name
		if len(log_id) == 0 or log_id.startswith("_"):
			continue

		# path_tracker_output = os.path.join(path_tracker_output_root, log_id)

		# print('\npath_tracker_output: ', path_tracker_output) # read tracked log files ALL, worked
		# /media/basic/Transcend/argoai_tracking/argoverse-tracking/val_output/val-split-track-preds-maxage15-minhits5-conf0.3/033669d3-3d6b-3d3d-bd93-7985d86653ea

		print('\npath_dataset: ', path_dataset)

		path_track_data = sorted(
			glob.glob(os.path.join(os.fspath(path_dataset), \
				"per_sweep_annotations_amodal", "*"))
		)


		# iterate each *.json in each log_id
		for ind_frame in range(len(path_track_data)):

			if ind_frame % 50 == 0:
				# print("%d/%d" % (ind_frame, len(path_track_data)))
				print("%d/%d" % (ind_frame, len(path_track_data)))

			timestamp_lidar = int(path_track_data[ind_frame].split("/")[-1].split("_")[-1].split(".")[0])
			

			# print('\ntimestamp: ', timestamp_lidar)

			# path of each json of gt
			path_gt = os.path.join(
				path_dataset, "per_sweep_annotations_amodal", f"tracked_object_labels_{timestamp_lidar}.json"
			)

			# print('\npath_dataset gt: ', path_gt) # corrected for reading val, all log files
			#/media/basic/Transcend/argoai_tracking/argoverse-tracking/val/00c561b9-2057-358d-82c6-5b06d76cebcf/per_sweep_annotations_amodal/tracked_object_labels_315969629019741000.json

			# gt_data = read_json_file(path_gt)
					  
			if not os.path.exists(path_gt):
				# logger.warning("Missing ", path_gt)
				continue

			gt_data = read_json_file(path_gt)

			# get data from gt
			for i in range(len(gt_data)):

				if gt_data[i]["label_class"] not in tracking_names:
					# print('\nGT ignored: ', gt_data[i]["label_class"])

					continue

				bbox, orientation = label_to_bbox(gt_data[i])

				x, y, z = gt_data[i]["center"]["x"], \
					gt_data[i]["center"]["y"], \
					gt_data[i]["center"]["z"]
	
				center = np.array([gt_data[i]["center"]["x"], gt_data[i]["center"]["y"], gt_data[i]["center"]["z"]])

				w = gt_data[i]['width']
				l = gt_data[i]["length"]
				h = gt_data[i]["height"]

				rotation = [gt_data[i]['rotation']['x'],
							gt_data[i]['rotation']['y'],
							gt_data[i]['rotation']['z'],
							gt_data[i]['rotation']['w']]

				# print('\nx,y,z, w, l, h: ', x, y, z, w, l, h)

				z_angle = rotation_to_positive_z_angle(rotation)

				# print('\nz_angle: ', z_angle)

				box_data = np.array([					
					h, w, l,
					x, y, z,
					z_angle
					# 0, 0, 0, 0 # x_dot, y_dot, z_dot, a_dot
					])

				# print('\nbox_data: ', box_data)
				track_label_uuid = gt_data[i]["track_label_uuid"]
				cat = gt_data[i]["label_class"]

				# get all gt

				# print('\n', 'GT '*20, '\n')
				# print('\ncat: ', cat)
				# print('\nlog_id: ', log_id)
				# print('\ntrack_label_uuid: ', track_label_uuid)
				# print('\nind_frame: ', ind_frame)

				# gt_trajectory_map[cat][log_id][track_label_uuid][ind_frame] = box_data
				gt_trajectory_map[cat][track_label_uuid][ind_frame] = box_data
				tmp_gts.append(box_data)
				# gts = np.stack([np.array([box_data])], axis=0)

				gts_ids.append(track_label_uuid) 
	gts = np.stack(tmp_gts, axis=0)


	# here to get preds
	path_datasets_output = glob.glob(os.path.join(args.path_tracker_output, "*"))
	
	for path_dataset_trck in path_datasets_output: 
			# get tracked data

		log_id = pathlib.Path(path_dataset_trck).name
		if len(log_id) == 0 or log_id.startswith("_"):
			continue
		path_tracker_output = os.path.join(args.path_tracker_output, log_id)

		print('\npath_tracker_output: ', path_tracker_output) # read tracked log files ALL, worked
		# /media/basic/Transcend/argoai_tracking/argoverse-tracking/val_output/val-split-track-preds-maxage15-minhits5-conf0.3/033669d3-3d6b-3d3d-bd93-7985d86653ea


		path_track_data = sorted(
			glob.glob(os.path.join(os.fspath(path_tracker_output), \
				"per_sweep_annotations_amodal", "*"))
		)

		print("log_id = %s", log_id)



		for ind_frame_track in range(len(path_track_data)):

			if ind_frame_track % 50 == 0:
				# print("%d/%d" % (ind_frame_track, len(path_track_data)))
				print("%d/%d" % (ind_frame_track, len(path_track_data)))

			timestamp_lidar = int(path_track_data[ind_frame_track].split("/")[-1].split("_")[-1].split(".")[0])
			
			path_pr = os.path.join(
				path_dataset_trck, "per_sweep_annotations_amodal", f"tracked_object_labels_{timestamp_lidar}.json"
			)

			# print('\npath_dataset gt: ', path_pr) # corrected for reading val, all log files
			#/media/basic/Transcend/argoai_tracking/argoverse-tracking/val/00c561b9-2057-358d-82c6-5b06d76cebcf/per_sweep_annotations_amodal/tracked_object_labels_315969629019741000.json
			
			if not os.path.exists(path_pr):
				print("Missing pr", path_pr)
				continue

			pr_data = read_json_file(path_pr)

			# get data from pr
			for i in range(len(pr_data)):

				if pr_data[i]["label_class"] not in tracking_names:
					# print('\nPR ignored: ', pr_data[i]["label_class"])

					continue

				bbox, orientation = label_to_bbox(pr_data[i])

				x, y, z = pr_data[i]["center"]["x"], \
					pr_data[i]["center"]["y"], \
					pr_data[i]["center"]["z"]
	
				center = np.array([pr_data[i]["center"]["x"], pr_data[i]["center"]["y"], pr_data[i]["center"]["z"]])

				w = pr_data[i]['width']
				l = pr_data[i]["length"]
				h = pr_data[i]["height"]

				rotation = [pr_data[i]['rotation']['x'],
							pr_data[i]['rotation']['y'],
							pr_data[i]['rotation']['z'],
							pr_data[i]['rotation']['w']]

				# print('\nx,y,z, w, l, h: ', x, y, z, w, l, h)

				z_angle = rotation_to_positive_z_angle(rotation)

				# print('\nz_angle: ', z_angle)

				box_data = np.array([					
					h, w, l,
					x, y, z,
					z_angle
					# 0, 0, 0, 0 # x_dot, y_dot, z_dot, a_dot
					])

				# print('\nbox_data: ', box_data)
				track_label_uuid = pr_data[i]["track_label_uuid"]
				cat = pr_data[i]["label_class"]

				# print('\n', 'PR '*20, '\n')
				# print('\ncat: ', cat)
				# print('\nlog_id: ', log_id)
				# print('\ntrack_label_uuid: ', track_label_uuid)
				# print('\nind_frame: ', ind_frame_track)

				# get all gt
				# pr_trajectory_map[cat][log_id][track_label_uuid][ind_frame_track] = box_data
				pr_trajectory_map[cat][track_label_uuid][ind_frame_track] = box_data

				tmp_prs.append(box_data)
				# prs = np.stack([np.array([box_data])], axis=0)

				prs_ids.append(track_label_uuid) 

	prs = np.stack(tmp_prs, axis=0)


	prs = prs[101:1000, reorder]
	gts = gts[101:1000, reorder]

	# if matching_dist == '3d_iou':


	dets_8corner = [convert_3dbox_to_8corner(det_tmp) for det_tmp in prs]
	gts_8corner = [convert_3dbox_to_8corner(gt_tmp) for gt_tmp in gts]
	print('\n Computing distance matrix...')
	print('\n dets len: ', len(dets_8corner))
	print('\n gts_8corner len: ', len(gts_8corner))


	iou_matrix = np.zeros((len(dets_8corner),len(gts_8corner)),dtype=np.float32)

	for d,det in enumerate(dets_8corner):
		for g,gt in enumerate(gts_8corner):
			iou_matrix[d,g] = iou3d(det,gt)[0]

	#print('iou_matrix: ', iou_matrix)
	distance_matrix = -iou_matrix
	threshold = 0.1

	print('\n linear_assignment...')

	matched_indices = linear_assignment(distance_matrix)


	#print('matched_indices: ', matched_indices)
	prs = prs[:, reorder_back]
	gts = gts[:, reorder_back]

	print('\n linear_assignment...DONE')


	# loop each category/tracking_name
	fl = False
	for tracking_name in tracking_names:
		# get pair id
		for pair_id in range(matched_indices.shape[0]):

			# compute diff_values
			if distance_matrix[matched_indices[pair_id][0]][matched_indices[pair_id][1]] \
			< threshold:

				print('\n Computing diff_value...')
				diff_value = prs[matched_indices[pair_id][0]] - gts[matched_indices[pair_id][1]]
				diff[tracking_name].append(diff_value)

				gt_track_id = gts_ids[matched_indices[pair_id][1]]


				# update match_diff_t_map
				if ind_frame not in match_diff_t_map[tracking_name]:
					match_diff_t_map[tracking_name][ind_frame] = {gt_track_id: diff_value}

				else:
				  	match_diff_t_map[tracking_name][ind_frame][gt_track_id] = diff_value
			# check if we have previous time_step's matching pair for current gt object
			#print('t: ', t)
			#print('len(match_diff_t_map): ', len(match_diff_t_map))


				# compute diff_vel
				try:
					if ind_frame > 0 and ind_frame-1 in match_diff_t_map[tracking_name] \
					  and gt_track_id in match_diff_t_map[tracking_name][ind_frame-1]:

						diff_vel_value = diff_value - \
						  match_diff_t_map[tracking_name][ind_frame-1][gt_track_id]
						diff_vel[tracking_name].append(diff_vel_value)
				except ValueError:
					fl = True



	diff = {tracking_name: np.stack(diff[tracking_name], axis=0) for tracking_name in tracking_names}
	mean = {tracking_name: np.mean(diff[tracking_name], axis=0) for tracking_name in tracking_names}
	std = {tracking_name: np.std(diff[tracking_name], axis=0) for tracking_name in tracking_names}
	var = {tracking_name: np.var(diff[tracking_name], axis=0) for tracking_name in tracking_names}
	print('Diff: Global coordinate system')
	print('h, w, l, x, y, z, a\n')
	print('mean: ', mean)
	print('std: ', std)
	print('var: ', var)
	if not fl:
		diff_vel = {tracking_name: np.stack(diff_vel[tracking_name], axis=0) for tracking_name in tracking_names}
		mean_vel = {tracking_name: np.mean(diff_vel[tracking_name], axis=0) for tracking_name in tracking_names}
		std_vel = {tracking_name: np.std(diff_vel[tracking_name], axis=0) for tracking_name in tracking_names}
		var_vel = {tracking_name: np.var(diff_vel[tracking_name], axis=0) for tracking_name in tracking_names}
		print('Diff: Global coordinate system')
		print('h, w, l, x, y, z, a\n')
		print('mean: ', mean)
		print('std: ', std)
		print('var: ', var)
		print('\nh_dot, w_dot, l_dot, x_dot, y_dot, z_dot, a_dot\n')
		print('mean_vel: ', mean_vel)
		print('std_vel: ', std_vel)
		print('var_vel: ', var_vel)
	
	else:
		print('Diff: Global coordinate system')
		print('h, w, l, x, y, z, a\n')
		print('mean: ', mean)
		print('std: ', std)
		print('var: ', var)
		print(mean, std, var)

	# return mean, std, var, mean_vel, std_vel, var_vel #R

if __name__ == '__main__':
	# Settings.

	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"--path_dataset_root",
		type=str,
		default="/media/basic/Transcend/argoai_tracking/argoverse-tracking/val/",
	)
	parser.add_argument(
		"--path_tracker_output", 
		type=str, 
		default="/media/basic/Transcend/argoai_tracking/argoverse-tracking/val_output/val-split-track-preds-maxage15-minhits5-conf0.3/"
	)

	parser.add_argument("--category", type=str, default="VEHICLE", required=False)
	args = parser.parse_args()

	# get_mean()

	# # for observation noise covariance R

	matching_and_get_diff_stats()


