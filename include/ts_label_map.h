#pragma once
#include "ts_types.h"
#include <string>
#include <map>
#include <vector>

namespace totalseg {

/// Label mapping: label index -> label name
using LabelMap = std::map<uint8_t, std::string>;

/// Get class map for a given task name (e.g., "total", "total_mr").
LabelMap get_class_map(const std::string& task_name);

/// Get the list of task IDs for a given task (e.g., total -> {291, 292, 293, 294, 295}).
std::vector<int> get_task_ids(const std::string& task_name);

/// Get the part name for a task ID (e.g., 291 -> "organs").
std::string get_part_name(int task_id);

/// Get total output channels (including background) for a task.
int get_task_num_classes(int task_id);

/// Merge multiple single-task label volumes into one multilabel volume.
/// Each task produces labels 1..N_classes; merge remaps them to the global label map.
LabelVolume merge_multilabel(const std::vector<LabelVolume>& parts,
                              const std::vector<int>& task_ids,
                              const std::string& task_name);

} // namespace totalseg
