import re
from typing import Any, NamedTuple

def is_real_loss_or_metric(name):
  return "real" in name

def is_pipeline_related_loss_or_metric(name):
  return "real" in name or "edge" in name or "seq" in name

def get_element_from_log_name(name, element):
  # name structure: {train_or_validation}p_{partition_id}_m_{micro_batch_id}_{real/edge/seq}_output_{output_id}_{metric_name}
  # e.g.: `p_1_m_1_real_output_0_sparse_categorical_accuracy` or `val_p_1_m_1_real_output_0_sparse_categorical_accuracy`
  assert element in ["train_or_validation", "micro_batch_id", "", "output_id", "metric_name"]

  m = re.match("(?P<train_or_validation>(val_)?)p_(?P<partition_id>.+)_m_(?P<micro_batch_id>.+)_(?P<type>.+)_output_(?P<output_id>\d+)_(?P<metric_name>.+)", name)
  return m.groupdict().get(element, None)

class OutputInfo(NamedTuple):
  is_validation: bool
  output_id: str

def remove_user_visible_metrics(metrics_name_and_info):
  remaining_metrics : dict[str, Any] = {}
  for name, info in metrics_name_and_info.items():
    if not is_pipeline_related_loss_or_metric(name):
      remaining_metrics[name] = info
  return remaining_metrics

def extract_user_visible_metrics(metrics_name_and_info):
  metrics_per_output : dict[OutputInfo, dict] = {}
  output_ids : list[str] = []

  for name, info in metrics_name_and_info.items():
    if not is_real_loss_or_metric(name):
      continue
    is_validation = len(get_element_from_log_name(name, "train_or_validation")) > 0
    output_id = get_element_from_log_name(name, "output_id")
    metric_name = get_element_from_log_name(name, "metric_name")

    output_ids.append(output_id)
    output_info = OutputInfo(is_validation, output_id)
    if output_info not in metrics_per_output.keys():
      metrics_per_output[output_info] = dict()
    if metric_name not in metrics_per_output[output_info].keys():
      metrics_per_output[output_info][metric_name] = list()
    metrics_per_output[output_info][metric_name].append(info)

  # return one metric of each type per output (otherwise the number of metrics would be multiplied by
  # the number of pipeline stages)
  user_defined_metrics = dict()
  for output_info in metrics_per_output.keys():
    for metric_name in metrics_per_output[output_info].keys():
      list_of_values = metrics_per_output[output_info][metric_name]

      new_name = "val_" if output_info.is_validation else ""
      is_multi_output_model = len(set(output_ids)) > 1
      new_name += f"output_{output_info.output_id}_" if is_multi_output_model else ""
      new_name += metric_name
      user_defined_metrics[new_name] = list_of_values
  return user_defined_metrics

def avg_metrics_over_pipeline_stages(metrics):
  avg_metrics = dict()
  for metric_name, list_of_values in list(metrics.items()):
    if len(list_of_values) > 0:
      avg_metrics[metric_name] = sum(list_of_values) / len(list_of_values)
    else:
      raise ValueError(f"[Pipelining][utilities] Cannot average an empty list of metrics: {metrics}")
  return avg_metrics
