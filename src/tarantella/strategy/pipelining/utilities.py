import re

def is_real_loss_or_metric(name):
  return "real" in name

def get_element_from_log_name(name, element):
  # name structure: p_{partition_id}_m_{micro_batch_id}_{real/edge/seq}_output_{output_id}_{metric_name}
  # e.g.: `p_1_m_1_real_output_0_sparse_categorical_accuracy`
  assert element in ["micro_batch_id", "", "output_id", "metric_name"]

  m = re.match("p_(?P<partition_id>.+)_m_(?P<micro_batch_id>.+)_(?P<type>.+)_output_(?P<output_id>\d+)_(?P<metric_name>.+)", name)
  return m.groupdict().get(element, None)

def extract_user_visible_metrics(metrics_name_and_info):
  metrics_per_output = dict()
  for name, info in metrics_name_and_info.items():
    if not is_real_loss_or_metric(name):
      continue
    output_id = get_element_from_log_name(name, "output_id")
    metric_name = get_element_from_log_name(name, "metric_name")

    if output_id not in metrics_per_output.keys():
      metrics_per_output[output_id] = dict()
    if metric_name not in metrics_per_output[output_id].keys():
      metrics_per_output[output_id][metric_name] = list()
    metrics_per_output[output_id][metric_name].append(info)

  # return one metric of each type per output (otherwise the number of metrics would be multiplied by
  # the number of pipeline stages)
  user_defined_metrics = dict()
  for output_id in metrics_per_output.keys():
    for metric_name in metrics_per_output[output_id].keys():
      list_of_values = metrics_per_output[output_id][metric_name]

      new_name = f"output_{output_id}_" if len(metrics_per_output) > 1 else ""
      new_name = new_name + metric_name
      user_defined_metrics[new_name] = list_of_values
  return user_defined_metrics
