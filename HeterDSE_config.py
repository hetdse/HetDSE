
def get_HeterDSE_config(tag):
    multiobj_config = {}
    multiobj_config['tag'] = tag
    multiobj_config['metric_prediction_model'] = None
    multiobj_config['pareto_crossover'] = 0
    multiobj_config['aggr_workloads_models'] = None
    multiobj_config['transfer_learning_model'] = None
    multiobj_config['train_score_based'] = 0
    multiobj_config['metric_pred_for_schedule'] = 0
    multiobj_config['get_obj_by_model_pred'] = 0
    multiobj_config['sche_explore'] = 1

    if 'HeterDSE' in tag:
        if 0 == multiobj_config['metric_pred_for_schedule']:
            multiobj_config['version'] = 1
        else:
            multiobj_config['version'] = 0
    else:
        multiobj_config['version'] = 1

    if '_awm' in tag:
        multiobj_config['aggr_workloads_models'] = 'Ridge'
        multiobj_config['tag'] += '-' + multiobj_config['aggr_workloads_models']

    if '_tf' in tag:
        multiobj_config['transfer_learning_model'] = 'Ridge'
        multiobj_config['tag'] += '-' + multiobj_config['transfer_learning_model']

    if '_pc' in tag:
        multiobj_config['pareto_crossover'] = 1
        multiobj_config['n_clusters'] = 15
        #multiobj_config['metric_prediction_model'] = 'Ridge'
        multiobj_config['metric_prediction_model'] = 'AdaGBRT'
        #multiobj_config['tag'] += '-' + multiobj_config['pareto_crossover']

    if '_m' in tag:
        multiobj_config['metric_prediction_model'] = 'AdaGBRT'
        multiobj_config['tag'] += '-' + multiobj_config['metric_prediction_model']

    if '_tsb' in tag:
        multiobj_config['train_score_based'] = 1

    multiobj_config['metric_model'] = 1
    multiobj_config['metric_model_argu'] = 0
    if multiobj_config['metric_model']:
        multiobj_config['tag'] += '-mm' + str(multiobj_config['metric_model'])
        if multiobj_config['metric_model_argu']:
            multiobj_config['tag'] += 'a' + str(multiobj_config['metric_model_argu'])

    multiobj_config['init_by_mask'] = 1
    if multiobj_config['init_by_mask']:
        multiobj_config['tag'] += '-init' + str(multiobj_config['init_by_mask'])

    if multiobj_config['get_obj_by_model_pred']:
        multiobj_config['tag'] += '-go' + str(multiobj_config['get_obj_by_model_pred'])

    if multiobj_config['sche_explore']:
        multiobj_config['tag'] += '-se' + str(multiobj_config['sche_explore'])

    multiobj_config['tag'] += '_v' + str(multiobj_config['version'])

    return multiobj_config