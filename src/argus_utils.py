from argus import Model


def load_compatible_weights(old_model: Model, new_model: Model) -> Model:
    """Load only compatible weights from older model.

    It can be useful if the new model is slighly different (e.g. changed number
    of classes).

    Args:
        old_model (Model): Original model (argus or raw pytorch), which is used as the base.
        new_model (Model): The new argus models to take new weights from.

    Returns:
        Model: Resulted argus model, as an updated version of new_model
    """
    state_dict_new = new_model.nn_module.state_dict()
    if isinstance(old_model, Model):
        state_dict_old = old_model.nn_module.state_dict()
        # If weights shape of the old model is different from the new model, use
        # weights from the new model
        for k in state_dict_old:
            if k in state_dict_new\
                    and state_dict_old[k].shape != state_dict_new[k].shape:
                print(f'Skip loading parameter {k}, '
                  f'required shape {state_dict_new[k].shape}, '
                  f'loaded shape {state_dict_old[k].shape}.')
                state_dict_old[k] = state_dict_new[k]
        # Add weights, which are present in the new model only
        for k in state_dict_new:
            if not (k in state_dict_old):
                # print(f'No param in old: {k}.')
                state_dict_old[k] = state_dict_new[k]
        # print(list(state_dict_old.keys()))
        new_model.nn_module.load_state_dict(state_dict_old)

    else:
        state_dict_old = old_model['model']
        for key in state_dict_old:
            model_key = 'model.'+key
            if model_key in state_dict_new:
                if state_dict_old[key].shape != state_dict_new[model_key].shape:
                    print(f'Skipping layer {key}:{state_dict_new[model_key].shape}')
                else:
                    state_dict_new[model_key] = state_dict_old[key]
        
        new_model.nn_module.load_state_dict(state_dict_new)

    return new_model