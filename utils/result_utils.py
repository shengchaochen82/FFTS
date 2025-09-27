import h5py
import numpy as np
import os

"""
This function for results statistic module need to be revised.
"""

def average_data(configs, algorithm="", dataset="", goal="", times=10):
    test_mae, test_rmse, test_mape = get_all_results_for_one_algo(configs, algorithm, dataset, goal, times)

    min_mae = []
    min_rmse = []
    min_mape = []

    for i in range(times):
        min_mae_ = min(test_mae[i])

        # index = test_mae.index(min_mae_)
        index = np.where(test_mae == min_mae_)[0][0]

        min_mae.append(min_mae_)
        min_rmse.append(test_rmse[i][index])
        min_mape.append(test_mape[i][index])

    # print("std for best accurancy:", np.std(max_accurancy))
    print("mean for best mae:", np.mean(min_mae))
    print("\n mean for best rmse:", np.mean(min_rmse))
    print("\n mean for best mape:", np.mean(min_mape))


def get_all_results_for_one_algo(args, algorithm="", dataset="", goal="", times=10):
    test_mae = []
    test_rmse = []
    test_mape = []
    algorithms_list = [algorithm] * times
    setting = '{}_{}_{}_bm{}_sl{}_ll{}_pl{}_fea{}_tag{}_ispeft{}_peft{}_rk{}_pf{}'.format(
                args.goal,
                args.algorithm,
                args.dataset,
                args.base_model,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.features,
                args.target,
                args.is_peft,
                args.peft,
                args.rank,
                args.freeze_part)
    for i in range(times):
        file_name = dataset + "_" + algorithms_list[i] + "_" + setting + "_" + str(i)
        test_mae.append(np.array(read_data_then_delete(file_name, delete=False)[0]))
        test_rmse.append(np.array(read_data_then_delete(file_name, delete=False)[1]))
        test_mape.append(np.array(read_data_then_delete(file_name, delete=False)[2]))

    return test_mae, test_rmse, test_mape


def read_data_then_delete(file_name, delete=False):
    file_path = "../results/" + file_name + ".h5"

    with h5py.File(file_path, 'r') as hf:
        rs_test_mae = np.array(hf.get('rs_test_mae'))
        rs_test_rmse = np.array(hf.get('rs_test_rmse'))
        rs_test_mape = np.array(hf.get('rs_test_mape'))
    
    if delete:
        os.remove(file_path)
    print("Length: ", len(rs_test_mae))

    return rs_test_mae, rs_test_rmse, rs_test_mape