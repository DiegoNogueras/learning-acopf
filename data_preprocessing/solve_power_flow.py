import helper_functions as hp
import pandas as pd
import numpy as np
import bus_bounds_data as bb_data
from time import perf_counter
from pypower.case118 import case118
from pypower.savecase import savecase
from keras.models import load_model
from egret.models.acpf import solve_acpf
from egret.parsers.matpower_parser import create_ModelData


def predict_acopf(model, X_test, y_test, slack_bus_data, scaler):
    # v_bounds = {(idx + 1) :  {"v_min": 0.94, "v_max": 1.06} for idx in range(54)}
    label_cols = hp.get_label_cols()
    bus_gen_nums = hp.get_bus_gen_nums()

    df = pd.DataFrame()
    for x_test, y_test_sample, slack_bus in zip(X_test, y_test, slack_bus_data.values):
        tic = perf_counter()
        y_pred = model.predict(x_test.reshape(1, -1), verbose=0)
        y_test_sample_df = pd.DataFrame(data=y_test_sample.reshape(1, -1), columns=label_cols)
        y_pred_df = pd.DataFrame(data=y_pred, columns=label_cols)

        for col, p_bounds_key in zip(label_cols[::2], bb_data.p_bounds.keys()):
            y_test_sample_df[col] = y_test_sample_df[col].apply(
                lambda x: hp.unparametrize_func(
                    x, bb_data.p_bounds[p_bounds_key]['p_min'],
                    bb_data.p_bounds[p_bounds_key]['p_max']))
            y_pred_df[col] = y_pred_df[col].apply(
                lambda x: hp.unparametrize_func(
                    x, bb_data.p_bounds[p_bounds_key]['p_min'],
                    bb_data.p_bounds[p_bounds_key]['p_max']))

        y_test_sample_df.loc[0, 'v_gen1'::2] = y_test_sample_df.loc[0, 'v_gen1'::2].apply(
            lambda x: hp.unparametrize_func(x, 0.94, 1.06))
        y_pred_df.loc[0, 'v_gen1'::2] = y_pred_df.loc[0, 'v_gen1'::2].apply(
            lambda x: hp.unparametrize_func(x, 0.94, 1.06))

        x_test = hp.unscale_X_test_data(x_test.reshape(1, -1), scaler)

        pv_bus_p_gen = {"p{}_gen".format(bus_num): p_gen for bus_num, p_gen in zip(
            bus_gen_nums, y_pred_df.loc[0, 'p_gen1'::2].values)}

        pq_bus_p_load = {
            "p{}_load".format(bus_num + 1): p_load for bus_num,
            p_load in zip(range(118),
                          x_test[0][:: 2])}
        pq_bus_q_load = {
            "q{}_load".format(bus_num + 1): q_load for bus_num,
            q_load in zip(range(118),
                          x_test[0][1:: 2])}

        y = {**pv_bus_p_gen, **pq_bus_p_load, **pq_bus_q_load}

        unknown_angles = {"Angle_{}".format(str(bus_num + 1)): 0 for bus_num in range(118)}
        del unknown_angles['Angle_69']

        unknown_pq_v_mags = {
            "Voltage_{}".format(str(bus_num + 1)): 1 for bus_num in range(118)
            if str(bus_num + 1) not in bus_gen_nums}

        x = {**unknown_angles, **unknown_pq_v_mags}

        gen_bus_idx = [bus_num + 1 for bus_num in range(118) if str(bus_num + 1) in bus_gen_nums]

        known_pv_v_mags = {
            "Voltage_{}".format(bus_num): y_pred_df.loc[0, "v_gen1":: 2][idx] for bus_num,
            idx in zip(gen_bus_idx, range(len(gen_bus_idx)))}

        slack_bus_v = slack_bus[2]
        slack_bus_a = slack_bus[3]

        V = []
        for idx in range(118):
            if idx + 1 == 69:
                V.append([slack_bus_v, slack_bus_a])
            elif str(idx + 1) not in bus_gen_nums:
                V.append([x["Voltage_{}".format(idx + 1)], x["Angle_{}".format(idx + 1)]])
            else:
                V.append(
                    [known_pv_v_mags["Voltage_{}".format(idx + 1)],
                     x["Angle_{}".format(idx + 1)]])

        case118_dict = case118()

        for idx in range(len(case118_dict['bus'])):
            p_key = "p{}_load".format(idx + 1)
            q_key = "q{}_load".format(idx + 1)
            case118_dict['bus'][idx][2] = y[p_key]
            case118_dict['bus'][idx][3] = y[q_key]
            case118_dict['bus'][idx][7] = V[idx][0]
            case118_dict['bus'][idx][8] = V[idx][1]

        V_gen = [v_elem[0] for v_elem in V if v_elem[0] != 1]

        for idx, bus_num in zip(
                range(len(case118_dict['gen'])),
                hp.get_bus_gen_nums(include_slack=True)):
            if idx != 29:
                p_key = "p{}_gen".format(bus_num)
                case118_dict['gen'][idx][1] = y[p_key]
            else:
                case118_dict['gen'][idx][1] = 0

            case118_dict['gen'][idx][5] = V_gen[idx]

        savecase("/Users/nogueras1/Documents/ACOPF_Workspace/new_case_data/new_case", ppc=case118_dict)

        with open("/Users/nogueras1/Documents/ACOPF_Workspace/new_case_data/new_case.py") as f:
            lines = f.readlines()
            data = lines[12:130]

        new_data = []
        for line in data:
            line_arr = line.strip()[1:-2].split(",")
            new_string = ""
            for element in line_arr:
                new_string += element.strip()
                new_string += "    "

            new_string = new_string[:-4] + ";"
            new_string += "\n"
            new_data.append(new_string)

        with open("/Users/nogueras1/Documents/ACOPF_Workspace/data_generation/case118_data/case118.txt") as f:
            lines = f.readlines()

        with open("/Users/nogueras1/Documents/ACOPF_Workspace/data_generation/generated_data/case118.txt", 'w') as f:
            f.writelines(lines[:23] + new_data + lines[141:])

        with open("/Users/nogueras1/Documents/ACOPF_Workspace/new_case_data/new_case.py") as f:
            lines = f.readlines()
            data = lines[135:189]

        new_data = []
        for line in data:
            line_arr = line.strip()[1:-2].split(",")
            new_string = ""
            for element in line_arr:
                new_string += element.strip()
                new_string += "    "

            new_string = new_string[:-4] + ";"
            new_string += "\n"
            new_data.append(new_string)

        with open("/Users/nogueras1/Documents/ACOPF_Workspace/data_generation/generated_data/case118.txt") as f:
            lines = f.readlines()

        with open("/Users/nogueras1/Documents/ACOPF_Workspace/data_generation/generated_data/case118.txt", 'w') as f:
            f.writelines(lines[:145] + new_data + lines[199:])

        model_data = create_ModelData(
            "/Users/nogueras1/Documents/ACOPF_Workspace/data_generation/generated_data/case118.txt")
        toc = perf_counter()

        md, results = solve_acpf(model_data, "ipopt", solver_tee=False, return_results=True)

        p_generated = [
            md.data['elements']['generator'][key]['pg']
            for key in md.data['elements']['generator'].keys()]
        q_generated = [
            md.data['elements']['generator'][key]['qg']
            for key in md.data['elements']['generator'].keys()]
        v_generated = [
            md.data['elements']['generator'][key]['vg']
            for key in md.data['elements']['generator'].keys()]
        vmg_at_bus = [md.data['elements']['bus'][key]['vm']
                      for key in md.data['elements']['bus'].keys()]
        voltage_angle = [md.data['elements']['bus'][key]['va']
                         for key in md.data['elements']['bus'].keys()]

        data = np.concatenate((p_generated, q_generated, v_generated, vmg_at_bus, voltage_angle))

        p_gen_cols = ["p_gen{}".format(i + 1) for i in range(54)]
        q_gen_cols = ["q_gen{}".format(i + 1) for i in range(54)]
        v_gen_cols = ["v_gen{}".format(i + 1) for i in range(54)]
        v_at_bus_cols = ["v_at_bus{}".format(i + 1) for i in range(118)]
        va_at_bus_cols = ["va_at_bus{}".format(i + 1) for i in range(118)]

        cols = np.concatenate((p_gen_cols, q_gen_cols, v_gen_cols, v_at_bus_cols, va_at_bus_cols))

        new_df = pd.DataFrame(data=[data], columns=cols)
        new_df['time_to_solve'] = (toc - tic) + results['Solver'][0]['Time']

        df = pd.concat([df, new_df])

    return df


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, df, slack_bus_data, scaler = hp.pre_process_data_from_file()

    previous_model = True
    if not previous_model:
        model = hp.train_model(X_train, X_test, y_train, y_test, plot_losses=True)
        y_pred = model.predict(X_test)
        label_cols = hp.get_label_cols()
        y_test_df = pd.DataFrame(data=y_test, columns=label_cols)
        y_pred_df = pd.DataFrame(data=y_pred, columns=label_cols)
        hp.evaluate_model(y_test_df, y_pred_df)
        model.save("../saved_models/acopf_model6.h5")  # REMEMBER TO CHANGE MODEL NUMBER
    else:
        model_path = "/Users/nogueras1/Documents/ACOPF_Workspace/saved_models/acopf_model5.h5"
        model = load_model(model_path)
        df = predict_acopf(model, X_test, y_test, slack_bus_data, scaler)
        df.to_csv("predicted_case118_data3.csv")
