import wfdb
import pandas as pd
import numpy as np
import scipy.signal as sgn
import tqdm
import h5py


# Leads available in the GE MUSE format.
# The remaining 4 in the 12-lead setup will not be calculated.
leads_used = ["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"]


def remove_baseline_filter(sample_rate):
    """For a given sampling rate"""
    fc = 0.8  # [Hz], cutoff frequency
    fst = 0.2  # [Hz], rejection band
    rp = 0.5  # [dB], ripple in passband
    rs = 40  # [dB], attenuation in rejection band
    wn = fc / (sample_rate / 2)
    wst = fst / (sample_rate / 2)

    filterorder, aux = sgn.ellipord(wn, wst, rp, rs)
    sos = sgn.iirfilter(
        filterorder, wn, rp, rs, btype="high", ftype="ellip", output="sos"
    )

    return sos


def normalize(ecg, sample_rate):
    """Take a stacked array with all lead data, remove the baseline, resample to 400Hz, and zero pad to length 4096."""
    # Remove baseline.
    sos = remove_baseline_filter(sample_rate)
    ecg_nobaseline = sgn.sosfiltfilt(sos, ecg, padtype="constant", axis=-1)

    # Resample to 400Hz (4000 samplings).
    new_freq = 400
    ecg_resampled = sgn.resample_poly(
        ecg_nobaseline, up=new_freq, down=sample_rate, axis=-1
    )

    # Zero pad from 4000 to a length of 4096 to match the CNN design used.
    new_len = 4096
    n_leads, len = ecg_resampled.shape
    ecg_zeropadded = np.zeros([n_leads, new_len])
    pad = (new_len - len) // 2
    ecg_zeropadded[..., pad : len + pad] = ecg_resampled

    return ecg_zeropadded


def main():
    base_path = "data/ptb-xl/"
    out_path = "data/"

    test_records = pd.read_csv(out_path + "ptbxl_selected.csv")

    test_traces = [
        wfdb.rdsamp(base_path + raw_file)
        for raw_file in tqdm.tqdm(test_records.filename_hr.values)
    ]
    test_traces = np.array([signal for signal, meta in test_traces])

    f = h5py.File(out_path + "test_data.h5", "w")
    set_name = "test"
    n = len(test_records)
    x_ecg = f.create_dataset(
        "x_ecg_{}".format(set_name), (n, len(leads_used), 4096), dtype="f8"
    )
    x_age = f.create_dataset("x_age_{}".format(set_name), (n,), dtype="i4")
    x_is_male = f.create_dataset("x_is_male_{}".format(set_name), (n,), dtype="i4")
    y = f.create_dataset("y_{}".format(set_name), (n,), dtype="i4")
    record_id = f.create_dataset("id_xmlfile_{}".format(set_name), (n,), dtype="S100")
    num_record_id = f.create_dataset("id_num_{}".format(set_name), (n,), dtype="i4")

    # leads to select and lead order: 'I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'
    lead_order = [0, 1, 6, 7, 8, 9, 10, 11]
    scale_factor = 1

    for i in tqdm.tqdm(range(len(test_records))):
        x_ecg[i, :, :] = normalize(test_traces[i, :, lead_order] * scale_factor, 500)
        x_age[i] = test_records.iloc[i]["age"]

        # sex: 0=male, 1=female (52% male in total PTB-XL dataset).
        if test_records.iloc[i]["sex"] == 0:
            x_is_male[i] = 1
        else:
            x_is_male[i] = 0

        # outcome labels: 0=control, 1=STEMI, 2=NSTEMI.
        if test_records.iloc[i]["label"] == "mi":
            y[i] = 1
        else:
            y[i] = 0

        record_id[i] = np.bytes_(test_records.iloc[i]["filename_hr"])
        num_record_id[i] = int(test_records.iloc[i]["patient_id"])
        
    f.close()


if __name__ == "__main__":
    main()
