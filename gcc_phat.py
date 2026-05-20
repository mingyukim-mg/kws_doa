import numpy as np

SPEED_OF_SOUND = 343.0
FS = 16000
INTERP = 16


# =========================
# GCC-PHAT
# =========================
def gcc_phat(sig, refsig,
             fs=16000,
             max_tau=0.00025,
             interp=16):

    n = sig.shape[0] + refsig.shape[0]

    SIG = np.fft.rfft(sig, n=n)
    REFSIG = np.fft.rfft(refsig, n=n)

    R = SIG * np.conj(REFSIG)

    # PHAT
    R /= np.abs(R) + 1e-15

    cc = np.fft.irfft(R, n=interp * n)

    max_shift = int(interp * fs * max_tau)

    cc = np.concatenate((
        cc[-max_shift:],
        cc[:max_shift + 1]
    ))

    shift = np.argmax(np.abs(cc)) - max_shift

    tau = shift / float(interp * fs)

    return tau


# =========================
# GLOBAL DOA
# =========================
def global_doa(pair_results):

    best_angle = 0
    min_error = float('inf')

    for phi in range(360):

        total_error = 0

        for tau, distance, axis in pair_results:

            expected_tau = (
                distance
                * np.cos(np.radians(phi - axis))
            ) / SPEED_OF_SOUND

            error = (tau - expected_tau) ** 2

            total_error += error

        if total_error < min_error:
            min_error = total_error
            best_angle = phi

    return best_angle, min_error


# =========================
# angle -> direction
# =========================
def angle_to_direction(angle):

    angle = angle % 360

    if angle >= 337.5 or angle < 22.5:
        return "R"

    elif angle < 67.5:
        return "BR"

    elif angle < 112.5:
        return "B"

    elif angle < 157.5:
        return "BL"

    elif angle < 202.5:
        return "L"

    elif angle < 247.5:
        return "FL"

    elif angle < 292.5:
        return "F"

    else:
        return "FR"


# =========================
# MAIN API
# =========================
def estimate_direction(audio_4ch):

    """
    input:
        audio_4ch shape = (4, N)

    MIC mapping:
        MIC1 = audio_4ch[0]
        MIC2 = audio_4ch[1]
        MIC3 = audio_4ch[2]
        MIC4 = audio_4ch[3]
    """

    mics = {
        1: audio_4ch[0],
        2: audio_4ch[1],
        3: audio_4ch[2],
        4: audio_4ch[3]
    }

    pairs = [

        ((2,1), 0.05, 0),
        ((3,4), 0.05, 0),

        ((2,3), 0.05, 90),
        ((1,4), 0.05, 90),

        ((2,4), 0.07, 45),
        ((1,3), 0.07, 135),
    ]

    pair_results = []

    for (a, b), distance, axis in pairs:

        pair_max_tau = distance / SPEED_OF_SOUND

        tau = gcc_phat(
            mics[a],
            mics[b],
            fs=FS,
            max_tau=pair_max_tau,
            interp=INTERP
        )

        pair_results.append(
            (tau, distance, axis)
        )

    best_angle, error = global_doa(pair_results)

    direction = angle_to_direction(best_angle)

    return {
        "angle": best_angle,
        "direction": direction,
        "error": error
    }