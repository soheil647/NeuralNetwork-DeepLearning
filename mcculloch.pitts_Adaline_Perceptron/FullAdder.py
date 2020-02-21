import numpy as np


def full_adder_1bit(a1, a2, b1, b2):
    bias = 1
    # Layer 1
    z11 = np.heaviside(np.sum(a1 * -1 + b1 * 1), 0)
    print(z11)
    z12 = np.heaviside(np.sum(a1 * 1 + b1 * -1), 0)
    print(z12)
    c1 = np.heaviside(np.sum(a1 * 1 + b1 * 1 + bias * -1), 0)
    print(c1)

    z21 = np.heaviside(np.sum(a2 * -1 + b2 * 1), 0)
    z22 = np.heaviside(np.sum(a2 * 1 + b2 * -1), 0)
    c2 = np.heaviside(np.sum(a2 * 1 + b2 * 1 + bias * -1), 0)

    #Layer 2
    sum1 = np.heaviside(np.sum(z11 * 1 + z12 * 1), 0)
    sum2_1 = np.heaviside(np.sum(z21 * 1 + z22 * 1), 0)

    #Layer 3
    z31 = np.heaviside(np.sum(c1 * -1 + sum2_1 * 1), 0)
    z32 = np.heaviside(np.sum(c1 * 1 + sum2_1 * -1), 0)
    and_sum2_c1 = np.heaviside(np.sum(c1 * 1 + sum2_1 * 1 + -1), 0)

    #Layer 4
    sum2 = np.heaviside(np.sum(z31 * 1 + z32 * 1), 0)
    carry = np.heaviside(np.sum(c2 * 1 + and_sum2_c1 * 1), 0)

    return sum1, sum2, carry


print(full_adder_1bit(1, 1, 0, 0))
