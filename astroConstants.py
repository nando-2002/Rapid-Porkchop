import numpy as np


def astroConstants(index):
    match index:
        case 1:
            output = 6.67259e-20  # From DITAN and Horizon
        case 2:
            output = 149597870.691  # From DE405
        case 3:
            # output = 700000  # From DITAN
            output = 6.955 * 10**5  # From Horizon [W3]
        case 4:
            # output = 0.19891000000000E+31*6.67259e-20  # From DITAN
            output = 1.32712440017987E+11  # From DE405 [A]
        case 5:
            output = 299792.458  # Definition in the SI, Horizon, DE405
        case 6:
            output = 9.80665  # Definition in Wertz, SMAD
        case 7:
            # output = 384401  # Definition in Wertz, SMAD
            output = 384400  # From Horizon [W3]
        case 8:
            # output = 23.43928111*pi/180  # Definition in Wertz, SMAD
            output = 84381.412 / 3600 * np.pi / 180  # Definition in Horizon
            # obliquity of ecliptic (J2000) epsilon = 84381.412 (± 0.005) arcsec
        case 9:
            output = 0.1082626925638815e-2  # Definition in Wertz, SMAD
        case 11:
            # output = 0.33020000000000E+24*6.67259e-20  # From DITAN
            # output = 0.330104E+24*6.67259e-20  # From Horizon [F]
            output = 2.203208E+4  # From DE405
        case 12:
            # output = 0.48685000000000E+25*6.67259e-20  # From DITAN
            # output = 4.86732E+24*6.67259e-20  # From Horizon [G]
            output = 3.24858599E+5  # From DE405
        case 13:
            # output = 0.59736990612667E+25*6.67259e-20  # From DITAN
            # output = 5.97219E+24*6.67259e-20  # From Horizon [H]
            output = 3.98600433e+5  # From DE405
        case 14:
            # output = 0.64184999247389E+24*6.67259e-20  # From DITAN
            # output = 0.641693E+24*6.67259e-20  # From Horizon [I]
            output = 4.2828314E+4  # From DE405
        case 15:
            # output = 0.18986000000000E+28*6.67259e-20  # From DITAN
            # output = 1898.13E+24*6.67259e-20  # From Horizon [J]
            output = 1.26712767863E+08  # From DE405
        case 16:
            # output = 0.56846000000000E+27*6.67259e-20  # From DITAN
            # output = 568.319E+24*6.67259e-20  # From Horizon [k]
            output = 3.79406260630E+07  # From DE405
        case 17:
            # output = 0.86832000000000E+26*6.67259e-20  # From DITAN
            # output = 86.8103E+24*6.67259e-20  # From Horizon [L]
            output = 5.79454900700E+06  # From DE405
        case 18:
            # output = 0.10243000000000E+27*6.67259e-20  # From DITAN
            # output = 102.410E+24*6.67259e-20  # From Horizon [M]
            output = 6.83653406400E+06  # From DE405
        case 19:
            # output = 0.14120000000000E+23*6.67259e-20  # From DITAN
            # output = .01309E+24*6.67259e-20  # From Horizon [N]
            output = 9.81601000000E+02  # From DE405
        case 20:
            # output = 0.73476418263373E+23*6.67259e-20  # From DITAN
            output = 4902.801  # From Horizon [M2]
            # output = 4902.801076  # From Horizon [M3]
        case 21:
            # output = 0.24400000000000E+04  # From DITAN
            output = 2439.7  # From Horizon [D]
        case 22:
            # output = 0.60518000000000E+04  # From DITAN
            output = 6051.8  # From Horizon [D]
        case 23:
            # output = 0.63781600000000E+04  # From DITAN
            # output = 6371.00  # From Horizon [B]
            output = 6371.01  # From Horizon [W3]
        case 24:
            # output = 0.33899200000000E+04  # From DITAN
            # output = 3389.50  # From Horizon [D]
            output = 3389.9  # From Horizon [W3]
        case 25:
            # output = 0.69911000000000E+05  # From DITAN
            output = 69911  # From Horizon [D]
        case 26:
            # output = 0.58232000000000E+05  # From DITAN
            output = 58232  # From Horizon [D]
        case 27:
            # output = 0.25362000000000E+05  # From DITAN
            output = 25362  # From Horizon [D]
        case 28:
            # output = 0.24624000000000E+05  # From DITAN
            # output = 24622  # From Horizon [D]
            output = 24624  # From Horizon [W3]
        case 29:
            # output = 0.11510000000000E+04  # From DITAN
            output = 1151  # From Horizon [C]
        case 30:
            # output = 0.17380000000000E+04  # From DITAN
            # output = 1737.5  # From Horizon [M1]
            output = 1738.0  # From Horizon [M3]
        # J2 Gravitational Harmonic coefficient [-]: 
        case 31:
            output = 50.3e-6  # [P1]
        case 32:
            output = 4.458e-6  # [P2]
        case 33:
            output = 0.1082626925638815e-2  # Definition in Wertz, SMAD
        case 34:
            output = 1960.45e-6  # [P4]
        case 35:
            output = 14696.5735e-6  # [P5]
        case 36:
            output = 16290.573e-6  # [P6]
        case 37:
            output = 3510.68e-6  # [P7]
        case 38:
            output = 3408.43e-6  # [P8]
        case 39:
            output = np.nan  # [P9] -> Not present in Nasa fact sheets
        case 40:
            output = 202.7e-6  # [P10]
        # Planetary oblateness
        case 41:
            output = 0.0009  # [P1]
        case 42:
            output = 0.00001  # [P2]
        case 43:
            output = 0.00335  # [P3]
        case 44:
            output = 0.00589  # [P4]
        case 45:
            output = 0.06487  # [P5]
        case 46:
            output = 0.09796  # [P6]
        case 47:
            output = 0.02293  # [P7]
        case 48:
            output = 0.01708  # [P8]
        case 49:
            output = 0.0  # [P9]
        case 50:
            output = 0.0012  # [P10]
        # Sidereal rotation period [hours]:
        case 51:
            output = 1407.6  # [P1]
        case 52:
            output = -5832.6  # [P2]
        case 53:
            output = 23.9345  # [P3]
        case 54:
            output = 24.6229  # [P4]
        case 55:
            output = 9.9250  # [P5]
        case 56:
            output = 10.656  # [P6]
        case 57:
            output = -17.24  # [P7]
        case 58:
            output = 16.11  # [P8]
        case 59:
            output = -153.2928  # [P9]
        case 60:
            output = 655.720  # [P10]
        # Axial tilt [deg]:
        case 61:
            output = 0.034  # [P1]
        case 62:
            output = 177.36  # [P2]
        case 63:
            output = 23.44  # [P3]
        case 64:
            output = 25.19  # [P4]
        case 65:
            output = 3.13  # [P5]
        case 66:
            output = 26.73  # [P6]
        case 67:
            output = 97.77  # [P7]
        case 68:
            output = 28.32  # [P8]
        case 69:
            output = 119.51  # [P9]
        case 70:
            output = 6.68  # [P10]
        # Solar irradiance [W/m^2]
        case 71:
            output = 9082.7  # [P1]
        case 72:
            output = 2601.3  # [P2]
        case 73:
            output = 1361.0  # [P3]
        case 74:
            output = 586.2  # [P4]
        case 75:
            output = 50.26  # [P5]
        case 76:
            output = 14.82  # [P6]
        case 77:
            output = 3.69  # [P7]
        case 78:
            output = 1.508  # [P8]
        case 79:
            output = 0.873  # [P9]
        case 80:
            output = 1361.0  # [P10]
        # Other constants
        case 81:
            output = 1367  # From Wertz, SMAD
            # output = 1367.6  # From Horizon [W3]
        case 82:
            output = 365.25  # From Horizon
        case _:
            raise ValueError(f"Unknown astro constant index: {index}")
    return output
