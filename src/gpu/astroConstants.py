import taichi as ti

# plain math pi value for taichi
pi = 3.141592653589793

@ti.func
def astroConstants(index: ti.i32) -> ti.f64: # type: ignore
    output = 0.0

    if index == 1:
        output = 6.67259e-20
    elif index == 2:
        output = 149597870.691
    elif index == 3:
        output = 6.955e5
    elif index == 4:
        output = 1.32712440017987e11
    elif index == 5:
        output = 299792.458
    elif index == 6:
        output = 9.80665
    elif index == 7:
        output = 384400.0
    elif index == 8:
        output = 84381.412 / 3600.0 * pi / 180.0
    elif index == 9:
        output = 0.1082626925638815e-2
    elif index == 11:
        output = 2.203208e4
    elif index == 12:
        output = 3.24858599e5
    elif index == 13:
        output = 3.98600433e5
    elif index == 14:
        output = 4.2828314e4
    elif index == 15:
        output = 1.26712767863e8
    elif index == 16:
        output = 3.79406260630e7
    elif index == 17:
        output = 5.79454900700e6
    elif index == 18:
        output = 6.83653406400e6
    elif index == 19:
        output = 9.81601000000e2
    elif index == 20:
        output = 4902.801
    elif index == 21:
        output = 2439.7
    elif index == 22:
        output = 6051.8
    elif index == 23:
        output = 6371.01
    elif index == 24:
        output = 3389.9
    elif index == 25:
        output = 69911.0
    elif index == 26:
        output = 58232.0
    elif index == 27:
        output = 25362.0
    elif index == 28:
        output = 24624.0
    elif index == 29:
        output = 1151.0
    elif index == 30:
        output = 1738.0
    elif index == 31:
        output = 50.3e-6
    elif index == 32:
        output = 4.458e-6
    elif index == 33:
        output = 0.1082626925638815e-2
    elif index == 34:
        output = 1960.45e-6
    elif index == 35:
        output = 14696.5735e-6
    elif index == 36:
        output = 16290.573e-6
    elif index == 37:
        output = 3510.68e-6
    elif index == 38:
        output = 3408.43e-6
    elif index == 39:
        output = ti.math.nan(ti.f64)
    elif index == 40:
        output = 202.7e-6
    elif index == 41:
        output = 0.0009
    elif index == 42:
        output = 0.00001
    elif index == 43:
        output = 0.00335
    elif index == 44:
        output = 0.00589
    elif index == 45:
        output = 0.06487
    elif index == 46:
        output = 0.09796
    elif index == 47:
        output = 0.02293
    elif index == 48:
        output = 0.01708
    elif index == 49:
        output = 0.0
    elif index == 50:
        output = 0.0012
    elif index == 51:
        output = 1407.6
    elif index == 52:
        output = -5832.6
    elif index == 53:
        output = 23.9345
    elif index == 54:
        output = 24.6229
    elif index == 55:
        output = 9.9250
    elif index == 56:
        output = 10.656
    elif index == 57:
        output = -17.24
    elif index == 58:
        output = 16.11
    elif index == 59:
        output = -153.2928
    elif index == 60:
        output = 655.720
    elif index == 61:
        output = 0.034
    elif index == 62:
        output = 177.36
    elif index == 63:
        output = 23.44
    elif index == 64:
        output = 25.19
    elif index == 65:
        output = 3.13
    elif index == 66:
        output = 26.73
    elif index == 67:
        output = 97.77
    elif index == 68:
        output = 28.32
    elif index == 69:
        output = 119.51
    elif index == 70:
        output = 6.68
    elif index == 71:
        output = 9082.7
    elif index == 72:
        output = 2601.3
    elif index == 73:
        output = 1361.0
    elif index == 74:
        output = 586.2
    elif index == 75:
        output = 50.26
    elif index == 76:
        output = 14.82
    elif index == 77:
        output = 3.69
    elif index == 78:
        output = 1.508
    elif index == 79:
        output = 0.873
    elif index == 80:
        output = 1361.0
    elif index == 81:
        output = 1367.0
    elif index == 82:
        output = 365.25
    else:
        # Taichi functions cannot raise Python exceptions
        output = 0.0

    return output
