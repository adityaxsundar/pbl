/* Auto-generated C function for WD-GPR-SR Symbolic Residual Correction */
#include <math.h>

double evaluate_symbolic_residual(const double* x) {
    /* Features:
       x[0]: cycle, x[1]: CCCT, x[2]: CVCT, x[3]: resistance,
       x[4]: T_rise, x[5]: T_max, x[6]: IC_peak_volts, x[7]: IC_peak_height
    */
    double x0 = x[0], x1 = x[1], x2 = x[2], x3 = x[3];
    double x4 = x[4], x5 = x[5], x6 = x[6], x7 = x[7];
    
    double term1 = sin(x1) * cos(x3) * 0.045;
    double term2 = log(fabs(x2) + 1.0) * -0.032;
    double term3 = (x0 > 0.5 ? -0.025 * x0 : 0.01);
    return term1 + term2 + term3;
}
