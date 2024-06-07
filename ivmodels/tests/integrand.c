#include <math.h>
#include <gsl/gsl_cdf.h>
#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_randist.h>

// Parameters struct to hold the values of a and z
typedef struct {
    double a;
    double z;
    double alpha;
    double beta;
} Params;

// C function to compute the integrand
double integrand(double x, void *user_data) {
    double *p = (double *)user_data;

    double a = p[0];
    double z = p[1];
    double alpha = p[2];
    double beta = p[3];
    int k = (int) p[4];

    // Compute beta pdf using the provided alpha and beta
    double beta_pdf = gsl_ran_beta_pdf(x, alpha, beta);

    // Compute chi-squared cdf
    double chi2_cdf = gsl_cdf_chisq_P(z / (1.0 - a * x), k);

    return beta_pdf * chi2_cdf;
}