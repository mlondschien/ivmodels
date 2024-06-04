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
double integrand(double b, void *params) {
    Params *x = (Params *) params;
    double a = x->a;
    double z = x->z;
    double alpha = x->alpha;
    double beta = x->beta;
    
    // Compute beta pdf (assuming alpha=2 and beta=2 for this example)
    double beta_pdf = gsl_ran_beta_pdf(b, alpha, beta);
    
    // Compute chi-squared cdf
    double chi2_cdf = gsl_cdf_chisq_P(z / (1.0 - a * b), 1);
    
    return beta_pdf * chi2_cdf;
}