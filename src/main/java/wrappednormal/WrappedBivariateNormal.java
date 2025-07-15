package wrappednormal;

import java.util.Random;

/**
 * A class representing a Wrapped Bivariate Normal distribution on [0, 2π) × [0, 2π).
 * This distribution is obtained by wrapping a bivariate normal distribution onto a torus.
 */
public class WrappedBivariateNormal {
    private static final double TWO_PI = 2 * Math.PI;
    private static final int DEFAULT_WRAPPING_NUMBER = 10; // Number of wrappings to approximate infinity
    
    // Parameters of the underlying bivariate normal
    private double mu1;     // Mean of first component
    private double mu2;     // Mean of second component
    private double sigma1;  // Standard deviation of first component
    private double sigma2;  // Standard deviation of second component
    private double rho;     // Correlation coefficient
    
    // Precomputed values for efficiency
    private double sigma1Sq;
    private double sigma2Sq;
    private double rhoSq;
    private double normalizationConstant;
    
    private Random random;
    
    /**
     * Constructs a wrapped bivariate normal distribution.
     * 
     * @param mu1 Mean of the first component (will be wrapped to [0, 2π))
     * @param mu2 Mean of the second component (will be wrapped to [0, 2π))
     * @param sigma1 Standard deviation of the first component (must be positive)
     * @param sigma2 Standard deviation of the second component (must be positive)
     * @param rho Correlation coefficient (must be in [-1, 1])
     */
    public WrappedBivariateNormal(double mu1, double mu2, double sigma1, double sigma2, double rho) {
        if (sigma1 <= 0 || sigma2 <= 0) {
            throw new IllegalArgumentException("Standard deviations must be positive");
        }
        if (rho < -1 || rho > 1) {
            throw new IllegalArgumentException("Correlation coefficient must be in [-1, 1]");
        }
        
        this.mu1 = wrapAngle(mu1);
        this.mu2 = wrapAngle(mu2);
        this.sigma1 = sigma1;
        this.sigma2 = sigma2;
        this.rho = rho;
        
        // Precompute frequently used values
        this.sigma1Sq = sigma1 * sigma1;
        this.sigma2Sq = sigma2 * sigma2;
        this.rhoSq = rho * rho;
        this.normalizationConstant = 1.0 / (2 * Math.PI * sigma1 * sigma2 * Math.sqrt(1 - rhoSq));
        
        this.random = new Random();
    }
    
    /**
     * Constructs a wrapped bivariate normal distribution with a specified random seed.
     */
    public WrappedBivariateNormal(double mu1, double mu2, double sigma1, double sigma2, double rho, long seed) {
        this(mu1, mu2, sigma1, sigma2, rho);
        this.random = new Random(seed);
    }
    
    /**
     * Wraps an angle to the range [0, 2π).
     */
    private static double wrapAngle(double angle) {
        double wrapped = angle % TWO_PI;
        if (wrapped < 0) {
            wrapped += TWO_PI;
        }
        return wrapped;
    }
    
    /**
     * Computes the probability density function at the given point.
     * 
     * @param theta1 First angular coordinate in [0, 2π)
     * @param theta2 Second angular coordinate in [0, 2π)
     * @return The probability density at (theta1, theta2)
     */
    public double pdf(double theta1, double theta2) {
        double density = 0.0;
        
        // Sum over multiple wrappings to approximate the infinite sum
        for (int k1 = -DEFAULT_WRAPPING_NUMBER; k1 <= DEFAULT_WRAPPING_NUMBER; k1++) {
            for (int k2 = -DEFAULT_WRAPPING_NUMBER; k2 <= DEFAULT_WRAPPING_NUMBER; k2++) {
                double x1 = theta1 + k1 * TWO_PI;
                double x2 = theta2 + k2 * TWO_PI;
                density += bivariateNormalPdf(x1, x2);
            }
        }
        
        return density;
    }
    
    /**
     * Computes the bivariate normal PDF at the given point.
     */
    private double bivariateNormalPdf(double x1, double x2) {
        double z1 = (x1 - mu1) / sigma1;
        double z2 = (x2 - mu2) / sigma2;
        
        double exponent = -0.5 / (1 - rhoSq) * (z1 * z1 - 2 * rho * z1 * z2 + z2 * z2);
        
        return normalizationConstant * Math.exp(exponent);
    }
    
    /**
     * Generates a random sample from the wrapped bivariate normal distribution.
     * 
     * @return A 2-element array containing the sampled angles in [0, 2π)
     */
    public double[] sample() {
        // Generate standard normal variates
        double z1 = random.nextGaussian();
        double z2 = random.nextGaussian();
        
        // Transform to correlated bivariate normal
        double x1 = mu1 + sigma1 * z1;
        double x2 = mu2 + sigma2 * (rho * z1 + Math.sqrt(1 - rhoSq) * z2);
        
        // Wrap to [0, 2π)
        return new double[] { wrapAngle(x1), wrapAngle(x2) };
    }
    
    /**
     * Generates multiple samples from the distribution.
     * 
     * @param n Number of samples to generate
     * @return An n×2 array of samples
     */
    public double[][] sample(int n) {
        double[][] samples = new double[n][2];
        for (int i = 0; i < n; i++) {
            samples[i] = sample();
        }
        return samples;
    }
    
    /**
     * Computes the circular mean of the first component.
     * 
     * @return The circular mean of theta1
     */
    public double circularMean1() {
        return mu1;
    }
    
    /**
     * Computes the circular mean of the second component.
     * 
     * @return The circular mean of theta2
     */
    public double circularMean2() {
        return mu2;
    }
    
    /**
     * Computes the circular variance of the first component.
     * For large sigma relative to 2π, this approaches 1 (uniform distribution).
     * 
     * @return The circular variance of theta1 (between 0 and 1)
     */
    public double circularVariance1() {
        return 1 - Math.exp(-sigma1Sq / 2);
    }
    
    /**
     * Computes the circular variance of the second component.
     * 
     * @return The circular variance of theta2 (between 0 and 1)
     */
    public double circularVariance2() {
        return 1 - Math.exp(-sigma2Sq / 2);
    }
    
    /**
     * Computes the log probability density at the given point.
     * This is more numerically stable for very small densities.
     * 
     * @param theta1 First angular coordinate in [0, 2π)
     * @param theta2 Second angular coordinate in [0, 2π)
     * @return The log probability density at (theta1, theta2)
     */
    public double logPdf(double theta1, double theta2) {
        // Use log-sum-exp trick for numerical stability
        double maxLogTerm = Double.NEGATIVE_INFINITY;
        
        // First pass: find maximum log term
        for (int k1 = -DEFAULT_WRAPPING_NUMBER; k1 <= DEFAULT_WRAPPING_NUMBER; k1++) {
            for (int k2 = -DEFAULT_WRAPPING_NUMBER; k2 <= DEFAULT_WRAPPING_NUMBER; k2++) {
                double x1 = theta1 + k1 * TWO_PI;
                double x2 = theta2 + k2 * TWO_PI;
                double logTerm = logBivariateNormalPdf(x1, x2);
                maxLogTerm = Math.max(maxLogTerm, logTerm);
            }
        }
        
        // Second pass: compute sum using log-sum-exp
        double sumExp = 0.0;
        for (int k1 = -DEFAULT_WRAPPING_NUMBER; k1 <= DEFAULT_WRAPPING_NUMBER; k1++) {
            for (int k2 = -DEFAULT_WRAPPING_NUMBER; k2 <= DEFAULT_WRAPPING_NUMBER; k2++) {
                double x1 = theta1 + k1 * TWO_PI;
                double x2 = theta2 + k2 * TWO_PI;
                double logTerm = logBivariateNormalPdf(x1, x2);
                sumExp += Math.exp(logTerm - maxLogTerm);
            }
        }
        
        return maxLogTerm + Math.log(sumExp);
    }
    
    /**
     * Computes the log of the bivariate normal PDF at the given point.
     */
    private double logBivariateNormalPdf(double x1, double x2) {
        double z1 = (x1 - mu1) / sigma1;
        double z2 = (x2 - mu2) / sigma2;
        
        double logNormConstant = -Math.log(2 * Math.PI) - Math.log(sigma1) - Math.log(sigma2) 
                                - 0.5 * Math.log(1 - rhoSq);
        double exponent = -0.5 / (1 - rhoSq) * (z1 * z1 - 2 * rho * z1 * z2 + z2 * z2);
        
        return logNormConstant + exponent;
    }
    
    // Getters for the distribution parameters
    public double getMu1() { return mu1; }
    public double getMu2() { return mu2; }
    public double getSigma1() { return sigma1; }
    public double getSigma2() { return sigma2; }
    public double getRho() { return rho; }
    
    @Override
    public String toString() {
        return String.format("WrappedBivariateNormal(mu1=%.4f, mu2=%.4f, sigma1=%.4f, sigma2=%.4f, rho=%.4f)",
                mu1, mu2, sigma1, sigma2, rho);
    }
    
    /**
     * Example usage and testing.
     */
    public static void main(String[] args) {
        // Create a wrapped bivariate normal distribution
        WrappedBivariateNormal wbn = new WrappedBivariateNormal(
            Math.PI,      // mu1
            Math.PI / 2,  // mu2
            0.5,          // sigma1
            0.7,          // sigma2
            0.3           // rho (correlation)
        );
        
        System.out.println("Distribution: " + wbn);
        System.out.println();
        
        // Evaluate PDF at a point
        double theta1 = Math.PI;
        double theta2 = Math.PI / 2;
        System.out.printf("PDF at (%.4f, %.4f): %.6f%n", theta1, theta2, wbn.pdf(theta1, theta2));
        System.out.printf("Log PDF at (%.4f, %.4f): %.6f%n", theta1, theta2, wbn.logPdf(theta1, theta2));
        System.out.println();
        
        // Generate some samples
        System.out.println("Generated samples:");
        double[][] samples = wbn.sample(5);
        for (int i = 0; i < samples.length; i++) {
            System.out.printf("Sample %d: (%.4f, %.4f)%n", i + 1, samples[i][0], samples[i][1]);
        }
        System.out.println();
        
        // Display circular statistics
        System.out.printf("Circular mean 1: %.4f%n", wbn.circularMean1());
        System.out.printf("Circular mean 2: %.4f%n", wbn.circularMean2());
        System.out.printf("Circular variance 1: %.4f%n", wbn.circularVariance1());
        System.out.printf("Circular variance 2: %.4f%n", wbn.circularVariance2());
    }
}
